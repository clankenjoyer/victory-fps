//! Dispatch capture (diagnostics builds): one compute dispatch written as a
//! directory `victory replay` runs on its own (see
//! `victory_shader::replay`), so a GPU crash of a translated compute shader
//! can be retried in seconds, and with a newer translator.
//!
//! `VICTORY_CAPTURE_DISPATCH=<hash>[,<n>]` captures the n-th dispatch (default the
//! first) of the compute shader whose container hash (as logged) starts
//! with `<hash>`, into `VICTORY_CAPTURE_DIR` (default `captures`). With
//! `VICTORY_SHADER_DUMP` set, the console container is copied along
//! (`victory replay --retranslate` needs it).
//!
//! Recorded when the dispatch is recorded: the translated DXIL and its
//! bindings, the title's compute root arguments, and the descriptors its
//! tables hold (the entries the shader names plus `VICTORY_CAPTURE_SPAN`
//! more, default 16; `all`: every slot of the view heap), as host views at
//! the host heap indices the game uses. Recorded when the list is submitted
//! (after the submission's refreshes): the bytes of every buffer those
//! views and root views name, from the buffers' host copies.
//!
//! shortcut: buffer bytes are read on the CPU at submission, so what the
//! same list's earlier dispatches (or still-running submissions) write is
//! not in them; texture contents are not captured (replayed as zeroed
//! textures of the same description).
//!
//! `VICTORY_CAPTURE_ON_FAIL=<dir>` captures without knowing the shader:
//! every compute dispatch keeps its plan (root arguments, the descriptors
//! its tables hold, its groups; no bytes) per host list, for the lists of
//! the last few submissions. When the device is removed, the dispatches
//! the device-removed breadcrumbs stopped among (the one the list stopped
//! at, then the others between the same barriers) are written under
//! `<dir>/fail_<time>/`, their buffer bytes read from title memory, which
//! outlives the device. It turns device-removed data on (breadcrumb markers
//! tie commands to dispatches).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use victory_shader::replay::{Buffer, Capture, Groups, Layout, Root, Slot};
use victory_shader::{ResourceType, RootType};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::View;
use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;
use crate::shadow::Shadow;

/// A title root argument (`draw.rs`'s `Root`).
pub(crate) enum TitleRoot {
    None,
    Table(u64),
    Constants(Vec<u32>),
    View(u64),
}

pub(crate) enum TitleGroups {
    Direct([u32; 3]),
    /// Records `stride` bytes apart at title address `arguments`.
    Indirect { arguments: u64, max: u32 },
}

struct Want {
    hash: String,
    nth: u32,
    /// `every:<k>`: every k-th dispatch (the n-th is ignored), up to
    /// [`MAX_CAPTURES`] of them.
    every: Option<u32>,
    seen: AtomicU32,
    /// The occurrence the last capture took (its directory's number).
    taken: AtomicU32,
    captures: AtomicU32,
}

/// Captures `every:<k>` takes at most.
const MAX_CAPTURES: u32 = 64;

fn want() -> Option<&'static Want> {
    static WANT: OnceLock<Option<Want>> = OnceLock::new();
    WANT.get_or_init(|| {
        let v = std::env::var("VICTORY_CAPTURE_DISPATCH").ok()?;
        let (hash, nth, every) = match v.split_once(',') {
            Some((h, n)) => match n.trim().strip_prefix("every:") {
                Some(k) => (h, 1, Some(k.trim().parse::<u32>().unwrap_or(100).max(1))),
                None => (h, n.trim().parse().unwrap_or(1), None),
            },
            None => (v.as_str(), 1, None),
        };
        let hash = hash.trim().to_lowercase();
        if hash.is_empty() {
            return None;
        }
        tracing::warn!(%hash, nth, ?every, "VICTORY_CAPTURE_DISPATCH: dispatches of this shader will be captured");
        Some(Want {
            hash,
            nth: nth.max(1),
            every,
            seen: AtomicU32::new(0),
            taken: AtomicU32::new(0),
            captures: AtomicU32::new(0),
        })
    })
    .as_ref()
}

/// Whether this dispatch of the compute shader `hash` is the one to
/// capture (counts the shader's dispatches).
pub(crate) fn wanted(hash: &str) -> bool {
    let Some(w) = want() else { return false };
    if !hash.starts_with(&w.hash) {
        return false;
    }
    let n = w.seen.fetch_add(1, Ordering::Relaxed) + 1;
    let take = match w.every {
        Some(k) => n % k == 1 % k && w.captures.fetch_add(1, Ordering::Relaxed) < MAX_CAPTURES,
        None => n == w.nth,
    };
    if take {
        w.taken.store(n, Ordering::Relaxed);
    }
    take
}

/// Where a captured buffer's bytes come from at submission.
enum Source {
    /// A title buffer's host copy (bytes from its start).
    Shadow(Arc<Shadow>),
    /// Title memory from this address (what the game copies for constant
    /// buffers made per draw).
    Title(u64),
}

struct Buf {
    source: Source,
    size: u64,
    lo: u64,
    hi: u64,
    title: u64,
}

#[derive(Default)]
struct Builder {
    cap: Capture,
    bufs: Vec<Buf>,
    by_shadow: HashMap<usize, u32>,
    slots: HashMap<u32, Slot>,
}

impl Builder {
    /// The captured buffer holding title `address`, with `[address, +len)`
    /// to be read: its id and `address`'s offset in it.
    fn shadow_range(&mut self, address: u64, len: u64) -> Option<(u32, u64)> {
        let s = crate::shadow::find(address, len.max(1))?;
        let offset = s.offset(address);
        let size = offset + s.left(address);
        let key = Arc::as_ptr(&s) as usize;
        let id = match self.by_shadow.get(&key) {
            Some(&id) => id,
            None => {
                let id = self.bufs.len() as u32;
                self.bufs.push(Buf { source: Source::Shadow(s), size, lo: u64::MAX, hi: 0, title: address - offset });
                self.by_shadow.insert(key, id);
                id
            }
        };
        let b = &mut self.bufs[id as usize];
        let end = (offset + len).min(b.size);
        if end > offset {
            b.lo = b.lo.min(offset);
            b.hi = b.hi.max(end);
        }
        Some((id, offset))
    }

    fn title_blob(&mut self, address: u64, len: u64) -> u32 {
        let id = self.bufs.len() as u32;
        self.bufs.push(Buf { source: Source::Title(address), size: len, lo: 0, hi: len, title: address });
        id
    }

    /// A buffer view as `descriptor/host.rs` makes it (`buffer_srv`,
    /// `buffer_uav`): the first element moved into the host copy, the count
    /// clamped to it.
    fn buffer_view(&mut self, uav: bool, address: u64, len: u64, words: &[u32; 10]) -> Slot {
        let Some(s) = crate::shadow::find(address, len) else {
            return Slot::Skipped(format!("buffer view of {address:#x}+{len:#x} without a host copy (null in the game too)"));
        };
        let mut w = *words;
        w[0] = crate::resource::pc_format(w[0]);
        let offset = s.offset(address);
        let room = offset + s.left(address);
        // (first element, count, element size) of the host view.
        let (first, count, element) = if uav {
            // SAFETY: 40 bytes in the host's layout.
            let mut d = unsafe { w.as_ptr().cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>().read_unaligned() };
            // SAFETY: a buffer view's union member.
            let b = unsafe { &mut d.Anonymous.Buffer };
            let element = if b.StructureByteStride > 0 {
                u64::from(b.StructureByteStride)
            } else if b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
                4
            } else {
                crate::resource::element(d.Format.0 as u32).map_or(0, |e| e.0)
            };
            if element == 0 || offset % element != 0 {
                return Slot::Skipped("unordered buffer view not on an element of its host copy (null in the game too)".into());
            }
            b.FirstElement += offset / element;
            b.NumElements = b.NumElements.min((room / element).saturating_sub(b.FirstElement).min(u64::from(u32::MAX)) as u32);
            b.CounterOffsetInBytes = 0;
            let r = (b.FirstElement, b.NumElements, element);
            // SAFETY: as above.
            unsafe { w.as_mut_ptr().cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>().write_unaligned(d) };
            r
        } else {
            // SAFETY: 40 bytes in the host's layout.
            let mut d = unsafe { w.as_ptr().cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>().read_unaligned() };
            // SAFETY: a buffer view's union member.
            let b = unsafe { &mut d.Anonymous.Buffer };
            let element = if b.StructureByteStride > 0 {
                u64::from(b.StructureByteStride)
            } else if b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
                4
            } else {
                crate::resource::element(d.Format.0 as u32).map_or(0, |e| e.0)
            };
            if element == 0 || offset % element != 0 {
                return Slot::Skipped("buffer view not on an element of its host copy (null in the game too)".into());
            }
            b.FirstElement += offset / element;
            b.NumElements = b.NumElements.min((room / element).saturating_sub(b.FirstElement).min(u64::from(u32::MAX)) as u32);
            let r = (b.FirstElement, b.NumElements, element);
            // SAFETY: as above.
            unsafe { w.as_mut_ptr().cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>().write_unaligned(d) };
            r
        };
        let start = address - offset;
        let Some((id, _)) = self.shadow_range(start + first * element, u64::from(count) * element) else {
            return Slot::Skipped("buffer view's host copy vanished".into());
        };
        if uav { Slot::Uav { buffer: id, desc: w } } else { Slot::Srv { buffer: id, desc: w } }
    }

    /// The host view the game makes from title view `view`.
    fn slot(&mut self, view: Option<View>) -> Slot {
        let pc = |d: &[u32; 10]| {
            let mut d = *d;
            d[0] = crate::resource::pc_format(d[0]);
            d
        };
        let texture = |t: &ID3D12Resource| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            [
                d.Dimension.0 as u64,
                d.Width,
                u64::from(d.Height),
                u64::from(d.DepthOrArraySize),
                u64::from(d.MipLevels),
                d.Format.0 as u64,
                u64::from(d.SampleDesc.Count),
                d.Flags.0 as u64,
            ]
        };
        match view {
            None => Slot::Null,
            Some(View::Constant(address, size)) => {
                Slot::Cbv { buffer: self.title_blob(address, u64::from(size)), offset: 0, size }
            }
            Some(View::Shader(r, desc)) => {
                let r = r as This;
                if let Some(t) = crate::resource::host_resource(r) {
                    Slot::TextureSrv { desc: desc.as_ref().map(pc), texture: texture(t) }
                } else if let (Some((data, len)), Some(desc)) = (crate::resource::buffer_bytes(r), desc) {
                    self.buffer_view(false, data as u64, len, &desc)
                } else {
                    Slot::Skipped("shader view without a host resource".into())
                }
            }
            Some(View::Unordered(r, _, desc)) => {
                let r = r as This;
                if let Some(t) = crate::resource::host_resource(r) {
                    Slot::TextureUav { desc: desc.as_ref().map(pc), texture: texture(t) }
                } else if let (Some((data, len)), Some(desc)) = (crate::resource::buffer_bytes(r), desc) {
                    self.buffer_view(true, data as u64, len, &desc)
                } else {
                    Slot::Skipped("unordered view without a host resource".into())
                }
            }
            Some(v @ View::Placed(..)) => {
                if let Some(t) = crate::descriptor::host::placed_texture(&v) {
                    return self.slot(Some(t));
                }
                let View::Placed(uav, resource, desc) = v else { unreachable!() };
                if resource[0] != 1 {
                    return Slot::Skipped("placed unordered texture view (not mirrored by the game either)".into());
                }
                let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let words: [u32; 10] = desc[..10].try_into().unwrap_or([0; 10]);
                self.buffer_view(uav, address, len, &words)
            }
            Some(View::Sampler(_) | View::ConsoleSampler(_)) => Slot::Skipped("a sampler in the view heap".into()),
        }
    }
}

fn sampler_words(view: Option<View>) -> Option<[u32; 13]> {
    let desc = match view? {
        View::Sampler(w) => return Some(w),
        View::ConsoleSampler(w) => crate::sampler::console(&w),
        _ => return None,
    };
    // SAFETY: a 52-byte description.
    Some(unsafe { std::ptr::from_ref(&desc).cast::<[u32; 13]>().read_unaligned() })
}

fn span() -> Option<u32> {
    match std::env::var("VICTORY_CAPTURE_SPAN") {
        Ok(v) if v.trim() == "all" => None,
        Ok(v) => Some(v.trim().parse().unwrap_or(16)),
        Err(_) => Some(16),
    }
}

/// A root argument as the capture will hold it; root views are placed in
/// captured buffers when the capture is built.
#[derive(Clone)]
enum PlanRoot {
    Done(Root),
    /// A root view at this title address, read by the shader as an SRV
    /// (`srv`) or a CBV, or neither.
    View { address: u64, srv: bool, cbv: bool },
}

/// What a dispatch's capture reads, worked out from its root arguments
/// and the title's heaps without touching any buffer: the descriptors
/// (title handles) and the host heap slots they go to.
#[derive(Clone, Default)]
struct Plan {
    roots: Vec<(u8, PlanRoot)>,
    /// (host slot, sampler, title handle).
    reads: Vec<(u32, bool, u64)>,
    notes: Vec<String>,
    /// `VICTORY_CAPTURE_SPAN=all`: the view heap (start, slots) too.
    all: Option<(u64, u32)>,
}

fn plan(stage: &crate::pipeline::host::Stage, roots: &[TitleRoot], heaps: [usize; 2], span: Option<u32>) -> Plan {
    let t = stage.translated();
    let mut plan = Plan::default();
    let title_heaps: Vec<&crate::descriptor::Heap> = heaps
        .iter()
        .filter(|&&h| h != 0)
        .map(|&h| crate::descriptor::heap(h as This))
        .collect();
    let heap_of = |handle: u64| title_heaps.iter().copied().find(|h| h.index(handle).is_some());
    // Fresh tables (the game's dynamic slots) go from the host heap's
    // title slots on, as the game's do.
    let mut fresh_next: Option<u32> = None;
    for (slot, root) in roots.iter().enumerate() {
        let slot = slot as u8;
        let r = match root {
            TitleRoot::None => PlanRoot::Done(Root::None),
            TitleRoot::Constants(c) => PlanRoot::Done(Root::Constants(c.clone())),
            TitleRoot::View(address) => PlanRoot::View {
                address: *address,
                srv: stage.root_srvs.iter().any(|&(s, _)| s == slot),
                cbv: stage.root_cbvs.iter().any(|&(s, _)| s == slot),
            },
            TitleRoot::Table(handle) => {
                let Some(heap) = heap_of(*handle) else {
                    plan.notes.push(format!("root {slot}: table {handle:#x} outside the bound heaps"));
                    continue;
                };
                let sampler = heap.samplers();
                let step: u64 = if sampler { 16 } else { 32 };
                let first = heap.index(*handle).unwrap_or(0);
                let n_used = t
                    .used
                    .iter()
                    .filter(|u| u.root == RootType::Table && u.slot == slot)
                    .filter(|u| (u.kind == ResourceType::Sampler) == sampler)
                    .map(|u| (u64::from(u.table_offset_dwords) * 4 / step) as u32)
                    .max()
                    .map_or(0, |m| m + 1);
                if n_used == 0 && span.is_some() {
                    plan.roots.push((slot, PlanRoot::Done(Root::Table { sampler, offset: (*handle - heap.start) as u32 })));
                    continue;
                }
                let count = heap.host().map_or(u32::MAX, |h| h.count());
                let fresh = !sampler && first + n_used > count;
                let n = match span {
                    Some(extra) => n_used + extra,
                    // `all`: the table to the heap's end.
                    None => u32::MAX,
                };
                let base = if fresh {
                    let at = *fresh_next.get_or_insert(count);
                    fresh_next = Some(at + n_used);
                    at
                } else {
                    first
                };
                let mut k = 0u32;
                while k < n {
                    let at = *handle + u64::from(k) * step;
                    let Some(index) = heap.index(at) else { break };
                    if (!sampler && !fresh && index >= count) || (fresh && k >= n_used) {
                        break;
                    }
                    plan.reads.push((base + k, sampler, at));
                    k += 1;
                }
                let offset = if fresh { base * 32 } else { (*handle - heap.start) as u32 };
                PlanRoot::Done(Root::Table { sampler, offset })
            }
        };
        plan.roots.push((slot, r));
    }
    if span.is_none()
        && let Some(heap) = title_heaps.iter().find(|h| !h.samplers())
    {
        plan.all = Some((heap.start, heap.host().map_or(0, |h| h.count())));
    }
    plan
}

/// The capture of a planned dispatch: `views[i]` is what `plan.reads[i]`
/// held when the dispatch was recorded; `all`, other slots of the heap.
fn build(shader: &crate::pipeline::Shader, plan: &Plan, views: Vec<Option<View>>, all: Vec<(u32, View)>) -> Option<Builder> {
    let t = shader.translated()?;
    let mut b = Builder::default();
    b.cap.shader = shader.hash.clone();
    b.cap.dxil = "shader.dxil".into();
    b.cap.layout = Layout::of(&t.host);
    b.cap.guard = victory_shader::emit::view_heap_guard();
    b.cap.strict_kinds = std::env::var("VICTORY_STRICT_KINDS").map_or(true, |v| v != "0");
    b.cap.root_srv_tables = crate::pipeline::host::bounded_root_srvs();
    b.cap.notes.extend(plan.notes.iter().cloned());
    if !t.host.root_uavs.is_empty() {
        // shortcut: a root UAV's buffer is not captured (its root reads as
        // unbacked) and the replay refuses the shader.
        b.cap.notes.push(format!("root UAVs {:?}: not captured", t.host.root_uavs));
    }
    if t.host.address_map || t.host.descriptor_words {
        // shortcut: neither the address map (every host copy of title
        // memory) nor the heap's descriptor words are captured; the
        // layout lists them and the replay refuses the shader.
        b.cap
            .notes
            .push("the address map or the descriptor words: not captured".into());
    }
    for (slot, r) in &plan.roots {
        let r = match r {
            PlanRoot::Done(r) => r.clone(),
            &PlanRoot::View { address, srv, cbv } => {
                let len = match crate::shadow::find(address, 1) {
                    Some(s) => s.left(address).min(if srv { 64 << 20 } else { 1 << 16 }),
                    None => 0,
                };
                match (srv || cbv).then(|| b.shadow_range(address, len)).flatten() {
                    Some((buffer, offset)) => Root::View { buffer, offset },
                    None => Root::Unbacked(address),
                }
            }
        };
        b.cap.roots.push((*slot, r));
    }
    for (&(placed, sampler, _), view) in plan.reads.iter().zip(views) {
        if sampler {
            if let Some(w) = sampler_words(view) {
                b.cap.samplers.push((placed, w));
            }
        } else if !b.slots.contains_key(&placed) {
            let s = b.slot(view);
            b.slots.insert(placed, s);
        }
    }
    for (i, view) in all {
        if !b.slots.contains_key(&i) {
            let s = b.slot(Some(view));
            b.slots.insert(i, s);
        }
    }
    let mut views: Vec<(u32, Slot)> = std::mem::take(&mut b.slots).into_iter().collect();
    views.sort_by_key(|(i, _)| *i);
    b.cap.views = views;
    b.cap.samplers.sort_by_key(|(i, _)| *i);
    b.cap.samplers.dedup_by_key(|(i, _)| *i);
    Some(b)
}

/// A fresh directory `name` (stamped if taken) under `dir` holding the
/// translated DXIL and, from `VICTORY_SHADER_DUMP`, the console container
/// (`--retranslate` input).
fn start_dir(dir: &std::path::Path, name: &str, shader: &crate::pipeline::Shader, b: &mut Builder) -> Option<std::path::PathBuf> {
    let mut out = dir.join(name);
    if out.exists() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        out = dir.join(format!("{name}_{stamp}"));
    }
    let t = shader.translated()?;
    let made = std::fs::create_dir_all(&out).and_then(|()| std::fs::write(out.join("shader.dxil"), &t.host.dxil));
    if let Err(e) = made {
        tracing::warn!(error = %e, dir = %out.display(), "VICTORY_CAPTURE: cannot write the capture");
        return None;
    }
    if let Some(dump) = std::env::var_os("VICTORY_SHADER_DUMP") {
        let bin = std::path::Path::new(&dump).join(format!("compute_{}.bin", shader.hash));
        if std::fs::copy(&bin, out.join("shader.bin")).is_ok() {
            b.cap.container = Some("shader.bin".into());
        }
    }
    if b.cap.container.is_none() {
        b.cap.notes.push(format!(
            "no container (set VICTORY_SHADER_DUMP, or pass --container compute_{}.bin to --retranslate)",
            shader.hash
        ));
    }
    Some(out)
}

/// Captures this dispatch (call where it is recorded, after `bind`).
pub(crate) fn dispatch(
    list: &HostList,
    p: &HostPipeline,
    roots: &[TitleRoot],
    heaps: [usize; 2],
    groups: TitleGroups,
) {
    let Some(stage) = p.stages.first() else { return };
    let plan = plan(stage, roots, heaps, span());
    let views = plan.reads.iter().map(|&(_, _, at)| crate::descriptor::view(at)).collect();
    let all = plan.all.map_or_else(Vec::new, |(start, count)| {
        (0..count)
            .map_while(|i| {
                let at = start + u64::from(i) * 32;
                Some((i, crate::descriptor::view(at)))
            })
            .filter_map(|(i, v)| Some((i, v?)))
            .collect()
    });
    let Some(mut b) = build(&stage.shader, &plan, views, all) else { return };
    b.cap.notes.push(format!("pipeline {}", p.name));
    b.cap.notes.push("buffer bytes read on the CPU at submission: writes of earlier work in the same list are not in them".into());
    let indirect = set_groups(&mut b, &groups);
    let dir = std::path::PathBuf::from(std::env::var_os("VICTORY_CAPTURE_DIR").unwrap_or_else(|| "captures".into()));
    let nth = want().map_or(1, |w| w.taken.load(Ordering::Relaxed));
    let hash = &stage.shader.hash;
    let Some(out) = start_dir(&dir, &format!("{hash}_{nth}"), &stage.shader, &mut b) else { return };
    tracing::warn!(dir = %out.display(), views = b.cap.views.len(), buffers = b.bufs.len(), "VICTORY_CAPTURE: dispatch captured; its buffers are written at submission");
    let indirect = indirect.and_then(|a| crate::shadow::find(a, 12).map(|s| Args::Shadow(s.clone(), s.offset(a))));
    // The bytes once the list is submitted, after its refreshes.
    let pending = Mutex::new(Some((b, out, indirect)));
    list.defer(move || {
        let Some((b, out, indirect)) = pending.lock().unwrap_or_else(|e| e.into_inner()).take() else {
            return;
        };
        write(b, &out, indirect, false);
    });
}

/// The capture's group counts; an indirect dispatch's title argument
/// address, whose first record is read when the bytes are.
fn set_groups(b: &mut Builder, groups: &TitleGroups) -> Option<u64> {
    match *groups {
        TitleGroups::Direct(g) => {
            b.cap.groups = Groups::Direct(g);
            None
        }
        TitleGroups::Indirect { arguments, max } => {
            b.cap.groups = Groups::Indirect { args: None, max };
            Some(arguments)
        }
    }
}

/// Where an indirect dispatch's first record is read.
enum Args {
    /// In a host copy, at this offset.
    Shadow(Arc<Shadow>, u64),
    /// In title memory.
    Title(u64),
}

/// Writes the buffers' bytes and the manifest: from the host copies, or
/// with `title` from title memory (after a device removal, when the host
/// copies may hold what the faulting work half wrote).
fn write(mut b: Builder, out: &std::path::Path, indirect: Option<Args>, title: bool) {
    let mut budget: u64 = std::env::var("VICTORY_CAPTURE_MAX_MB")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(2048)
        << 20;
    let read_shadow = |s: &Shadow, lo: u64, hi: u64| -> Option<Vec<u8>> {
        let mut p = std::ptr::null_mut();
        // SAFETY: host copies are mapped write-back memory; reading a range
        // inside the buffer.
        unsafe {
            s.buffer.Map(0, None, Some(&mut p)).ok()?;
            let v = std::slice::from_raw_parts(p.cast::<u8>().add(lo as usize), (hi - lo) as usize).to_vec();
            s.buffer.Unmap(0, None);
            Some(v)
        }
    };
    let read_title = |address: u64, len: u64| -> Option<Vec<u8>> {
        let mut v = vec![0; len as usize];
        crate::host::read_into(address as *const u8, &mut v).then_some(v)
    };
    if let (Groups::Indirect { max, .. }, Some(a)) = (&b.cap.groups, &indirect) {
        let bytes = match a {
            Args::Shadow(s, offset) => read_shadow(s, *offset, offset + 12),
            Args::Title(address) => read_title(*address, 12),
        };
        let args = bytes.map(|v| {
            std::array::from_fn(|k| u32::from_le_bytes(v[4 * k..4 * k + 4].try_into().unwrap_or([0; 4])))
        });
        b.cap.groups = Groups::Indirect { args, max: *max };
    }
    for (id, buf) in b.bufs.iter().enumerate() {
        let (lo, mut hi) = if buf.lo == u64::MAX { (0, 0) } else { (buf.lo, buf.hi) };
        if hi - lo > budget {
            b.cap.notes.push(format!("buffer {id}: {:#x} bytes cut to {budget:#x} (VICTORY_CAPTURE_MAX_MB)", hi - lo));
            hi = lo + budget;
        }
        budget -= hi - lo;
        let bytes = match &buf.source {
            Source::Shadow(s) if !title => read_shadow(s, lo, hi).unwrap_or_else(|| {
                b.cap.notes.push(format!("buffer {id}: host copy not readable; zeros"));
                vec![0; (hi - lo) as usize]
            }),
            // A host copy's title buffer starts at `buf.title`.
            Source::Shadow(_) | Source::Title(_) => {
                let start = match &buf.source {
                    Source::Title(address) => *address,
                    Source::Shadow(_) => buf.title,
                };
                read_title(start + lo, hi - lo).unwrap_or_else(|| {
                    b.cap.notes.push(format!("buffer {id}: title memory not readable; zeros"));
                    vec![0; (hi - lo) as usize]
                })
            }
        };
        let file = format!("buffer{id}.bin");
        if let Err(e) = std::fs::write(out.join(&file), &bytes) {
            tracing::warn!(error = %e, "VICTORY_CAPTURE: buffer not written");
        }
        b.cap.buffers.push(Buffer { id: id as u32, size: buf.size, lo, hi, file, title: buf.title });
    }
    match std::fs::write(out.join(victory_shader::replay::MANIFEST), b.cap.to_text()) {
        Ok(()) => tracing::warn!(dir = %out.display(), "VICTORY_CAPTURE: capture written; run `victory replay` on it"),
        Err(e) => tracing::warn!(error = %e, "VICTORY_CAPTURE: manifest not written"),
    }
}

// Capture on failure (`VICTORY_CAPTURE_ON_FAIL=<dir>`): every compute
// dispatch's plan and descriptors are kept per host list (no buffer
// bytes); when the device is removed, the dispatches the breadcrumbs name
// are written, their buffers from title memory.

/// `VICTORY_CAPTURE_ON_FAIL`: where captures of a failure go.
pub(crate) fn on_fail_dir() -> Option<&'static std::path::Path> {
    static DIR: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let d = std::env::var_os("VICTORY_CAPTURE_ON_FAIL").filter(|d| !d.is_empty())?;
        tracing::warn!(
            dir = ?d,
            "VICTORY_CAPTURE_ON_FAIL: compute dispatches are recorded; a device removal writes captures of those it stopped among (device-removed data is on)"
        );
        Some(d.into())
    })
    .as_deref()
}

/// What a recorded dispatch needs for a capture, but buffer bytes.
pub(crate) struct Recorded {
    shader: Arc<crate::pipeline::Shader>,
    plan: Plan,
    views: Vec<Option<View>>,
    groups: TitleGroups,
}

/// Entries recorded per host list (by list pointer), keyed by the index of
/// the list's note (breadcrumb marker) each follows. A list's entries go
/// when it is recorded again (its notes restart), and lists not recorded
/// again within [`KEEP`] recordings of others go too.
pub(crate) struct Log<T> {
    lists: HashMap<usize, Entries<T>>,
    /// List recordings started (of any list).
    clock: u64,
}

struct Entries<T> {
    /// The clock when the list's recording started.
    started: u64,
    entries: Vec<(u32, T)>,
}

/// List recordings a list's entries outlive its own by: the lists of the
/// last few submissions.
const KEEP: u64 = 256;

impl<T> Default for Log<T> {
    fn default() -> Self {
        Log { lists: HashMap::new(), clock: 0 }
    }
}

impl<T> Log<T> {
    /// Records `value` as following note `note` of `list`.
    pub(crate) fn record(&mut self, list: usize, note: u32, value: T) {
        let restarted = self
            .lists
            .get(&list)
            .is_none_or(|e| e.entries.last().is_some_and(|&(last, _)| note <= last));
        if restarted {
            self.clock += 1;
            let clock = self.clock;
            if clock.is_multiple_of(64) {
                self.lists.retain(|_, e| clock - e.started <= KEEP);
            }
            let e = self.lists.entry(list).or_insert_with(|| Entries { started: clock, entries: Vec::new() });
            e.started = clock;
            e.entries.clear();
        }
        if let Some(e) = self.lists.get_mut(&list) {
            e.entries.push((note, value));
        }
    }

    /// The entry recorded after note `note` of `list`.
    pub(crate) fn find(&self, list: usize, note: u32) -> Option<&T> {
        let e = self.lists.get(&list)?;
        let i = e.entries.binary_search_by_key(&note, |&(n, _)| n).ok()?;
        Some(&e.entries[i].1)
    }

    /// The last entry of `list`.
    pub(crate) fn last(&self, list: usize) -> Option<(u32, &T)> {
        self.lists.get(&list)?.entries.last().map(|(n, v)| (*n, v))
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.lists.values().map(|e| e.entries.len()).sum()
    }
}

fn log() -> &'static Mutex<Log<Arc<Recorded>>> {
    static LOG: OnceLock<Mutex<Log<Arc<Recorded>>>> = OnceLock::new();
    LOG.get_or_init(Mutex::default)
}

/// Records a dispatch for a capture on failure (call after its `note_op`,
/// with [`on_fail_dir`] set): its plan and the descriptors it reads, no
/// buffer bytes.
pub(crate) fn record(list: &HostList, p: &HostPipeline, roots: &[TitleRoot], heaps: [usize; 2], groups: TitleGroups) {
    let (Some(stage), Some(note)) = (p.stages.first(), list.note_count().checked_sub(1)) else { return };
    // Never the whole heap per dispatch.
    let plan = plan(stage, roots, heaps, Some(span().unwrap_or(16)));
    let views = plan.reads.iter().map(|&(_, _, at)| crate::descriptor::view(at)).collect();
    let r = Arc::new(Recorded { shader: stage.shader.clone(), plan, views, groups });
    let key = windows::core::Interface::as_raw(&list.list) as usize;
    log().lock().unwrap_or_else(|e| e.into_inner()).record(key, note, r);
}

/// The commands a capture on failure takes from a list's breadcrumbs, as
/// (command, note index): dispatches (direct or indirect) right after a
/// marker, between the barriers around the command the list stopped at
/// (`done` commands done), the first at or after it first.
pub(crate) fn stopped_dispatches(history: &[D3D12_AUTO_BREADCRUMB_OP], done: usize) -> Vec<(usize, u32)> {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let done = done.min(history.len());
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1))
        .min(history.len());
    let marker = D3D12_AUTO_BREADCRUMB_OP_SETMARKER;
    let mut markers = history[..from].iter().filter(|&&op| op == marker).count() as u32;
    let mut out: Vec<(usize, u32)> = Vec::new();
    for i in from..to {
        let op = history[i];
        if op == marker {
            markers += 1;
            continue;
        }
        let dispatch = op == D3D12_AUTO_BREADCRUMB_OP_DISPATCH || op == D3D12_AUTO_BREADCRUMB_OP_EXECUTEINDIRECT;
        if dispatch && i > 0 && history[i - 1] == marker {
            out.push((i, markers - 1));
        }
    }
    if let Some(k) = out.iter().position(|&(i, _)| i >= done) {
        out[..=k].rotate_right(1);
    }
    out
}

/// Captures at most this many dispatches per failed list.
const MAX_FAIL_CAPTURES: usize = 8;

/// After a device removal: writes captures of the dispatches `list`
/// stopped among (`history` and `done` from its breadcrumbs; `notes`, its
/// note texts), or of its last dispatch without breadcrumbs. Returns text
/// for the failure report.
pub(crate) fn on_failure(list: usize, history: Option<(&[D3D12_AUTO_BREADCRUMB_OP], usize)>, notes: &[&str]) -> String {
    let Some(root) = on_fail_dir() else { return String::new() };
    let found: Vec<(Option<usize>, u32, Arc<Recorded>)> = {
        let log = log().lock().unwrap_or_else(|e| e.into_inner());
        let wanted: Vec<(Option<usize>, u32)> = match history {
            Some((h, done)) => stopped_dispatches(h, done).into_iter().map(|(i, n)| (Some(i), n)).collect(),
            None => log.last(list).map(|(n, _)| (None, n)).into_iter().collect(),
        };
        wanted
            .iter()
            .filter_map(|&(cmd, note)| {
                let r = log.find(list, note)?;
                // The note must be this dispatch's: a list recorded again
                // without dispatches still has the old recording's.
                notes
                    .get(note as usize)
                    .is_some_and(|t| t.contains(&r.shader.hash))
                    .then(|| (cmd, note, r.clone()))
            })
            .take(MAX_FAIL_CAPTURES)
            .collect()
    };
    if found.is_empty() {
        return "; VICTORY_CAPTURE_ON_FAIL: no recorded compute dispatch where the list stopped".into();
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let dir = root.join(format!("fail_{stamp}"));
    let mut written = Vec::new();
    for (k, (cmd, note, r)) in found.iter().enumerate() {
        let Some(mut b) = build(&r.shader, &r.plan, r.views.clone(), Vec::new()) else { continue };
        let indirect = set_groups(&mut b, &r.groups).map(Args::Title);
        let what = match (cmd, history) {
            (Some(c), Some((_, done))) if k == 0 && *c >= done => format!("command {c}, where the list stopped ({done} done)"),
            (Some(c), Some((_, done))) => format!("command {c}, beside command {done} where the list stopped"),
            _ => "the list's last dispatch (no breadcrumbs named a command)".into(),
        };
        let text = notes.get(*note as usize).copied().unwrap_or("");
        b.cap.notes.push(format!("captured on device removal: {what}; its note: {text}"));
        b.cap.notes.push(
            "buffer bytes are the title's memory as of the failure (not the host copies the GPU read; later work of the title may be in them)".into(),
        );
        let name = match cmd {
            Some(c) => format!("{}cmd{c:05}_{}", if k == 0 { "stopped_" } else { "" }, r.shader.hash),
            None => format!("last_{}", r.shader.hash),
        };
        let Some(out) = start_dir(&dir, &name, &r.shader, &mut b) else { continue };
        write(b, &out, indirect, true);
        written.push(name);
    }
    format!("; VICTORY_CAPTURE_ON_FAIL: captures {written:?} in {}", dir.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_log_keeps_each_lists_last_recording() {
        let mut log = Log::default();
        log.record(1, 3, "a3");
        log.record(1, 7, "a7");
        log.record(2, 0, "b0");
        assert_eq!(log.find(1, 7), Some(&"a7"));
        assert_eq!(log.find(1, 5), None);
        assert_eq!(log.last(1), Some((7, &"a7")));
        // List 1 recorded again: its notes restart.
        log.record(1, 2, "c2");
        assert_eq!(log.find(1, 3), None);
        assert_eq!(log.find(1, 2), Some(&"c2"));
        assert_eq!(log.find(2, 0), Some(&"b0"));
        assert_eq!(log.len(), 2);
        // Lists not recorded again for a while go.
        for k in 0..2 * KEEP {
            log.record(100 + k as usize, 0, "x");
        }
        assert_eq!(log.find(2, 0), None);
        assert_eq!(log.find(1, 2), None);
        assert!(log.len() as u64 <= KEEP + 64);
        assert_eq!(log.find(100 + 2 * KEEP as usize - 1, 0), Some(&"x"));
    }

    /// The bookkeeping of a recorded dispatch without the title's heaps: a
    /// global lock per descriptor read (as `descriptor::view`), 40 views
    /// cloned, the plan's vectors, the log's lock and entry.
    /// `cargo test --release -p victory-gpu --features diagnostics -- --ignored record_cost --nocapture`
    #[test]
    #[ignore = "timing"]
    fn record_cost() {
        let views_lock = Mutex::new(vec![View::Shader(1, Some([7; 10])); 64]);
        type Entry = Arc<(Plan, Vec<Option<View>>, Vec<TitleRoot>)>;
        let log: Mutex<Log<Entry>> = Mutex::default();
        let n = 200_000u32;
        let started = std::time::Instant::now();
        for k in 0..n {
            let roots = vec![TitleRoot::Constants(vec![1, 2, 3, 4]), TitleRoot::Table(0x1000), TitleRoot::Table(0x2000)];
            let mut plan = Plan::default();
            plan.roots.push((0, PlanRoot::Done(Root::Constants(vec![1, 2, 3, 4]))));
            plan.roots.push((1, PlanRoot::Done(Root::Table { sampler: false, offset: 0x40 })));
            plan.reads.extend((0..40).map(|i| (i, false, 0x1000 + u64::from(i) * 32)));
            let views: Vec<Option<View>> = plan
                .reads
                .iter()
                .map(|&(i, _, _)| views_lock.lock().unwrap().get(i as usize).cloned())
                .collect();
            log.lock().unwrap().record((k / 500) as usize % 8, k % 500, Arc::new((plan, views, roots)));
        }
        let ns = started.elapsed().as_nanos() / u128::from(n);
        eprintln!("{ns} ns per recorded dispatch (40 descriptors)");
    }

    #[test]
    fn breadcrumbs_name_the_stopped_dispatch_first() {
        use D3D12_AUTO_BREADCRUMB_OP_DISPATCH as D;
        use D3D12_AUTO_BREADCRUMB_OP_EXECUTEINDIRECT as E;
        use D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER as B;
        use D3D12_AUTO_BREADCRUMB_OP_SETMARKER as M;
        let copy = D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION;
        // Notes 0: dispatch | barrier | 1: copy, 2: dispatch, a clamp's
        // dispatch (no note), 3: indirect, 4: dispatch | barrier | 5:
        // dispatch.
        let h = [M, D, B, M, copy, M, D, D, M, E, M, D, B, M, D];
        // Stopped at the indirect one (9): it first, then 6 and 11.
        assert_eq!(stopped_dispatches(&h, 9), vec![(9, 3), (6, 2), (11, 4)]);
        // At the marker before 11.
        assert_eq!(stopped_dispatches(&h, 10), vec![(11, 4), (6, 2), (9, 3)]);
        // At the start; done with all.
        assert_eq!(stopped_dispatches(&h, 0), vec![(1, 0)]);
        assert_eq!(stopped_dispatches(&h, 15), vec![(14, 5)]);
        // At a barrier: the dispatches before it.
        assert_eq!(stopped_dispatches(&h, 12), vec![(6, 2), (9, 3), (11, 4)]);
    }
}
