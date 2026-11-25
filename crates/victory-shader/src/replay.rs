//! Dispatch captures: one compute dispatch of a title, written by
//! `victory-gpu` (diagnostics builds, `VICTORY_CAPTURE`) as a directory with
//! a text manifest (`capture.txt`) and binary blobs, and replayed on its own
//! by `victory replay` ([`run`], host D3D12): a GPU crash of a translated
//! compute shader becomes a test that runs in seconds.
//!
//! The manifest is line based: a keyword, then space-separated fields;
//! numbers are decimal or `0x` hex; `#` starts a comment. Heap slots keep
//! the host heap index the game gave them, so user data holding a table's
//! byte offset needs no patching (tables the game made in dynamic slots are
//! placed by the capture).

use std::fmt::Write as _;

use crate::emit::{HeapGuard, UserData};
use crate::native::RootType;

#[cfg(all(windows, feature = "host"))]
pub mod run;
pub mod suite;

/// The manifest's file name in a capture directory.
pub const MANIFEST: &str = "capture.txt";
pub const VERSION: u32 = 1;

/// A captured dispatch.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Capture {
    /// The console container's hash (16 hex digits, as logged).
    pub shader: String,
    /// File names in the directory: the translated DXIL, and the console
    /// container if the capture had it (`--retranslate` needs one).
    pub dxil: String,
    pub container: Option<String>,
    /// The translation's bindings (what its root signature declares).
    pub layout: Layout,
    /// The view heap guard the translation was made for.
    pub guard: Option<HeapGuard>,
    /// Buffer views serve only reads of their own kind (NVIDIA).
    pub strict_kinds: bool,
    /// Root SRVs are one-view tables (`VICTORY_ROOT_SRV_DIRECT` unset).
    pub root_srv_tables: bool,
    pub groups: Groups,
    /// The title's compute root arguments, by title root parameter.
    pub roots: Vec<(u8, Root)>,
    pub buffers: Vec<Buffer>,
    /// View heap slots (host index) and what they hold.
    pub views: Vec<(u32, Slot)>,
    /// Sampler heap slots: `D3D12_SAMPLER_DESC` words.
    pub samplers: Vec<(u32, [u32; 13])>,
    /// Free text: what the capture could not record.
    pub notes: Vec<String>,
}

/// The bindings of a translated shader ([`crate::emit::HostShader`]
/// without its code).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Layout {
    pub user_data: Vec<UserData>,
    pub samplers: Vec<[u32; 4]>,
    pub root_cbvs: Vec<u8>,
    pub root_srvs: Vec<u8>,
    /// Root UAVs: listed, not captured or replayed yet.
    pub root_uavs: Vec<u8>,
    /// The address map and the descriptor words
    /// ([`crate::emit::HostShader::address_map`]): listed, not captured
    /// or replayed yet.
    pub address_map: bool,
    pub descriptor_words: bool,
}

impl Layout {
    pub fn of(s: &crate::emit::HostShader) -> Layout {
        Layout {
            user_data: s.user_data.clone(),
            samplers: s.samplers.clone(),
            root_cbvs: s.root_cbvs.clone(),
            root_srvs: s.root_srvs.clone(),
            root_uavs: s.root_uavs.clone(),
            address_map: s.address_map,
            descriptor_words: s.descriptor_words,
        }
    }

    /// A compute [`crate::emit::HostShader`] of `dxil` with these bindings.
    pub fn shader(&self, dxil: Vec<u8>) -> crate::emit::HostShader {
        crate::emit::HostShader {
            stage: crate::native::HwStage::Cs,
            dxil,
            user_data: self.user_data.clone(),
            samplers: self.samplers.clone(),
            static_samplers: self
                .samplers
                .iter()
                .map(|&w| crate::emit::StaticSampler::from_words(w))
                .collect(),
            root_cbvs: self.root_cbvs.clone(),
            root_srvs: self.root_srvs.clone(),
            root_uavs: self.root_uavs.clone(),
            address_map: self.address_map,
            descriptor_words: self.descriptor_words,
            root_indexed_views: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Groups {
    Direct([u32; 3]),
    /// An indirect dispatch: the first record's group counts as the CPU saw
    /// them at submission (before the host's clamp), if they could be read.
    Indirect {
        args: Option<[u32; 3]>,
        max: u32,
    },
}

impl Default for Groups {
    fn default() -> Self {
        Groups::Direct([1, 1, 1])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Root {
    None,
    Constants(Vec<u32>),
    /// A descriptor table: the user-data value translated shaders get (the
    /// first slot's byte offset in its heap, 32 bytes per slot).
    Table { sampler: bool, offset: u32 },
    /// A root view: a captured buffer and the byte offset in it.
    View { buffer: u32, offset: u64 },
    /// A root view whose memory has no host copy (title address).
    Unbacked(u64),
}

/// A host buffer: `size` bytes, of which `[lo, hi)` were captured into
/// `file` (the rest replays as zeros).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buffer {
    pub id: u32,
    pub size: u64,
    pub lo: u64,
    pub hi: u64,
    pub file: String,
    /// The title address of byte 0 (for reports).
    pub title: u64,
}

/// A view heap slot. Descriptions are the host's (`D3D12_*_VIEW_DESC` as
/// 10 dwords), with buffer views' first element already in the host buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Slot {
    Null,
    /// A constant buffer view of `size` bytes from `offset` (the game reads
    /// a fresh copy, so no alignment holds).
    Cbv { buffer: u32, offset: u64, size: u32 },
    Srv { buffer: u32, desc: [u32; 10] },
    Uav { buffer: u32, desc: [u32; 10] },
    /// A texture view: its description (none: the resource's own) and the
    /// texture's (dimension, width, height, depth or array size, mips,
    /// format, sample count, flags). Contents are not captured.
    TextureSrv { desc: Option<[u32; 10]>, texture: [u64; 8] },
    TextureUav { desc: Option<[u32; 10]>, texture: [u64; 8] },
    /// What the capture could not record (replayed as a null view).
    Skipped(String),
}

fn root_name(r: RootType) -> String {
    match r {
        RootType::Constants => "const".into(),
        RootType::Descriptor => "descriptor".into(),
        RootType::Table => "table".into(),
        RootType::ImmediateSampler => "immsampler".into(),
        RootType::ResourceHeap => "resheap".into(),
        RootType::SamplerHeap => "samplerheap".into(),
        RootType::Other(n) => format!("other{n}"),
    }
}

fn root_of(s: &str) -> Option<RootType> {
    Some(match s {
        "const" => RootType::Constants,
        "descriptor" => RootType::Descriptor,
        "table" => RootType::Table,
        "immsampler" => RootType::ImmediateSampler,
        "resheap" => RootType::ResourceHeap,
        "samplerheap" => RootType::SamplerHeap,
        _ => RootType::Other(s.strip_prefix("other")?.parse().ok()?),
    })
}

fn words(w: &[u32]) -> String {
    let parts: Vec<String> = w.iter().map(|v| format!("{v:#x}")).collect();
    parts.join(",")
}

fn words64(w: &[u64]) -> String {
    let parts: Vec<String> = w.iter().map(|v| format!("{v:#x}")).collect();
    parts.join(",")
}

fn num(s: &str) -> Result<u64, String> {
    let r = match s.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16),
        None => s.parse(),
    };
    r.map_err(|_| format!("not a number: {s:?}"))
}

fn num32(s: &str) -> Result<u32, String> {
    u32::try_from(num(s)?).map_err(|_| format!("too big: {s}"))
}

fn list(s: &str) -> Result<Vec<u64>, String> {
    if s == "-" || s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',').map(num).collect()
}

fn list32(s: &str) -> Result<Vec<u32>, String> {
    list(s)?
        .into_iter()
        .map(|v| u32::try_from(v).map_err(|_| format!("too big: {v}")))
        .collect()
}

fn fixed<const N: usize>(s: &str) -> Result<[u32; N], String> {
    list32(s)?
        .try_into()
        .map_err(|_| format!("want {N} words: {s}"))
}

fn desc_text(d: &Option<[u32; 10]>) -> String {
    d.as_ref().map_or("-".into(), |d| words(d))
}

fn desc_of(s: &str) -> Result<Option<[u32; 10]>, String> {
    if s == "-" { Ok(None) } else { fixed(s).map(Some) }
}

impl Capture {
    /// The manifest text.
    pub fn to_text(&self) -> String {
        let mut t = String::new();
        let _ = writeln!(t, "# VIctory dispatch capture; `victory replay <dir>` runs it");
        let _ = writeln!(t, "version {VERSION}");
        let _ = writeln!(t, "shader {}", self.shader);
        let _ = writeln!(t, "dxil {}", self.dxil);
        if let Some(c) = &self.container {
            let _ = writeln!(t, "container {c}");
        }
        match self.guard {
            Some(g) => {
                let _ = writeln!(t, "guard {} {} {}", g.limit, g.kinds_slot, g.null_base);
            }
            None => {
                let _ = writeln!(t, "guard none");
            }
        }
        let _ = writeln!(t, "strict_kinds {}", u8::from(self.strict_kinds));
        let _ = writeln!(t, "root_srv_tables {}", u8::from(self.root_srv_tables));
        match &self.groups {
            Groups::Direct([x, y, z]) => {
                let _ = writeln!(t, "groups {x} {y} {z}");
            }
            Groups::Indirect { args, max } => {
                let a = args.map_or("-".into(), |a| words(&a));
                let _ = writeln!(t, "indirect {a} {max}");
            }
        }
        for u in &self.layout.user_data {
            let _ = writeln!(
                t,
                "user {} {} {} {}",
                u.index,
                u.slot,
                root_name(u.root),
                u.dword
            );
        }
        for s in &self.layout.samplers {
            let _ = writeln!(t, "static_sampler {}", words(s));
        }
        for (key, slots) in [("root_cbvs", &self.layout.root_cbvs), ("root_srvs", &self.layout.root_srvs)] {
            let v: Vec<u32> = slots.iter().map(|&s| u32::from(s)).collect();
            let v = if v.is_empty() { "-".to_owned() } else { words(&v) };
            let _ = writeln!(t, "{key} {v}");
        }
        if !self.layout.root_uavs.is_empty() {
            let v: Vec<u32> = self.layout.root_uavs.iter().map(|&s| u32::from(s)).collect();
            let _ = writeln!(t, "root_uavs {}", words(&v));
        }
        if self.layout.address_map {
            let _ = writeln!(t, "address_map 1");
        }
        if self.layout.descriptor_words {
            let _ = writeln!(t, "descriptor_words 1");
        }
        for b in &self.buffers {
            let _ = writeln!(
                t,
                "buffer {} {:#x} {:#x} {:#x} {} {:#x}",
                b.id, b.size, b.lo, b.hi, b.file, b.title
            );
        }
        for (slot, r) in &self.roots {
            let _ = match r {
                Root::None => writeln!(t, "root {slot} none"),
                Root::Constants(c) => writeln!(t, "root {slot} const {}", if c.is_empty() { "-".into() } else { words(c) }),
                Root::Table { sampler, offset } => writeln!(
                    t,
                    "root {slot} table {} {offset:#x}",
                    if *sampler { "samplers" } else { "views" }
                ),
                Root::View { buffer, offset } => writeln!(t, "root {slot} view {buffer} {offset:#x}"),
                Root::Unbacked(a) => writeln!(t, "root {slot} unbacked {a:#x}"),
            };
        }
        for (i, s) in &self.views {
            let _ = match s {
                Slot::Null => writeln!(t, "slot {i} null"),
                Slot::Cbv { buffer, offset, size } => {
                    writeln!(t, "slot {i} cbv {buffer} {offset:#x} {size:#x}")
                }
                Slot::Srv { buffer, desc } => writeln!(t, "slot {i} srv {buffer} {}", words(desc)),
                Slot::Uav { buffer, desc } => writeln!(t, "slot {i} uav {buffer} {}", words(desc)),
                Slot::TextureSrv { desc, texture } => writeln!(
                    t,
                    "slot {i} texsrv {} {}",
                    desc_text(desc),
                    words64(texture)
                ),
                Slot::TextureUav { desc, texture } => writeln!(
                    t,
                    "slot {i} texuav {} {}",
                    desc_text(desc),
                    words64(texture)
                ),
                Slot::Skipped(why) => writeln!(t, "slot {i} skipped {}", why.replace('\n', " ")),
            };
        }
        for (i, w) in &self.samplers {
            let _ = writeln!(t, "sampler {i} {}", words(w));
        }
        for n in &self.notes {
            let _ = writeln!(t, "note {}", n.replace('\n', " "));
        }
        t
    }

    /// Reads a manifest.
    pub fn parse(text: &str) -> Result<Capture, String> {
        let mut c = Capture::default();
        let mut version = None;
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let at = |e: String| format!("line {}: {e}: {line}", n + 1);
            let f: Vec<&str> = line.split_whitespace().collect();
            let field = |k: usize| f.get(k).copied().ok_or_else(|| at("missing field".into()));
            let rest = |k: usize| f.get(k..).map(|r| r.join(" ")).unwrap_or_default();
            match f[0] {
                "version" => version = Some(num32(field(1)?).map_err(at)?),
                "shader" => c.shader = field(1)?.to_owned(),
                "dxil" => c.dxil = field(1)?.to_owned(),
                "container" => c.container = Some(field(1)?.to_owned()),
                "guard" => {
                    c.guard = if field(1)? == "none" {
                        None
                    } else {
                        Some(HeapGuard {
                            limit: num32(field(1)?).map_err(at)?,
                            kinds_slot: num32(field(2)?).map_err(at)?,
                            null_base: num32(field(3)?).map_err(at)?,
                        })
                    }
                }
                "strict_kinds" => c.strict_kinds = field(1)? == "1",
                "root_srv_tables" => c.root_srv_tables = field(1)? == "1",
                "groups" => {
                    c.groups = Groups::Direct([
                        num32(field(1)?).map_err(at)?,
                        num32(field(2)?).map_err(at)?,
                        num32(field(3)?).map_err(at)?,
                    ])
                }
                "indirect" => {
                    let a = field(1)?;
                    c.groups = Groups::Indirect {
                        args: if a == "-" { None } else { Some(fixed(a).map_err(at)?) },
                        max: num32(field(2)?).map_err(at)?,
                    }
                }
                "user" => c.layout.user_data.push(UserData {
                    index: num32(field(1)?).map_err(at)?,
                    slot: u8::try_from(num(field(2)?).map_err(at)?).map_err(|_| at("slot".into()))?,
                    root: root_of(field(3)?).ok_or_else(|| at("root type".into()))?,
                    dword: num32(field(4)?).map_err(at)?,
                }),
                "static_sampler" => c.layout.samplers.push(fixed(field(1)?).map_err(at)?),
                "root_cbvs" | "root_srvs" | "root_uavs" => {
                    let v: Vec<u8> = list32(field(1)?)
                        .map_err(at)?
                        .into_iter()
                        .map(|v| v as u8)
                        .collect();
                    if f[0] == "root_cbvs" {
                        c.layout.root_cbvs = v;
                    } else if f[0] == "root_srvs" {
                        c.layout.root_srvs = v;
                    } else {
                        c.layout.root_uavs = v;
                    }
                }
                "address_map" => c.layout.address_map = num32(field(1)?).map_err(at)? != 0,
                "descriptor_words" => {
                    c.layout.descriptor_words = num32(field(1)?).map_err(at)? != 0
                }
                "buffer" => c.buffers.push(Buffer {
                    id: num32(field(1)?).map_err(at)?,
                    size: num(field(2)?).map_err(at)?,
                    lo: num(field(3)?).map_err(at)?,
                    hi: num(field(4)?).map_err(at)?,
                    file: field(5)?.to_owned(),
                    title: num(field(6)?).map_err(at)?,
                }),
                "root" => {
                    let slot = u8::try_from(num(field(1)?).map_err(at)?).map_err(|_| at("slot".into()))?;
                    let r = match field(2)? {
                        "none" => Root::None,
                        "const" => Root::Constants(list32(field(3)?).map_err(at)?),
                        "table" => Root::Table {
                            sampler: field(3)? == "samplers",
                            offset: num32(field(4)?).map_err(at)?,
                        },
                        "view" => Root::View {
                            buffer: num32(field(3)?).map_err(at)?,
                            offset: num(field(4)?).map_err(at)?,
                        },
                        "unbacked" => Root::Unbacked(num(field(3)?).map_err(at)?),
                        _ => return Err(at("root kind".into())),
                    };
                    c.roots.push((slot, r));
                }
                "slot" => {
                    let i = num32(field(1)?).map_err(at)?;
                    let s = match field(2)? {
                        "null" => Slot::Null,
                        "cbv" => Slot::Cbv {
                            buffer: num32(field(3)?).map_err(at)?,
                            offset: num(field(4)?).map_err(at)?,
                            size: num32(field(5)?).map_err(at)?,
                        },
                        "srv" => Slot::Srv {
                            buffer: num32(field(3)?).map_err(at)?,
                            desc: fixed(field(4)?).map_err(at)?,
                        },
                        "uav" => Slot::Uav {
                            buffer: num32(field(3)?).map_err(at)?,
                            desc: fixed(field(4)?).map_err(at)?,
                        },
                        k @ ("texsrv" | "texuav") => {
                            let desc = desc_of(field(3)?).map_err(at)?;
                            let texture: [u64; 8] = list(field(4)?)
                                .map_err(at)?
                                .try_into()
                                .map_err(|_| at("texture: 8 words".into()))?;
                            if k == "texsrv" {
                                Slot::TextureSrv { desc, texture }
                            } else {
                                Slot::TextureUav { desc, texture }
                            }
                        }
                        "skipped" => Slot::Skipped(rest(3)),
                        _ => return Err(at("slot kind".into())),
                    };
                    c.views.push((i, s));
                }
                "sampler" => c
                    .samplers
                    .push((num32(field(1)?).map_err(at)?, fixed(field(2)?).map_err(at)?)),
                "note" => c.notes.push(rest(1)),
                _ => return Err(at("unknown keyword".into())),
            }
        }
        match version {
            Some(VERSION) => Ok(c),
            Some(v) => Err(format!("capture version {v}, this build reads {VERSION}")),
            None => Err("no version line: not a capture manifest".into()),
        }
    }

    /// The user-data dwords translated shaders get for `layout`, as the game
    /// computes them (`draw.rs`, `State::user_value`): a table's offset, a
    /// constant's value, else 0.
    pub fn user_values(&self, layout: &Layout) -> Vec<u32> {
        let dwords = layout
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0)
            .next_multiple_of(4);
        let mut v = vec![0u32; dwords as usize];
        for u in &layout.user_data {
            let root = self.roots.iter().find(|(s, _)| *s == u.slot).map(|(_, r)| r);
            v[u.index as usize] = match (root, u.root) {
                (Some(Root::Table { offset, .. }), RootType::Table) => *offset,
                (Some(Root::Constants(c)), RootType::Constants) => {
                    c.get(u.dword as usize).copied().unwrap_or(0)
                }
                _ => 0,
            };
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifests_round_trip() {
        let c = Capture {
            shader: "0123456789abcdef".into(),
            dxil: "shader.dxil".into(),
            container: Some("shader.bin".into()),
            layout: Layout {
                user_data: vec![
                    UserData { index: 0, slot: 1, root: RootType::Table, dword: 0 },
                    UserData { index: 1, slot: 0, root: RootType::Constants, dword: 3 },
                ],
                samplers: vec![[1, 2, 3, 4]],
                root_cbvs: vec![2],
                root_srvs: vec![],
                root_uavs: vec![5],
                address_map: true,
                descriptor_words: true,
            },
            guard: Some(HeapGuard { limit: 100, kinds_slot: 100, null_base: 100 }),
            strict_kinds: true,
            root_srv_tables: true,
            groups: Groups::Indirect { args: Some([4, 5, 6]), max: 1 },
            roots: vec![
                (0, Root::Constants(vec![9, 8, 7, 6])),
                (1, Root::Table { sampler: false, offset: 0x40 }),
                (2, Root::View { buffer: 0, offset: 0x10 }),
                (3, Root::Unbacked(0xdead)),
                (4, Root::None),
            ],
            buffers: vec![Buffer { id: 0, size: 0x1000, lo: 0, hi: 0x100, file: "b0.bin".into(), title: 0x1234 }],
            views: vec![
                (2, Slot::Srv { buffer: 0, desc: [1; 10] }),
                (3, Slot::Cbv { buffer: 0, offset: 4, size: 16 }),
                (4, Slot::TextureSrv { desc: None, texture: [4, 16, 16, 1, 1, 28, 1, 0] }),
                (5, Slot::Skipped("a placed texture UAV".into())),
                (6, Slot::Null),
            ],
            samplers: vec![(0, [0; 13])],
            notes: vec!["contents read at submission".into()],
        };
        let back = Capture::parse(&c.to_text()).unwrap();
        assert_eq!(back, c);
        assert_eq!(c.user_values(&c.layout), vec![0x40, 6, 0, 0]);
    }
}
