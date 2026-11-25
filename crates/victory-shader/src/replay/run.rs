//! Replays a [`super::Capture`] on a fresh host D3D12 device: the buffers
//! and descriptors at the heap slots the game used, the root signature the
//! game makes for the translation, one dispatch, then the device's verdict.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::{Capture, Groups, MANIFEST, Root, Slot};
use crate::emit::host::Host;
use crate::emit::{GUARD_LOG_MARK, GUARD_LOG_RECORDS, GUARD_LOG_SLOT, GUARD_SLOTS, HostShader, kind, null_offset};

/// How to replay.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Translate the console container again with this build's translator
    /// instead of running the captured DXIL.
    pub retranslate: bool,
    /// The console container, if the capture has none (a
    /// `VICTORY_SHADER_DUMP` `compute_<hash>.bin`).
    pub container: Option<PathBuf>,
    pub warp: bool,
    /// The first hardware adapter whose name holds this.
    pub adapter: Option<String>,
    /// Write every buffer's bytes after the dispatch here.
    pub dump: Option<PathBuf>,
    pub debug_layer: bool,
    /// Device-removed data: the faulting address, placed in our buffers.
    pub dred: bool,
    /// How long to wait for the GPU (default 10 s).
    pub timeout: Option<Duration>,
    /// Submit the dispatch this many times (default 1).
    pub repeat: u32,
    /// Buffer views serve reads of every buffer kind (AMD) whatever the
    /// capture says.
    pub lax_kinds: bool,
}

/// What a replay found.
#[derive(Debug, Default)]
pub struct Report {
    pub lines: Vec<String>,
    /// The device-removed reason, with DRED data if it was on.
    pub removed: Option<String>,
    pub timed_out: bool,
}

fn err(e: windows::core::Error) -> String {
    format!("{:#010x} {}", e.code().0, e.message())
}

/// Classes (bit per `emit::kind_class`) the kinds table holds.
const CBV: u32 = 1 << 1;
const TEXTURE_SRV: u32 = 1 << 4;
const TEXTURE_UAV: u32 = 1 << 5;
const BUFFER_SRVS: [u32; 3] = [1 << 2, 1 << 6, 1 << 7];
const BUFFER_UAVS: [u32; 3] = [1 << 3, 1 << 8, 1 << 9];

fn buffer_mask(uav: bool, raw: bool, stride: u32, strict: bool) -> u32 {
    let masks = if uav { BUFFER_UAVS } else { BUFFER_SRVS };
    if !strict {
        return masks.iter().fold(0, |a, m| a | m);
    }
    masks[if raw { 1 } else if stride > 0 { 2 } else { 0 }]
}

fn srv_desc(w: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    // SAFETY: 40 bytes in the host's layout.
    unsafe { std::ptr::from_ref(w).cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>().read_unaligned() }
}

fn uav_desc(w: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    // SAFETY: 40 bytes in the host's layout.
    unsafe { std::ptr::from_ref(w).cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>().read_unaligned() }
}

fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

/// A buffer the CPU maps and shaders may write (as the game's host copies
/// of title buffers: write-back system memory).
struct Mapped {
    resource: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    size: u64,
}

struct Device<'a> {
    dev: &'a ID3D12Device,
    keep: Vec<ID3D12Resource>,
}

impl Device<'_> {
    fn committed(
        &mut self,
        props: D3D12_HEAP_PROPERTIES,
        desc: &D3D12_RESOURCE_DESC,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<ID3D12Resource, String> {
        let mut r: Option<ID3D12Resource> = None;
        // SAFETY: valid descriptions and out pointer.
        unsafe {
            self.dev
                .CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, desc, state, None, &mut r)
                .map_err(err)?;
        }
        let r = r.ok_or("no resource")?;
        self.keep.push(r.clone());
        Ok(r)
    }

    fn mapped(&mut self, size: u64, upload: bool) -> Result<Mapped, String> {
        let size = size.max(1).next_multiple_of(256);
        let (props, flags, state) = if upload {
            (
                D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )
        } else {
            (
                D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_CUSTOM,
                    CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                    MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                    ..Default::default()
                },
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )
        };
        let resource = self.committed(props, &buffer_desc(size, flags), state)?;
        let mut data = std::ptr::null_mut();
        // SAFETY: a CPU-visible buffer, mapped for its life.
        unsafe {
            resource.Map(0, None, Some(&mut data)).map_err(err)?;
            std::ptr::write_bytes(data.cast::<u8>(), 0, size as usize);
        }
        // SAFETY: a live buffer.
        let gpu = unsafe { resource.GetGPUVirtualAddress() };
        Ok(Mapped { resource, data: data.cast(), gpu, size })
    }
}

/// The root signature the game makes for a compute translation
/// (`victory-gpu` `pipeline/host.rs`): user data, root CBVs, root SRVs (as
/// one-view tables if `tables`), static samplers. Returns it with the
/// parameters: user data, (slot, parameter) of the CBVs and of the SRVs.
type Params = (Option<u32>, Vec<(u8, u32)>, Vec<(u8, u32)>);

fn root_signature(
    dev: &ID3D12Device,
    s: &HostShader,
    tables: bool,
) -> Result<(ID3D12RootSignature, Params), String> {
    use crate::emit::{ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE};
    if !s.root_uavs.is_empty() {
        // shortcut: a capture holds no root UAV contents and nothing reads
        // them back here.
        return Err(format!("root UAVs {:?}: not replayed yet", s.root_uavs));
    }
    if s.address_map || s.descriptor_words {
        // shortcut: a capture holds neither the address map (every host
        // copy of title memory) nor the view heap's descriptor words.
        return Err("the address map or the descriptor words: not replayed yet".into());
    }
    let mut params = Vec::new();
    let mut ranges: Vec<Box<D3D12_DESCRIPTOR_RANGE1>> = Vec::new();
    let dwords = s.user_data_dwords();
    let user = (dwords > 0).then(|| {
        params.push(D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: USER_DATA_SPACE,
                    Num32BitValues: dwords,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
        0
    });
    let (mut cbvs, mut srvs) = (Vec::new(), Vec::new());
    for &slot in &s.root_cbvs {
        cbvs.push((slot, params.len() as u32));
        params.push(D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: u32::from(slot),
                    RegisterSpace: ROOT_CBV_SPACE,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
    }
    for &slot in &s.root_srvs {
        srvs.push((slot, params.len() as u32));
        if tables {
            ranges.push(Box::new(D3D12_DESCRIPTOR_RANGE1 {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 1,
                BaseShaderRegister: u32::from(slot),
                RegisterSpace: ROOT_CBV_SPACE,
                Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE
                    | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                OffsetInDescriptorsFromTableStart: 0,
            }));
            let range: *const D3D12_DESCRIPTOR_RANGE1 = &**ranges.last().ok_or("range")?;
            params.push(D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: range,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
        } else {
            params.push(D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: u32::from(slot),
                        RegisterSpace: ROOT_CBV_SPACE,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
        }
    }
    let samplers: Vec<D3D12_STATIC_SAMPLER_DESC> = s
        .static_samplers
        .iter()
        .enumerate()
        .map(|(n, d)| D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER(d.filter as i32),
            AddressU: D3D12_TEXTURE_ADDRESS_MODE(d.address[0] as i32),
            AddressV: D3D12_TEXTURE_ADDRESS_MODE(d.address[1] as i32),
            AddressW: D3D12_TEXTURE_ADDRESS_MODE(d.address[2] as i32),
            MipLODBias: d.mip_lod_bias,
            MaxAnisotropy: d.max_anisotropy,
            ComparisonFunc: D3D12_COMPARISON_FUNC(d.comparison as i32),
            BorderColor: D3D12_STATIC_BORDER_COLOR(d.border as i32),
            MinLOD: d.min_lod,
            MaxLOD: d.max_lod,
            ShaderRegister: n as u32,
            RegisterSpace: SAMPLER_SPACE,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        })
        .collect();
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    // SAFETY: the description points at live arrays.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            let why = error.map(|b| {
                String::from_utf8_lossy(std::slice::from_raw_parts(
                    b.GetBufferPointer().cast::<u8>(),
                    b.GetBufferSize(),
                ))
                .into_owned()
            });
            return Err(format!("root signature: {} {}", err(e), why.unwrap_or_default()));
        }
        let blob = blob.ok_or("no root signature blob")?;
        let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        let root = dev.CreateRootSignature(0, bytes).map_err(err)?;
        Ok((root, (user, cbvs, srvs)))
    }
}

/// A null view in each heap kind's style (`null_view` in the game: a 2D
/// texture SRV).
fn null_texture_srv() -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut d = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ..Default::default()
    };
    d.Anonymous.Texture2D.MipLevels = 1;
    d
}

fn null_raw_srv() -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    }
}

/// The guard's null views (as `HostHeap::guard` makes them), one per kind.
///
/// # Safety
/// `at(i)` must be a slot of a live view heap.
unsafe fn guard_nulls(dev: &ID3D12Device, at: impl Fn(u32) -> D3D12_CPU_DESCRIPTOR_HANDLE) {
    let srv = |dim: D3D12_SRV_DIMENSION| D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: dim,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ..Default::default()
    };
    // SAFETY: the caller's slots; null resources.
    unsafe {
        for k in 1..=13u32 {
            let slot = at(null_offset(k, false));
            let mut d = match k {
                kind::TEXTURE1D => srv(D3D12_SRV_DIMENSION_TEXTURE1D),
                kind::TEXTURE2DMS => srv(D3D12_SRV_DIMENSION_TEXTURE2DMS),
                kind::TEXTURE3D => srv(D3D12_SRV_DIMENSION_TEXTURE3D),
                kind::TEXTURECUBE => srv(D3D12_SRV_DIMENSION_TEXTURECUBE),
                kind::TEXTURE1D_ARRAY => srv(D3D12_SRV_DIMENSION_TEXTURE1DARRAY),
                kind::TEXTURE2D_ARRAY => srv(D3D12_SRV_DIMENSION_TEXTURE2DARRAY),
                kind::TEXTURE2DMS_ARRAY => srv(D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY),
                kind::TEXTURECUBE_ARRAY => srv(D3D12_SRV_DIMENSION_TEXTURECUBEARRAY),
                kind::TYPED_BUFFER | kind::RAW_BUFFER | kind::STRUCTURED_BUFFER => {
                    srv(D3D12_SRV_DIMENSION_BUFFER)
                }
                kind::CBUFFER => {
                    dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC::default()), slot);
                    continue;
                }
                _ => srv(D3D12_SRV_DIMENSION_TEXTURE2D),
            };
            match k {
                kind::TEXTURE1D => d.Anonymous.Texture1D.MipLevels = 1,
                kind::TEXTURE3D => d.Anonymous.Texture3D.MipLevels = 1,
                kind::TEXTURECUBE => d.Anonymous.TextureCube.MipLevels = 1,
                kind::TEXTURE1D_ARRAY => {
                    d.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_SRV { MipLevels: 1, ArraySize: 1, ..Default::default() }
                }
                kind::TEXTURE2D_ARRAY => {
                    d.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, ArraySize: 1, ..Default::default() }
                }
                kind::TEXTURE2DMS_ARRAY => {
                    d.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_SRV { ArraySize: 1, ..Default::default() }
                }
                kind::TEXTURECUBE_ARRAY => {
                    d.Anonymous.TextureCubeArray = D3D12_TEXCUBE_ARRAY_SRV { MipLevels: 1, NumCubes: 1, ..Default::default() }
                }
                kind::TYPED_BUFFER => {
                    d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
                    d.Anonymous.Buffer = D3D12_BUFFER_SRV { NumElements: 1, ..Default::default() };
                }
                kind::RAW_BUFFER => d = null_raw_srv(),
                kind::STRUCTURED_BUFFER => {
                    d.Format = DXGI_FORMAT_UNKNOWN;
                    d.Anonymous.Buffer = D3D12_BUFFER_SRV { NumElements: 1, StructureByteStride: 4, ..Default::default() };
                }
                kind::TEXTURE2DMS => {}
                _ => d.Anonymous.Texture2D.MipLevels = 1,
            }
            dev.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&d), slot);
        }
        let uav_tex = |dim: D3D12_UAV_DIMENSION| {
            let mut d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: dim,
                ..Default::default()
            };
            if dim == D3D12_UAV_DIMENSION_TEXTURE1DARRAY {
                d.Anonymous.Texture1DArray.ArraySize = 1;
            } else if dim == D3D12_UAV_DIMENSION_TEXTURE2DARRAY {
                d.Anonymous.Texture2DArray.ArraySize = 1;
            } else if dim == D3D12_UAV_DIMENSION_TEXTURE3D {
                d.Anonymous.Texture3D.WSize = 1;
            }
            d
        };
        let uav_buf = |format, raw: bool, stride| D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: 1,
                    StructureByteStride: stride,
                    Flags: if raw { D3D12_BUFFER_UAV_FLAG_RAW } else { D3D12_BUFFER_UAV_FLAG_NONE },
                    ..Default::default()
                },
            },
        };
        for k in [
            kind::TEXTURE1D,
            kind::TEXTURE2D,
            kind::TEXTURE3D,
            kind::TEXTURE1D_ARRAY,
            kind::TEXTURE2D_ARRAY,
            kind::TYPED_BUFFER,
            kind::RAW_BUFFER,
            kind::STRUCTURED_BUFFER,
        ] {
            let d = match k {
                kind::TYPED_BUFFER => uav_buf(DXGI_FORMAT_R32_UINT, false, 0),
                kind::RAW_BUFFER => uav_buf(DXGI_FORMAT_R32_TYPELESS, true, 0),
                kind::STRUCTURED_BUFFER => uav_buf(DXGI_FORMAT_UNKNOWN, false, 4),
                kind::TEXTURE1D => uav_tex(D3D12_UAV_DIMENSION_TEXTURE1D),
                kind::TEXTURE3D => uav_tex(D3D12_UAV_DIMENSION_TEXTURE3D),
                kind::TEXTURE1D_ARRAY => uav_tex(D3D12_UAV_DIMENSION_TEXTURE1DARRAY),
                kind::TEXTURE2D_ARRAY => uav_tex(D3D12_UAV_DIMENSION_TEXTURE2DARRAY),
                _ => uav_tex(D3D12_UAV_DIMENSION_TEXTURE2D),
            };
            dev.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&d), at(null_offset(k, true)));
        }
    }
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

/// Replays the capture in `dir`.
pub fn replay(dir: &Path, o: &Options) -> Result<Report, String> {
    let text = std::fs::read_to_string(dir.join(MANIFEST))
        .map_err(|e| format!("{}: {e}", dir.join(MANIFEST).display()))?;
    let begun = Instant::now();
    let mut phases: Vec<(&str, Instant)> = Vec::new();
    let cap = Capture::parse(&text)?;
    let mut report = Report::default();
    let say = |r: &mut Report, s: String| r.lines.push(s);
    say(&mut report, format!("capture of shader {} ({} buffers, {} view slots, {} samplers)",
        cap.shader, cap.buffers.len(), cap.views.len(), cap.samplers.len()));
    for n in &cap.notes {
        say(&mut report, format!("capture note: {n}"));
    }

    // SAFETY: plain debug-interface calls before the device exists.
    unsafe {
        if o.debug_layer {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => say(&mut report, format!("no debug layer: {}", err(e))),
            }
        }
        if o.dred {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
    }
    let host = match &o.adapter {
        Some(name) => Host::on(name)?,
        None => Host::new(o.warp)?,
    };
    phases.push(("device", Instant::now()));
    let dev = &host.device;

    // The shader: the captured DXIL, or this build's translation.
    let shader = if o.retranslate {
        let path = o
            .container
            .clone()
            .or_else(|| cap.container.as_ref().map(|c| dir.join(c)))
            .ok_or("--retranslate: the capture has no container; pass --container <compute_HASH.bin> from a VICTORY_SHADER_DUMP directory")?;
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if let Some(g) = cap.guard {
            crate::emit::set_view_heap_guard(g);
        }
        let s = crate::emit::translate_for(&bytes, host.target()).map_err(|e| format!("retranslation: {e}"))?;
        // `VICTORY_REPLAY_DXIL=<file>`: keep the retranslated DXIL (to diff
        // translator variants).
        if let Some(out) = std::env::var_os("VICTORY_REPLAY_DXIL") {
            let _ = std::fs::write(out, &s.dxil);
        }
        let same = super::Layout::of(&s) == cap.layout;
        say(&mut report, format!(
            "retranslated {} ({} bytes of DXIL; bindings {} the capture's)",
            path.display(),
            s.dxil.len(),
            if same { "same as" } else { "DIFFER from" }
        ));
        s
    } else {
        let path = dir.join(&cap.dxil);
        let dxil = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        cap.layout.shader(dxil)
    };
    let strict = cap.strict_kinds && !o.lax_kinds;
    let (root, (user_param, cbv_params, srv_params)) = root_signature(dev, &shader, cap.root_srv_tables)?;
    phases.push(("shader", Instant::now()));
    let pso = host.compute_pipeline_with(&root, &shader.dxil).map_err(|e| format!("pipeline refused: {e}"))?;
    phases.push(("pipeline", Instant::now()));

    let mut d = Device { dev, keep: Vec::new() };

    // Buffers: the host copies, captured bytes at their offsets.
    let mut buffers: Vec<(u32, Mapped, Vec<u8>)> = Vec::new();
    for b in &cap.buffers {
        let m = d.mapped(b.size, false).map_err(|e| format!("buffer {} ({:#x} bytes): {e}", b.id, b.size))?;
        let path = dir.join(&b.file);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let n = bytes.len().min(m.size.saturating_sub(b.lo) as usize);
        // SAFETY: inside the mapping.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), m.data.add(b.lo as usize), n) };
        buffers.push((b.id, m, bytes));
    }
    let buffer = |id: u32| buffers.iter().find(|(i, ..)| *i == id).map(|(_, m, _)| m);

    // The view heap: the guard's size if the translation has one.
    let max_view = cap.views.iter().map(|(i, _)| *i + 1).max().unwrap_or(0);
    let tabled = if cap.root_srv_tables { srv_params.len() as u32 } else { 0 };
    let total = match cap.guard {
        Some(g) => {
            if max_view > g.limit {
                say(&mut report, format!("slots past the guard limit {} are left out", g.limit));
            }
            g.null_base + GUARD_SLOTS
        }
        None => max_view + tabled + 1,
    };
    // SAFETY: plain D3D12 calls on live objects; every handle is a slot of
    // our heaps; buffers outlive the GPU work (`d.keep`, `buffers`).
    unsafe {
        let views: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: total,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .map_err(|e| format!("view heap of {total}: {}", err(e)))?;
        let vstep = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let vcpu = views.GetCPUDescriptorHandleForHeapStart().ptr;
        let vgpu = views.GetGPUDescriptorHandleForHeapStart().ptr;
        let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: vcpu + i as usize * vstep };
        // Every slot null first, by copies from a CPU-side block.
        {
            const BLOCK: u32 = 1 << 14;
            let block: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })
                .map_err(err)?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            dev.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&null_texture_srv()), start);
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                dev.CopyDescriptorsSimple(n, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start.ptr + filled as usize * vstep }, start, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                dev.CopyDescriptorsSimple(n, at(done), start, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                done += n;
            }
        }
        // The guard: kinds table, null views, the log.
        let mut kinds: Option<*mut u32> = None;
        let mut guard_log: Option<Mapped> = None;
        if let Some(g) = cap.guard {
            let slots = (g.null_base + GUARD_SLOTS) as u64;
            // (Host slot, classes) per slot; here every slot is its own.
            let table = d.mapped(slots * 8, true)?;
            let mut desc = null_raw_srv();
            desc.Anonymous.Buffer.NumElements = 2 * slots as u32;
            dev.CreateShaderResourceView(&table.resource, Some(&desc), at(g.kinds_slot));
            let words: *mut u32 = table.data.cast();
            for i in 0..slots as usize {
                words.add(2 * i).write(i as u32);
            }
            kinds = Some(words);
            guard_nulls(dev, |k| at(g.null_base + k));
            let log = d.mapped(16 * (u64::from(GUARD_LOG_RECORDS) + 2), false)?;
            let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        NumElements: (log.size / 4) as u32,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            dev.CreateUnorderedAccessView(&log.resource, None::<&ID3D12Resource>, Some(&desc), at(g.null_base + GUARD_LOG_SLOT));
            guard_log = Some(log);
        }
        let note_kind = |i: u32, class: u32| {
            if let (Some(k), Some(g)) = (kinds, cap.guard)
                && i < g.null_base
            {
                k.add(2 * i as usize + 1).write(class);
            }
        };

        // The captured slots.
        let (mut made, mut textures, mut skipped) = (0, 0, Vec::new());
        for (i, s) in &cap.views {
            let i = *i;
            if i >= total || cap.guard.is_some_and(|g| i >= g.limit) {
                skipped.push(format!("{i}: past the heap"));
                continue;
            }
            match s {
                Slot::Null => {}
                Slot::Cbv { buffer: b, offset, size } => {
                    let Some(src) = buffer(*b) else {
                        skipped.push(format!("{i}: cbv of unknown buffer {b}"));
                        continue;
                    };
                    let copy = d.mapped(u64::from(*size), true)?;
                    let n = (u64::from(*size)).min(src.size.saturating_sub(*offset)) as usize;
                    std::ptr::copy_nonoverlapping(src.data.add(*offset as usize), copy.data, n);
                    dev.CreateConstantBufferView(
                        Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC {
                            BufferLocation: copy.gpu,
                            SizeInBytes: size.next_multiple_of(256).max(256),
                        }),
                        at(i),
                    );
                    note_kind(i, CBV);
                    made += 1;
                }
                Slot::Srv { buffer: b, desc } => {
                    let Some(m) = buffer(*b) else {
                        skipped.push(format!("{i}: srv of unknown buffer {b}"));
                        continue;
                    };
                    let desc = srv_desc(desc);
                    dev.CreateShaderResourceView(&m.resource, Some(&desc), at(i));
                    let bd = desc.Anonymous.Buffer;
                    note_kind(i, buffer_mask(false, bd.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0, bd.StructureByteStride, strict));
                    made += 1;
                }
                Slot::Uav { buffer: b, desc } => {
                    let Some(m) = buffer(*b) else {
                        skipped.push(format!("{i}: uav of unknown buffer {b}"));
                        continue;
                    };
                    let desc = uav_desc(desc);
                    dev.CreateUnorderedAccessView(&m.resource, None::<&ID3D12Resource>, Some(&desc), at(i));
                    let bd = desc.Anonymous.Buffer;
                    note_kind(i, buffer_mask(true, bd.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0, bd.StructureByteStride, strict));
                    made += 1;
                }
                Slot::TextureSrv { desc, texture } | Slot::TextureUav { desc, texture } => {
                    let uav = matches!(s, Slot::TextureUav { .. });
                    let [dim, width, height, depth, mips, mut format, samples, _flags] = *texture;
                    // Contents are not captured: the texture takes the
                    // view's format (so any view fits) and no target flags.
                    if let Some(w) = desc
                        && w[0] != 0
                    {
                        format = u64::from(w[0]);
                    }
                    let flags = if uav {
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                    } else {
                        D3D12_RESOURCE_FLAG_NONE
                    };
                    let rd = D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION(dim as i32),
                        Width: width,
                        Height: height as u32,
                        DepthOrArraySize: depth as u16,
                        MipLevels: mips as u16,
                        Format: DXGI_FORMAT(format as i32),
                        SampleDesc: DXGI_SAMPLE_DESC { Count: (samples as u32).max(1), Quality: 0 },
                        Flags: flags,
                        ..Default::default()
                    };
                    let state = if uav {
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                    } else {
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                    };
                    let tex = match d.committed(D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, &rd, state) {
                        Ok(t) => t,
                        Err(e) => {
                            skipped.push(format!("{i}: texture {texture:x?} not made ({e}); null view"));
                            continue;
                        }
                    };
                    if uav {
                        let desc = desc.as_ref().map(uav_desc);
                        dev.CreateUnorderedAccessView(&tex, None::<&ID3D12Resource>, desc.as_ref().map(std::ptr::from_ref), at(i));
                        note_kind(i, TEXTURE_UAV);
                    } else {
                        let desc = desc.as_ref().map(srv_desc);
                        dev.CreateShaderResourceView(&tex, desc.as_ref().map(std::ptr::from_ref), at(i));
                        note_kind(i, TEXTURE_SRV);
                    }
                    textures += 1;
                }
                Slot::Skipped(why) => skipped.push(format!("{i}: {why}")),
            }
        }
        say(&mut report, format!(
            "heap: {total} view slots; {made} buffer/constant views, {textures} textures (zeroed: contents are not captured)"
        ));
        for s in skipped.iter().take(20) {
            say(&mut report, format!("  null instead: slot {s}"));
        }
        if skipped.len() > 20 {
            say(&mut report, format!("  ... {} more", skipped.len() - 20));
        }

        // Samplers.
        let scount = cap.samplers.iter().map(|(i, _)| i + 1).max().unwrap_or(1);
        let samplers: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                NumDescriptors: scount,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .map_err(|e| format!("sampler heap of {scount}: {}", err(e)))?;
        let sstep = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
        let scpu = samplers.GetCPUDescriptorHandleForHeapStart().ptr;
        let sat = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: scpu + i as usize * sstep };
        let point = D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            ..Default::default()
        };
        for i in 0..scount {
            dev.CreateSampler(&point, sat(i));
        }
        for (i, w) in &cap.samplers {
            let desc = std::ptr::from_ref(w).cast::<D3D12_SAMPLER_DESC>().read_unaligned();
            dev.CreateSampler(&desc, sat(*i));
        }

        // The list.
        let queue: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })
            .map_err(err)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(err)?;
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso)
            .map_err(err)?;
        list.SetDescriptorHeaps(&[Some(views.clone()), Some(samplers.clone())]);
        list.SetComputeRootSignature(&root);
        let root_of = |slot: u8| cap.roots.iter().find(|(s, _)| *s == slot).map(|(_, r)| r);
        if let Some(param) = user_param {
            let values = cap.user_values(&super::Layout::of(&shader));
            list.SetComputeRoot32BitConstants(param, values.len() as u32, values.as_ptr().cast(), 0);
            say(&mut report, format!("user data {values:x?}"));
        }
        for &(slot, param) in &cbv_params {
            match root_of(slot) {
                Some(Root::View { buffer: b, offset }) if buffer(*b).is_some() => {
                    let m = buffer(*b).ok_or("buffer")?;
                    let mut gpu = m.gpu + offset;
                    if gpu % 256 != 0 {
                        // The game copies an unaligned constant buffer too.
                        let len = m.size.saturating_sub(*offset).min(1 << 16);
                        let copy = d.mapped(len, true)?;
                        std::ptr::copy_nonoverlapping(m.data.add(*offset as usize), copy.data, len as usize);
                        gpu = copy.gpu;
                    }
                    list.SetComputeRootConstantBufferView(param, gpu);
                }
                other => say(&mut report, format!("root CBV {slot} left unset (the game leaves it so too): {other:?}")),
            }
        }
        let mut next_free = match cap.guard {
            Some(g) => g.limit,
            None => max_view + 1,
        };
        let used: std::collections::HashSet<u32> = cap.views.iter().map(|(i, _)| *i).collect();
        for &(slot, param) in &srv_params {
            let view = match root_of(slot) {
                Some(Root::View { buffer: b, offset }) => buffer(*b).map(|m| (m, *offset)),
                _ => None,
            };
            if cap.root_srv_tables {
                // A dynamic slot: below the guard (or past the captured ones).
                let index = if cap.guard.is_some() {
                    loop {
                        next_free -= 1;
                        if !used.contains(&next_free) {
                            break next_free;
                        }
                    }
                } else {
                    next_free += 1;
                    next_free - 1
                };
                let mut desc = null_raw_srv();
                match view {
                    Some((m, offset)) if offset % 16 == 0 => {
                        desc.Anonymous.Buffer.FirstElement = offset / 4;
                        desc.Anonymous.Buffer.NumElements = ((m.size - offset) / 4).clamp(1, u64::from(u32::MAX)) as u32;
                        dev.CreateShaderResourceView(&m.resource, Some(&desc), at(index));
                        note_kind(index, buffer_mask(false, true, 0, strict));
                    }
                    _ => {
                        dev.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at(index));
                        say(&mut report, format!("root SRV {slot}: a null raw view (unset, unbacked or not on 16 bytes)"));
                    }
                }
                list.SetComputeRootDescriptorTable(param, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: vgpu + u64::from(index) * vstep as u64 });
            } else {
                match view {
                    Some((m, offset)) => list.SetComputeRootShaderResourceView(param, m.gpu + offset),
                    None => say(&mut report, format!("root SRV {slot} left unset")),
                }
            }
        }
        let groups = match cap.groups {
            Groups::Direct(g) => g,
            Groups::Indirect { args: Some(a), max } => {
                let g = a.map(|v| v.min(65535));
                say(&mut report, format!("indirect dispatch (max {max}) replayed as Dispatch{g:?} from the first record (captured {a:?})"));
                g
            }
            Groups::Indirect { args: None, .. } => return Err("indirect dispatch whose arguments were not captured".into()),
        };
        list.Dispatch(groups[0], groups[1], groups[2]);
        list.Close().map_err(err)?;
        phases.push(("setup", Instant::now()));
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
        let started = Instant::now();
        let timeout = o.timeout.unwrap_or(Duration::from_secs(10));
        let repeat = o.repeat.max(1);
        let lists = [Some(list.cast::<ID3D12CommandList>().map_err(err)?)];
        for _ in 0..repeat {
            queue.ExecuteCommandLists(&lists);
        }
        queue.Signal(&fence, 1).map_err(err)?;
        say(&mut report, format!("dispatch {groups:?} submitted {repeat} time(s)"));
        while fence.GetCompletedValue() < 1 {
            if let Err(e) = dev.GetDeviceRemovedReason() {
                report.removed = Some(removal(dev, &e, &buffers, (vgpu, total, vstep as u64)));
                break;
            }
            if started.elapsed() > timeout {
                report.timed_out = true;
                break;
            }
            std::thread::sleep(Duration::from_micros(200));
        }
        if report.removed.is_none()
            && let Err(e) = dev.GetDeviceRemovedReason()
        {
            report.removed = Some(removal(dev, &e, &buffers, (vgpu, total, vstep as u64)));
        }
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        if let Some(why) = report.removed.clone() {
            say(&mut report, format!("DEVICE REMOVED after {ms:.1} ms: {why}"));
            return Ok(report);
        }
        if report.timed_out {
            say(&mut report, format!("TIMEOUT: the GPU did not finish in {timeout:?}"));
            return Ok(report);
        }
        say(&mut report, format!("OK: finished in {ms:.1} ms"));
        phases.push(("gpu", Instant::now()));

        // What the dispatch wrote.
        let written: std::collections::HashSet<u32> = cap
            .views
            .iter()
            .filter_map(|(_, s)| match s {
                Slot::Uav { buffer, .. } => Some(*buffer),
                _ => None,
            })
            .collect();
        if let Some(out) = &o.dump {
            std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
        }
        for (b, (id, m, before)) in cap.buffers.iter().zip(&buffers) {
            let after = std::slice::from_raw_parts(m.data.add(b.lo as usize), before.len().min(m.size.saturating_sub(b.lo) as usize));
            // Whole chunks compared first (memcmp): cheap in unoptimized builds.
            let changed: usize = after
                .chunks(4096)
                .zip(before.chunks(4096))
                .filter(|(a, b)| a != b)
                .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
                .sum();
            if written.contains(id) || changed > 0 {
                say(&mut report, format!(
                    "buffer {id}{}: {changed} of {} captured bytes changed, fnv {:#018x}",
                    if written.contains(id) { " (UAV)" } else { "" },
                    after.len(),
                    fnv(after)
                ));
            }
            if let Some(out) = &o.dump {
                let path = out.join(format!("buffer{id}.bin"));
                std::fs::write(&path, after).map_err(|e| format!("{}: {e}", path.display()))?;
            }
        }
        if let Some(log) = &guard_log {
            let count = log.data.cast::<u32>().read_volatile().min(GUARD_LOG_RECORDS);
            let mut seen = std::collections::BTreeSet::new();
            for r in 0..count as usize {
                let rec: [u32; 4] = std::array::from_fn(|k| log.data.cast::<u32>().add(4 * (1 + r) + k).read_volatile());
                if rec[3] == GUARD_LOG_MARK {
                    seen.insert((rec[0], rec[1] & 0xff, rec[1] & 0x1000 != 0, rec[2]));
                }
            }
            for (slot, kind, uav, classes) in seen.iter().take(20) {
                say(&mut report, format!("guard: slot {slot} read as kind {kind} (uav {uav}) holds classes {classes:#x}"));
            }
        }
        phases.push(("check", Instant::now()));
        let mut last = begun;
        let times: Vec<String> = phases
            .iter()
            .map(|(what, at)| {
                let ms = (*at - last).as_secs_f64() * 1000.0;
                last = *at;
                format!("{what} {ms:.0}")
            })
            .collect();
        say(&mut report, format!("times (ms): {}", times.join(", ")));
        drop(d);
        Ok(report)
    }
}

/// The removal reason, and with DRED on, the faulting address placed in
/// our buffers and heap.
fn removal(
    dev: &ID3D12Device,
    e: &windows::core::Error,
    buffers: &[(u32, Mapped, Vec<u8>)],
    (heap, slots, step): (u64, u32, u64),
) -> String {
    let mut text = err(e.clone());
    let Ok(dred) = dev.cast::<ID3D12DeviceRemovedExtendedData1>() else {
        return text;
    };
    // SAFETY: the runtime's output, valid while the device lives.
    unsafe {
        if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
            && fault.PageFaultVA != 0
        {
            let va = fault.PageFaultVA;
            text.push_str(&format!("; page fault at {va:#x}"));
            for (id, m, _) in buffers {
                if va >= m.gpu && va < m.gpu + m.size + (1 << 20) {
                    text.push_str(&format!("; buffer {id} +{:#x} (size {:#x})", va - m.gpu, m.size));
                }
            }
            let end = heap + u64::from(slots) * step;
            if va >= heap && va < end + (1 << 20) {
                text.push_str(&format!("; view heap slot {}", (va - heap) / step));
            }
        }
        if let Ok(crumbs) = dred.GetAutoBreadcrumbsOutput1() {
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(n) = node.as_ref() {
                let done = n.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                text.push_str(&format!("; breadcrumbs {done}/{}", n.BreadcrumbCount));
                node = n.pNext;
            }
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};
    use crate::replay::{Buffer, Layout};

    /// The waterfall tests' buffer program (each lane loads from the
    /// structured buffer its own descriptor names, stores to a raw UAV),
    /// translated as the game would: views table at root 0, samplers at 1.
    fn shader() -> HostShader {
        let table = |user_data: u8| RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data,
            slot: user_data,
        };
        let range = |kind, table_offset_dwords, dwords, stride, user_data| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords,
            lower_bound: 0,
            stride,
            dwords,
            user_data,
            slot: user_data,
        };
        let words = crate::resource::tests::WATERFALL_JOINED_BUFFER_WAVE32;
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 32,
            layout: vec![table(0), table(1)],
            used: vec![
                range(ResourceType::Srv, 0, 4, 4, 0),
                range(ResourceType::Uav, 16, 4, 0, 0),
                range(ResourceType::Sampler, 0, 4, 0, 1),
            ],
            io: StageIo::Cs(ComputeIo { rsrc2: 2 << 1, threads: [32, 1, 1] }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).expect("ir");
        let st = crate::structure::structure(&p).expect("structure");
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
        let io = crate::io::raise(&p, &native).expect("io");
        let bytes = crate::dxbc::build(&[]);
        let c = crate::Container::parse(&bytes).expect("container");
        crate::emit::emit(&c, &native, &p, &st, &io, &res).expect("emit")
    }

    fn le(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// A capture as `victory-gpu` writes one: two structured buffers and a
    /// raw UAV at view slots 3..6 (a table at byte offset 3 * 32), the
    /// buffers' bytes at offsets inside larger host copies.
    fn write_capture(dir: &Path) -> Capture {
        let s = shader();
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("shader.dxil"), &s.dxil).unwrap();
        let first: Vec<u32> = (0..32).map(|i| 100 + i).collect();
        let second: Vec<u32> = (0..32).map(|i| 200 + i).collect();
        std::fs::write(dir.join("buffer0.bin"), le(&first)).unwrap();
        std::fs::write(dir.join("buffer1.bin"), le(&second)).unwrap();
        std::fs::write(dir.join("buffer2.bin"), vec![0xEEu8; 128]).unwrap();
        let structured = |first_element: u32| {
            let d = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: u64::from(first_element),
                        NumElements: 32,
                        StructureByteStride: 4,
                        Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                    },
                },
            };
            // SAFETY: 40 bytes.
            unsafe { std::ptr::from_ref(&d).cast::<[u32; 10]>().read_unaligned() }
        };
        let raw = {
            let d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV { NumElements: 32, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() },
                },
            };
            // SAFETY: 40 bytes.
            unsafe { std::ptr::from_ref(&d).cast::<[u32; 10]>().read_unaligned() }
        };
        let point = D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxAnisotropy: 1,
            MaxLOD: f32::MAX,
            ..Default::default()
        };
        let cap = Capture {
            shader: "synthetic".into(),
            dxil: "shader.dxil".into(),
            container: None,
            layout: Layout::of(&s),
            guard: crate::emit::view_heap_guard(),
            strict_kinds: true,
            root_srv_tables: true,
            groups: Groups::Direct([1, 1, 1]),
            roots: vec![
                (0, Root::Table { sampler: false, offset: 3 * 32 }),
                (1, Root::Table { sampler: true, offset: 0 }),
            ],
            buffers: vec![
                // 32 dwords at 0x100 of a 0x1000-byte host copy.
                Buffer { id: 0, size: 0x1000, lo: 0x100, hi: 0x180, file: "buffer0.bin".into(), title: 0x10_0000 },
                Buffer { id: 1, size: 0x200, lo: 0, hi: 0x80, file: "buffer1.bin".into(), title: 0x20_0000 },
                Buffer { id: 2, size: 0x80, lo: 0, hi: 0x80, file: "buffer2.bin".into(), title: 0x30_0000 },
            ],
            views: vec![
                (3, Slot::Srv { buffer: 0, desc: structured(0x40) }),
                (4, Slot::Srv { buffer: 1, desc: structured(0) }),
                (5, Slot::Uav { buffer: 2, desc: raw }),
            ],
            // SAFETY: 52 bytes.
            samplers: vec![(0, unsafe { std::ptr::from_ref(&point).cast::<[u32; 13]>().read_unaligned() })],
            notes: vec!["synthetic: replay unit test".into()],
        };
        std::fs::write(dir.join(MANIFEST), cap.to_text()).unwrap();
        cap
    }

    #[test]
    fn a_synthetic_capture_replays_and_its_uav_is_dumped() {
        // `VICTORY_REPLAY_KEEP=<dir>` leaves the capture there (to try
        // `victory replay` on it).
        let dir = match std::env::var_os("VICTORY_REPLAY_KEEP") {
            Some(d) => PathBuf::from(d),
            None => std::env::temp_dir().join(format!("victory-replay-test-{}", std::process::id())),
        };
        write_capture(&dir);
        let out = dir.join("out");
        let o = Options { dump: Some(out.clone()), ..Default::default() };
        let report = match replay(&dir, &o) {
            Ok(r) => r,
            Err(e) if e.contains("SM") || e.contains("adapter") || e.contains("device") => {
                eprintln!("no device: {e}; skipped");
                return;
            }
            Err(e) => panic!("replay: {e}"),
        };
        for l in &report.lines {
            eprintln!("{l}");
        }
        assert!(report.removed.is_none(), "{report:?}");
        assert!(!report.timed_out);
        let got: Vec<u32> = std::fs::read(out.join("buffer2.bin"))
            .unwrap()
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // As the waterfall test: the first buffer below lane 16, the second
        // to 23, nothing stored from 24 on (the capture's 0xEE bytes stay).
        let want: Vec<u32> = (0..32)
            .map(|l| match l {
                0..16 => 100 + l,
                16..24 => 200 + l,
                _ => 0xEEEE_EEEE,
            })
            .collect();
        assert_eq!(got, want);
        if std::env::var_os("VICTORY_REPLAY_KEEP").is_none() {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
