//! Hot reload of the shader translator (diagnostics): with
//! `VICTORY_HOT_SHADERS=<path to victory_shader_dylib.dll>` shaders are
//! translated by that DLL (crate `victory-shader-dylib`) instead of the
//! translator linked in, and the on-disk shader cache is not used. A watcher
//! polls the DLL (or `VICTORY_HOT_SHADERS_TRIGGER`, a file
//! `tools/hot-shaders.ps1` touches after a build) every half second; on a
//! change it loads a copy of the new DLL (so cargo can write the original
//! again) and starts a new generation:
//!
//! - every shader is translated again on its next use, by the new DLL;
//! - every host pipeline is made again, from those translations: the ones
//!   drawn with at once (in the background), the rest on their next use.
//!
//! Until its new host pipeline is made a pipeline keeps drawing with the
//! old one, and the old pipeline's stages keep their old translations, so
//! a draw never mixes generations. Old host pipelines, translations and
//! DLLs are leaked, never freed: lists in flight may still use them.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime};

use victory_shader::emit::{HostShader, Target};
use victory_shader::{ResourceType, UsedResource};

use super::Translated;

/// The interface version of `victory-shader-dylib`.
const ABI: u32 = 3;

static GENERATION: AtomicU64 = AtomicU64::new(0);

/// Reloads so far: 0 until the first. Translations and host pipelines are
/// made per generation.
pub(crate) fn generation() -> u64 {
    GENERATION.load(Ordering::Acquire)
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Guard {
    on: u32,
    limit: u32,
    kinds_slot: u32,
    null_base: u32,
}

type AbiFn = unsafe extern "C" fn() -> u32;
type TranslateFn = unsafe extern "C" fn(*const u8, usize, u32, u32, *const Guard, *mut *mut u8, *mut usize) -> i32;
type FreeFn = unsafe extern "C" fn(*mut u8, usize);

/// A loaded translator DLL (never unloaded).
pub(crate) struct Dll {
    translate: TranslateFn,
    free: FreeFn,
    /// The loaded copy, and the BLAKE3 of the DLL.
    pub(crate) path: PathBuf,
    hash: [u8; 32],
}

/// The DLL translations go through now.
static CURRENT: Mutex<Option<&'static Dll>> = Mutex::new(None);

/// Whether hot reload is on; the first call loads the DLL and starts the
/// watcher.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let Some(path) = std::env::var_os("VICTORY_HOT_SHADERS").map(PathBuf::from) else {
            return false;
        };
        match load_copy(&path, 0) {
            Ok(dll) => {
                tracing::warn!(dll = %path.display(), copy = %dll.path.display(), "VICTORY_HOT_SHADERS: shaders translated by a reloadable DLL; the shader cache is not used");
                *CURRENT.lock().unwrap_or_else(|e| e.into_inner()) = Some(dll);
            }
            Err(e) => {
                // Still on: the watcher loads it once it is there.
                tracing::error!(dll = %path.display(), error = %e, "VICTORY_HOT_SHADERS: no translator DLL yet; shaders do not translate until it loads");
            }
        }
        let trigger = std::env::var_os("VICTORY_HOT_SHADERS_TRIGGER").map(PathBuf::from);
        let spawned = std::thread::Builder::new()
            .name("victory-hot-shaders".into())
            .spawn(move || watch(&path, trigger.as_deref()));
        if let Err(e) = spawned {
            tracing::error!(error = %e, "no hot shader watcher");
        }
        true
    })
}

/// The translation of `container` by the current DLL; `None` when hot
/// reload is off.
pub(crate) fn translate(container: &[u8], target: Target) -> Option<Result<Translated, String>> {
    if !enabled() {
        return None;
    }
    let dll = *CURRENT.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dll) = dll else {
        return Some(Err("VICTORY_HOT_SHADERS: no translator DLL loaded".into()));
    };
    let guard = victory_shader::emit::view_heap_guard().map_or(Guard::default(), |g| Guard {
        on: 1,
        limit: g.limit,
        kinds_slot: g.kinds_slot,
        null_base: g.null_base,
    });
    Some(dll.translate(container, target, guard))
}

impl Dll {
    fn translate(&self, container: &[u8], target: Target, guard: Guard) -> Result<Translated, String> {
        let mut out = std::ptr::null_mut();
        let mut len = 0;
        // SAFETY: the interface of `victory-shader-dylib` (version checked
        // at load).
        let status = unsafe {
            (self.translate)(
                container.as_ptr(),
                container.len(),
                target.lanes_min,
                target.lanes_max,
                &guard,
                &mut out,
                &mut len,
            )
        };
        let bytes = if out.is_null() {
            Vec::new()
        } else {
            // SAFETY: the DLL hands out `len` bytes, ours until freed.
            let v = unsafe { std::slice::from_raw_parts(out, len) }.to_vec();
            // SAFETY: once, as it was handed out.
            unsafe { (self.free)(out, len) };
            v
        };
        match status {
            0 => decode(&bytes).ok_or_else(|| "VICTORY_HOT_SHADERS: undecodable translation from the DLL".into()),
            _ => Err(String::from_utf8_lossy(&bytes).into_owned()),
        }
    }
}

/// Copies `path` to a fresh name in the temporary folder and loads it.
fn load_copy(path: &Path, generation: u64) -> Result<&'static Dll, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let hash = *blake3::hash(&bytes).as_bytes();
    let dir = std::env::temp_dir().join("victory-hot-shaders");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let copy = dir.join(format!(
        "victory_shader_{}_{generation}_{}.dll",
        std::process::id(),
        &blake3::Hash::from(hash).to_hex()[..12]
    ));
    std::fs::write(&copy, &bytes).map_err(|e| format!("cannot write {}: {e}", copy.display()))?;
    load(&copy, hash)
}

/// Loads the DLL at `path` (leaked: it is never unloaded).
pub(crate) fn load(path: &Path, hash: [u8; 32]) -> Result<&'static Dll, String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    // SAFETY: a NUL-terminated path.
    let module = unsafe { LoadLibraryW(wide.as_ptr()) };
    if module.is_null() {
        return Err(format!("LoadLibrary({}) failed: {}", path.display(), std::io::Error::last_os_error()));
    }
    let symbol = |name: &[u8]| {
        // SAFETY: a loaded module and a NUL-terminated name.
        unsafe { GetProcAddress(module, name.as_ptr()) }.ok_or_else(|| {
            format!("{} lacks {}", path.display(), String::from_utf8_lossy(&name[..name.len() - 1]))
        })
    };
    // SAFETY: the symbols' types are the interface of `victory-shader-dylib`.
    let (abi, translate, free) = unsafe {
        (
            std::mem::transmute::<_, AbiFn>(symbol(b"victory_hot_abi\0")?),
            std::mem::transmute::<_, TranslateFn>(symbol(b"victory_hot_translate\0")?),
            std::mem::transmute::<_, FreeFn>(symbol(b"victory_hot_free\0")?),
        )
    };
    // SAFETY: as above.
    let version = unsafe { abi() };
    if version != ABI {
        return Err(format!("{} has interface {version}, not {ABI}", path.display()));
    }
    Ok(Box::leak(Box::new(Dll { translate, free, path: path.to_owned(), hash })))
}

/// Polls the DLL (or the trigger) and reloads on a change.
fn watch(dll: &Path, trigger: Option<&Path>) {
    let watched = trigger.unwrap_or(dll);
    let stamp = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let mut seen = stamp(watched);
    loop {
        std::thread::sleep(Duration::from_millis(500));
        let now = stamp(watched);
        if now.is_none() || now == seen {
            continue;
        }
        // Wait for the writer (the linker) to finish: both unchanged for
        // 300 ms.
        let mut last: (Option<SystemTime>, Option<SystemTime>) = (stamp(dll), stamp(watched));
        loop {
            std::thread::sleep(Duration::from_millis(300));
            let again = (stamp(dll), stamp(watched));
            if again == last {
                break;
            }
            last = again;
        }
        seen = last.1;
        reload(dll);
    }
}

/// Loads the DLL anew and starts a new generation.
fn reload(path: &Path) {
    let started = std::time::Instant::now();
    let generation = generation() + 1;
    let mut tries = 0;
    let dll = loop {
        match load_copy(path, generation) {
            Ok(dll) => break dll,
            Err(e) if tries < 10 => {
                tries += 1;
                tracing::debug!(error = %e, "hot shader reload: DLL not ready; retrying");
                std::thread::sleep(Duration::from_millis(300));
            }
            Err(e) => {
                tracing::error!(error = %e, "hot shader reload failed; the old translator stays");
                return;
            }
        }
    };
    {
        let mut current = CURRENT.lock().unwrap_or_else(|e| e.into_inner());
        if current.is_some_and(|c| c.hash == dll.hash) {
            tracing::info!(dll = %path.display(), "hot shader reload: the DLL is unchanged; nothing invalidated");
            return;
        }
        *current = Some(dll);
    }
    GENERATION.store(generation, Ordering::Release);
    let shaders = super::translated_shaders();
    let (requeued, others) = super::make::remake_all(generation);
    tracing::warn!(
        generation,
        dll = %dll.path.display(),
        requeued,
        on_next_use = others,
        ms = started.elapsed().as_millis() as u64,
        "hot shader reload: {shaders} shaders, {} pipelines invalidated",
        requeued + others
    );
}

/// The DLL's encoding (`victory-shader-dylib`): `VHT1`, a length and the
/// shader cache's entry, then the used resources.
fn decode(bytes: &[u8]) -> Option<Translated> {
    let word = |b: &[u8], at: usize| Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?));
    if bytes.get(..4)? != b"VHT1" {
        return None;
    }
    let n = word(bytes, 4)? as usize;
    let host: HostShader = crate::shader_cache::decode(bytes.get(8..8 + n)?)?;
    let rest = bytes.get(8 + n..)?;
    let count = word(rest, 0)? as usize;
    if rest.len() != 4 + count * 32 {
        return None;
    }
    let mut used = Vec::with_capacity(count);
    for i in 0..count {
        let w: Vec<u32> = (0..8).map(|j| word(rest, 4 + i * 32 + j * 4)).collect::<Option<_>>()?;
        used.push(UsedResource {
            kind: match w[0] {
                0 => ResourceType::Srv,
                1 => ResourceType::Sampler,
                2 => ResourceType::Cbv,
                3 => ResourceType::Uav,
                x => ResourceType::Other(u8::try_from(x - 4).ok()?),
            },
            root: crate::shader_cache::root_type(w[1]),
            table_offset_dwords: w[2],
            lower_bound: w[3],
            stride: w[4],
            dwords: u8::try_from(w[5]).ok()?,
            user_data: u8::try_from(w[6]).ok()?,
            slot: u8::try_from(w[7]).ok()?,
        });
    }
    // Unknown from the DLL: what shaders store is copied back.
    Some(Translated { host, used, stores: true })
}

#[cfg(test)]
mod tests {
    //! The DLL against the translator linked in. Needs the DLL built
    //! (`cargo build --profile fastfix -p victory-shader-dylib`, or
    //! `VICTORY_HOT_TEST_DLL`) and containers (`VICTORY_HOT_TEST_CONTAINERS`,
    //! a folder of `*.bin` as `VICTORY_SHADER_DUMP` writes them); skipped
    //! without either.
    use super::*;

    #[test]
    fn the_dll_translates_as_the_linked_translator() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dll = std::env::var_os("VICTORY_HOT_TEST_DLL")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("target/fastfix/victory_shader_dylib.dll"));
        let Some(dir) = std::env::var_os("VICTORY_HOT_TEST_CONTAINERS") else {
            eprintln!("VICTORY_HOT_TEST_CONTAINERS unset; skipped");
            return;
        };
        if !dll.exists() {
            eprintln!("{} not built; skipped", dll.display());
            return;
        }
        let dll = load_copy(&dll, 0).expect("the DLL loads");
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "bin"))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no containers");
        let targets = [Target::ANY, Target { lanes_min: 32, lanes_max: 32 }];
        let mut translated = 0;
        for file in &files {
            let container = std::fs::read(file).unwrap();
            for target in targets {
                let ours = super::super::translate_linked(&container, target);
                let theirs = dll.translate(&container, target, Guard::default());
                match (&ours, &theirs) {
                    (Ok(a), Ok(b)) => {
                        assert_eq!(a.host, b.host, "{} {target:?}", file.display());
                        assert_eq!(a.used, b.used, "{} {target:?}", file.display());
                        translated += 1;
                    }
                    (Err(a), Err(b)) => assert_eq!(a, b, "{}", file.display()),
                    _ => panic!(
                        "{} {target:?}: linked {:?}, DLL {:?}",
                        file.display(),
                        ours.as_ref().err(),
                        theirs.as_ref().err()
                    ),
                }
            }
        }
        eprintln!("{} containers, {translated} translations equal", files.len());
        assert!(translated > 0);
    }

    #[test]
    fn damaged_encodings_are_refused() {
        assert!(decode(b"VHT1").is_none());
        assert!(decode(b"VHT0\0\0\0\0").is_none());
    }
}
