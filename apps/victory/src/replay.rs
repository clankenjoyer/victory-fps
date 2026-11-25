//! `victory replay <capture dir>`: runs one compute dispatch a diagnostics
//! build captured (`VICTORY_CAPTURE=<shader hash>[,<n>]`, into
//! `VICTORY_CAPTURE_DIR`) on a fresh D3D12 device and reports whether the
//! device survived. Exit status: 0 ran, 2 device removed, 3 timed out.

use std::path::PathBuf;

use anyhow::Result;

#[derive(clap::Args)]
pub struct Args {
    /// A capture directory (holding `capture.txt`).
    dir: PathBuf,
    /// Translate the captured console container with this build's
    /// translator instead of running the captured DXIL.
    #[arg(long)]
    retranslate: bool,
    /// The console container for --retranslate when the capture has none
    /// (`compute_<hash>.bin` from a `VICTORY_SHADER_DUMP` directory).
    #[arg(long, value_name = "FILE")]
    container: Option<PathBuf>,
    /// Write every captured buffer's bytes after the dispatch to this
    /// directory (`buffer<id>.bin`), for diffing translator versions.
    #[arg(long, value_name = "DIR")]
    dump: Option<PathBuf>,
    /// Run on WARP.
    #[arg(long)]
    warp: bool,
    /// The first hardware adapter whose name holds this (`nvidia`, ...).
    #[arg(long)]
    adapter: Option<String>,
    /// Enable the D3D12 debug layer.
    #[arg(long)]
    debug_layer: bool,
    /// Device-removed data: the faulting address, placed in the replay's
    /// buffers and heap.
    #[arg(long)]
    dred: bool,
    /// Seconds to wait for the GPU.
    #[arg(long, default_value_t = 10)]
    timeout: u64,
    /// Submit the dispatch this many times.
    #[arg(long, default_value_t = 1)]
    repeat: u32,
    /// Buffer views serve reads of every buffer kind (as on AMD).
    #[arg(long)]
    lax_kinds: bool,
}

#[cfg(windows)]
pub fn run(a: Args) -> Result<()> {
    use victory_shader::replay::run::{Options, replay};
    let o = Options {
        retranslate: a.retranslate,
        container: a.container,
        warp: a.warp,
        adapter: a.adapter,
        dump: a.dump,
        debug_layer: a.debug_layer,
        dred: a.dred,
        timeout: Some(std::time::Duration::from_secs(a.timeout)),
        repeat: a.repeat,
        lax_kinds: a.lax_kinds,
    };
    let report = replay(&a.dir, &o).map_err(anyhow::Error::msg)?;
    for line in &report.lines {
        println!("{line}");
    }
    if report.removed.is_some() {
        std::process::exit(2);
    }
    if report.timed_out {
        std::process::exit(3);
    }
    Ok(())
}

#[cfg(not(windows))]
pub fn run(_: Args) -> Result<()> {
    anyhow::bail!("replay needs host D3D12 (Windows)")
}

/// `victory replay-all <dir>`: every capture under a directory, each
/// replayed by a child `victory replay` (a device removal or hang ends only
/// the child). Exit status 0 only if all ran.
#[derive(clap::Args)]
pub struct AllArgs {
    /// A directory of capture directories (or one capture).
    dir: PathBuf,
    /// Retranslate each capture's console container with this build.
    #[arg(long)]
    retranslate: bool,
    /// Seconds each replay waits for the GPU (the child is killed a few
    /// seconds after).
    #[arg(long, default_value_t = 10)]
    timeout: u64,
    /// Run on WARP.
    #[arg(long)]
    warp: bool,
    /// Enable the D3D12 debug layer in each replay.
    #[arg(long)]
    debug_layer: bool,
    /// Buffer views serve reads of every buffer kind (as on AMD).
    #[arg(long)]
    lax_kinds: bool,
    /// Replays at a time (a capture that fails beside others is replayed
    /// again alone).
    #[arg(long, short, default_value_t = 1)]
    jobs: usize,
}

pub fn run_all(a: AllArgs) -> Result<()> {
    use std::ffi::OsString;
    let exe = std::env::current_exe()?;
    let mut args: Vec<OsString> = vec!["replay".into(), "--timeout".into(), a.timeout.to_string().into()];
    for (on, flag) in [
        (a.retranslate, "--retranslate"),
        (a.warp, "--warp"),
        (a.debug_layer, "--debug-layer"),
        (a.lax_kinds, "--lax-kinds"),
    ] {
        if on {
            args.push(flag.into());
        }
    }
    // Device creation, shader compilation and buffer upload on top of the
    // GPU wait.
    let limit = std::time::Duration::from_secs(a.timeout + 20);
    let lines = victory_shader::replay::suite::run_all(&exe, &args, &a.dir, limit, a.jobs);
    if lines.is_empty() || lines.iter().any(|l| l.outcome != victory_shader::replay::suite::Outcome::Ok) {
        std::process::exit(1);
    }
    Ok(())
}
