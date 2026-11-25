//! Replays every capture under a directory, each in a child process (a
//! device removal, a hang or a crash of the driver must not end the run),
//! one line per capture and a summary: the regression suite of the
//! crash-fix loop (`victory replay-all`, `victory-replay`).

use std::ffi::OsString;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::MANIFEST;

/// How a capture's replay ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Ok,
    Removed,
    Timeout,
    /// The child failed some other way (its message, or how it ended).
    Error(String),
}

impl Outcome {
    pub fn label(&self) -> &'static str {
        match self {
            Outcome::Ok => "OK",
            Outcome::Removed => "DEVICE REMOVED",
            Outcome::Timeout => "TIMEOUT",
            Outcome::Error(_) => "ERROR",
        }
    }
}

/// One capture's result.
#[derive(Debug, Clone)]
pub struct Line {
    pub dir: PathBuf,
    pub outcome: Outcome,
    /// The child's verdict line (`OK: ...`, `DEVICE REMOVED ...`) or error.
    pub detail: String,
    pub elapsed: Duration,
}

/// The capture directories under `root`: `root` itself if it is one, else
/// its subdirectories (two levels deep) holding a manifest, sorted.
pub fn captures(root: &Path) -> Vec<PathBuf> {
    if root.join(MANIFEST).is_file() {
        return vec![root.to_owned()];
    }
    let mut out = Vec::new();
    let mut walk = |dir: &Path, deeper: &mut Vec<PathBuf>| {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if !p.is_dir() {
                continue;
            }
            if p.join(MANIFEST).is_file() {
                out.push(p);
            } else {
                deeper.push(p);
            }
        }
    };
    let mut next = Vec::new();
    walk(root, &mut next);
    for d in next {
        walk(&d, &mut Vec::new());
    }
    out.sort();
    out
}

/// Reads a child's verdict from its output and exit code (`victory
/// replay`: 0 ran, 2 device removed, 3 timed out).
pub fn verdict(stdout: &str, stderr: &str, code: Option<i32>) -> (Outcome, String) {
    let find = |p: &str| {
        stdout
            .lines()
            .rev()
            .find(|l| l.starts_with(p))
            .map(str::to_owned)
    };
    match code {
        Some(0) => (Outcome::Ok, find("OK").unwrap_or_else(|| "OK".into())),
        Some(2) => (
            Outcome::Removed,
            find("DEVICE REMOVED").unwrap_or_else(|| "DEVICE REMOVED".into()),
        ),
        Some(3) => (
            Outcome::Timeout,
            find("TIMEOUT").unwrap_or_else(|| "TIMEOUT".into()),
        ),
        _ => {
            let why = stderr
                .lines()
                .rev()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .or_else(|| stdout.lines().rev().map(str::trim).find(|l| !l.is_empty()))
                .unwrap_or("no output")
                .to_owned();
            let how = match code {
                Some(c) => format!("exit {c:#x}"),
                None => "killed".into(),
            };
            (Outcome::Error(why.clone()), format!("{how}: {why}"))
        }
    }
}

/// Runs `program args... <dir>` for one capture, killed after `limit`.
pub fn run_one(program: &Path, args: &[OsString], dir: &Path, limit: Duration) -> Line {
    let started = Instant::now();
    let child = Command::new(program)
        .args(args)
        .arg(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            return Line {
                dir: dir.to_owned(),
                outcome: Outcome::Error(e.to_string()),
                detail: format!("cannot start {}: {e}", program.display()),
                elapsed: started.elapsed(),
            };
        }
    };
    // Drained on threads: a chatty child must not block on a full pipe;
    // and a killed child's own children may hold the pipes open, so the
    // output is not waited for long after it ends.
    let drain = |r: Option<Box<dyn std::io::Read + Send>>| {
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut s = Vec::new();
            if let Some(mut r) = r {
                let _ = r.read_to_end(&mut s);
            }
            let _ = send.send(String::from_utf8_lossy(&s).into_owned());
        });
        receive
    };
    let out = drain(child.stdout.take().map(|r| Box::new(r) as _));
    let err = drain(child.stderr.take().map(|r| Box::new(r) as _));
    let mut killed = false;
    let code = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status.code(),
            Ok(None) if started.elapsed() > limit => {
                let _ = child.kill();
                let _ = child.wait();
                killed = true;
                break None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(5)),
            Err(_) => break None,
        }
    };
    let elapsed = started.elapsed();
    let wait = Duration::from_millis(if killed { 200 } else { 2000 });
    let stdout = out.recv_timeout(wait).unwrap_or_default();
    let stderr = err.recv_timeout(wait).unwrap_or_default();
    let (outcome, detail) = if killed {
        (
            Outcome::Timeout,
            format!("TIMEOUT: child killed after {limit:?}"),
        )
    } else {
        verdict(&stdout, &stderr, code)
    };
    Line {
        dir: dir.to_owned(),
        outcome,
        detail,
        elapsed,
    }
}

/// Replays every capture under `root` with `program args... <dir>`,
/// `jobs` children at a time, printing a line per capture (in order) and a
/// summary. Returns the lines.
///
/// With more than one job, a capture that does not end OK is replayed again
/// alone before it is reported: a device removal or hang in one child can
/// take the other children's devices with it.
pub fn run_all(
    program: &Path,
    args: &[OsString],
    root: &Path,
    limit: Duration,
    jobs: usize,
) -> Vec<Line> {
    let dirs = captures(root);
    let started = Instant::now();
    if dirs.is_empty() {
        println!(
            "no captures (directories holding {MANIFEST}) under {}",
            root.display()
        );
        return Vec::new();
    }
    let jobs = jobs.clamp(1, dirs.len());
    let next = AtomicUsize::new(0);
    let (send, receive) = std::sync::mpsc::channel::<(usize, Line)>();
    let mut done: Vec<Option<Line>> = vec![None; dirs.len()];
    let mut lines: Vec<Line> = Vec::new();
    std::thread::scope(|s| {
        for _ in 0..jobs {
            let send = send.clone();
            let (next, dirs) = (&next, &dirs);
            s.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(dir) = dirs.get(i) else { break };
                    if send.send((i, run_one(program, args, dir, limit))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(send);
        for (i, line) in receive {
            done[i] = Some(line);
            while let Some(line) = done.get_mut(lines.len()).and_then(Option::take) {
                let line = settle(program, args, limit, line, jobs > 1, lines.last());
                print_line(root, &line);
                lines.push(line);
            }
        }
    });
    summary(&lines, started);
    lines
}

/// A result confirmed: replayed alone if it ran beside others and did not
/// end OK, and once more if the device was not back from a reset yet.
fn settle(
    program: &Path,
    args: &[OsString],
    limit: Duration,
    mut line: Line,
    parallel: bool,
    previous: Option<&Line>,
) -> Line {
    let dir = line.dir.clone();
    if parallel && line.outcome != Outcome::Ok {
        line = run_one(program, args, &dir, limit);
        line.detail.push_str(" (confirmed alone)");
    }
    if matches!(&line.outcome, Outcome::Error(e) if e.contains("device") || e.contains("adapter"))
        && (parallel || previous.is_some_and(|l| l.outcome != Outcome::Ok))
    {
        std::thread::sleep(Duration::from_secs(1));
        line = run_one(program, args, &dir, limit);
    }
    line
}

fn print_line(root: &Path, line: &Line) {
    let name = line
        .dir
        .strip_prefix(root)
        .ok()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(&line.dir);
    println!(
        "{:<15} {:>7} ms  {}  {}",
        line.outcome.label(),
        line.elapsed.as_millis(),
        name.display(),
        line.detail
    );
}

fn summary(lines: &[Line], started: Instant) {
    let count = |o: fn(&Outcome) -> bool| lines.iter().filter(|l| o(&l.outcome)).count();
    println!(
        "{} captures: {} OK, {} device removed, {} timeout, {} error; {} ms",
        lines.len(),
        count(|o| *o == Outcome::Ok),
        count(|o| *o == Outcome::Removed),
        count(|o| *o == Outcome::Timeout),
        count(|o| matches!(o, Outcome::Error(_))),
        started.elapsed().as_millis()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdicts_follow_exit_codes() {
        let (o, d) = verdict("capture of shader x\nOK: finished in 1.0 ms\n", "", Some(0));
        assert_eq!((o, d.as_str()), (Outcome::Ok, "OK: finished in 1.0 ms"));
        assert_eq!(
            verdict("DEVICE REMOVED after 3 ms: hung\n", "", Some(2)).0,
            Outcome::Removed
        );
        assert_eq!(verdict("", "", Some(3)).0, Outcome::Timeout);
        let (o, d) = verdict("", "Error: retranslation: bad\n", Some(1));
        assert_eq!(o, Outcome::Error("Error: retranslation: bad".into()));
        assert!(d.starts_with("exit 0x1"));
    }

    #[cfg(windows)]
    #[test]
    fn a_hung_child_is_killed() {
        let cmd = std::path::PathBuf::from(
            std::env::var_os("ComSpec").unwrap_or_else(|| "cmd.exe".into()),
        );
        let args: Vec<OsString> = ["/c", "ping", "-n", "30", "127.0.0.1", "&&", "rem"]
            .map(Into::into)
            .to_vec();
        let started = Instant::now();
        let line = run_one(&cmd, &args, Path::new("x"), Duration::from_millis(300));
        assert_eq!(line.outcome, Outcome::Timeout, "{line:?}");
        assert!(started.elapsed() < Duration::from_secs(10));
        let ok: Vec<OsString> = ["/c", "echo", "OK:fine", "&&", "rem"]
            .map(Into::into)
            .to_vec();
        let line = run_one(&cmd, &ok, Path::new("x"), Duration::from_secs(10));
        assert_eq!((line.outcome, line.detail.trim()), (Outcome::Ok, "OK:fine"));
    }

    #[test]
    fn capture_directories_are_found() {
        let root = std::env::temp_dir().join(format!("victory-suite-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for d in ["b", "a", "group/c", "empty"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        for d in ["a", "b", "group/c"] {
            std::fs::write(root.join(d).join(MANIFEST), "version 1\n").unwrap();
        }
        let found: Vec<PathBuf> = captures(&root)
            .iter()
            .map(|p| p.strip_prefix(&root).unwrap().to_owned())
            .collect();
        assert_eq!(
            found,
            [
                PathBuf::from("a"),
                "b".into(),
                PathBuf::from("group").join("c")
            ]
        );
        assert_eq!(captures(&root.join("a")), [root.join("a")]);
        let _ = std::fs::remove_dir_all(&root);
    }
}
