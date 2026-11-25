//! Where every thread is: `VICTORY_STACKS=<seconds>[,<seconds>...]` prints
//! each thread's unwound stack as `module+rva` that many seconds after
//! [`dump_stacks_on_schedule`], to find what a title waits on.

use std::ffi::c_void;

use windows_sys::Wdk::System::Threading::{
    NtQueryInformationThread, ThreadBasicInformation, ThreadQuerySetWin32StartAddress,
};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, LocalFree};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_FULL_AMD64, GetThreadContext,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcessId, GetCurrentThreadId, GetThreadDescription, OpenThread, ResumeThread,
    SetThreadAffinityMask, SetThreadPriority, SuspendThread, THREAD_GET_CONTEXT, THREAD_PRIORITY_IDLE,
    THREAD_QUERY_INFORMATION, THREAD_SET_INFORMATION, THREAD_SUSPEND_RESUME,
};

use crate::crash::{MAX_FRAMES, locate, unwind_into};

/// Reads `VICTORY_STACKS` and, if set, starts a thread that prints every
/// thread's stack at each listed time (seconds from now).
pub fn dump_stacks_on_schedule() {
    profile_on_schedule();
    calm_background_threads();
    let Ok(list) = std::env::var("VICTORY_STACKS") else {
        return;
    };
    let mut times: Vec<f64> = list
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    times.sort_by(f64::total_cmp);
    let start = std::time::Instant::now();
    std::thread::spawn(move || {
        for t in times {
            let at = std::time::Duration::from_secs_f64(t.max(0.0));
            std::thread::sleep(at.saturating_sub(start.elapsed()));
            dump_stacks();
        }
    });
}

/// `THREAD_BASIC_INFORMATION` (ntddk).
#[repr(C)]
struct BasicInfo {
    exit_status: i32,
    teb: *const usize,
    client_id: [usize; 2],
    affinity: usize,
    priority: i32,
    base_priority: i32,
}

/// `GetThreadContext` wants a 16-byte aligned context.
#[repr(C, align(16))]
struct Aligned(CONTEXT);

/// Prints every other thread's stack to stderr. Each thread is suspended
/// only while its context is read and unwound into a buffer allocated
/// beforehand (it may hold the heap lock).
pub fn dump_stacks() {
    // SAFETY: plain queries and a snapshot of this process's threads.
    let (me, process, snapshot) = unsafe {
        (
            GetCurrentThreadId(),
            GetCurrentProcessId(),
            CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0),
        )
    };
    if snapshot == INVALID_HANDLE_VALUE {
        return;
    }
    let mut ids = Vec::new();
    // SAFETY: zeroed is a valid THREADENTRY32; its size is set before use.
    let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<THREADENTRY32>() as u32;
    // SAFETY: `snapshot` is open; `entry` is sized.
    let mut more = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while more {
        if entry.th32OwnerProcessID == process && entry.th32ThreadID != me {
            ids.push(entry.th32ThreadID);
        }
        // SAFETY: as above.
        more = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    // SAFETY: we opened it.
    unsafe { CloseHandle(snapshot) };
    eprintln!("VIctory: stacks of {} threads", ids.len());
    for id in ids {
        // SAFETY: opens a thread of our own process.
        let thread = unsafe {
            OpenThread(
                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                0,
                id,
            )
        };
        if thread.is_null() {
            continue;
        }
        let (frames, (priority, base)) = frames_of(thread);
        eprintln!(
            "  thread {id} \"{}\" start {} priority {priority} (base {base})",
            name_of(thread),
            locate(start_of(thread))
        );
        for (i, pc) in frames.into_iter().enumerate() {
            eprintln!("    #{i:<2} {}", locate(pc as usize));
        }
        // SAFETY: we opened it.
        unsafe { CloseHandle(thread) };
    }
}

/// The thread's frames and its current and base scheduling priority (0-31).
fn frames_of(thread: HANDLE) -> (Vec<u64>, (i32, i32)) {
    let mut frames = Vec::with_capacity(MAX_FRAMES);
    let mut info: BasicInfo = BasicInfo {
        exit_status: 0,
        teb: std::ptr::null(),
        client_id: [0; 2],
        affinity: 0,
        priority: 0,
        base_priority: 0,
    };
    // SAFETY: writes `info`, whose layout and size are the system's.
    let status = unsafe {
        NtQueryInformationThread(
            thread,
            ThreadBasicInformation,
            (&raw mut info).cast(),
            size_of::<BasicInfo>() as u32,
            std::ptr::null_mut(),
        )
    };
    if status != 0 || info.teb.is_null() {
        return (frames, (0, 0));
    }
    // SAFETY: the TEB of a live thread of this process: its NT_TIB holds
    // the stack base at +8 and the stack limit at +0x10.
    let (high, low) = unsafe { (info.teb.add(1).read(), info.teb.add(2).read()) };
    // SAFETY: zeroed is a valid CONTEXT.
    let mut context = Aligned(unsafe { std::mem::zeroed() });
    context.0.ContextFlags = CONTEXT_FULL_AMD64;
    // SAFETY: suspends, reads and resumes a thread that is not this one;
    // nothing between suspend and resume allocates or takes a lock the
    // thread could hold besides the unwinder's own.
    unsafe {
        if SuspendThread(thread) == u32::MAX {
            return (frames, (info.priority, info.base_priority));
        }
        if GetThreadContext(thread, &mut context.0) != 0 {
            unwind_into(context.0, low, high, &mut frames);
        }
        ResumeThread(thread);
    }
    (frames, (info.priority, info.base_priority))
}

fn start_of(thread: HANDLE) -> usize {
    let mut start = 0usize;
    // SAFETY: writes one pointer-sized start address.
    unsafe {
        NtQueryInformationThread(
            thread,
            ThreadQuerySetWin32StartAddress,
            (&raw mut start).cast(),
            size_of::<usize>() as u32,
            std::ptr::null_mut(),
        )
    };
    start
}

fn name_of(thread: HANDLE) -> String {
    let mut name = std::ptr::null_mut();
    // SAFETY: the system allocates the string; we free it with LocalFree.
    unsafe {
        if GetThreadDescription(thread, &mut name) < 0 || name.is_null() {
            return String::new();
        }
        let len = (0..).take_while(|&i| *name.add(i) != 0).count();
        let text = String::from_utf16_lossy(std::slice::from_raw_parts(name, len));
        LocalFree(name.cast::<c_void>());
        text
    }
}

/// `VICTORY_PROFILE=<start>,<seconds>[,<interval ms>]`: from `start` seconds
/// on, samples the threads' stacks each interval (2 ms unless given) for
/// `seconds`, then writes per thread the stacks it was seen in most, with
/// their share of its samples, to `VICTORY_PROFILE_FILE` (or stderr): where
/// a title's frame goes, waits included. A thread seen in one place only
/// during the first rounds is waiting and is left out from then on.
///
/// `VICTORY_PROFILE_NOW_FILE=<path>`: writing a number of seconds there
/// (checked each second; emptied when taken) profiles for that long from
/// then on, so one run serves several profiles; the second and later ones
/// go to the profile file with their number in its extension.
fn profile_on_schedule() {
    let trigger = std::env::var_os("VICTORY_PROFILE_NOW_FILE");
    let text = std::env::var("VICTORY_PROFILE").unwrap_or_default();
    if text.is_empty() && trigger.is_none() {
        return;
    }
    let v: Vec<f64> = text
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let start = v.first().copied().unwrap_or(0.0);
    let mut seconds = v.get(1).copied().unwrap_or(0.0);
    let interval = std::time::Duration::from_secs_f64(v.get(2).copied().unwrap_or(2.0) / 1000.0);
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs_f64(start.max(0.0)));
        let mut turn = 0u32;
        loop {
            if seconds <= 0.0 {
                let Some(path) = &trigger else { return };
                std::thread::sleep(std::time::Duration::from_secs(1));
                let asked = std::fs::read_to_string(path)
                    .ok()
                    .and_then(|t| t.trim().parse::<f64>().ok());
                if let Some(s) = asked {
                    let _ = std::fs::write(path, "");
                    seconds = s;
                }
                continue;
            }
            let out = profile(seconds, interval);
            match std::env::var_os("VICTORY_PROFILE_FILE") {
                Some(path) => {
                    let mut path = std::path::PathBuf::from(path);
                    if turn > 0 {
                        path.set_extension(format!("{turn}.txt"));
                    }
                    drop(std::fs::write(path, out));
                }
                None => eprint!("{out}"),
            }
            turn += 1;
            seconds = 0.0;
        }
    });
}

/// Samples for `seconds` and returns the report.
fn profile(seconds: f64, interval: std::time::Duration) -> String {
    use std::collections::{BTreeMap, HashMap};
    // SAFETY: plain queries.
    let (me, process) = unsafe { (GetCurrentThreadId(), GetCurrentProcessId()) };
    let mut threads: HashMap<u32, HANDLE> = HashMap::new();
    let mut seen: HashMap<(u32, Vec<u64>), u32> = HashMap::new();
    let mut totals: HashMap<u32, u32> = HashMap::new();
    let mut idle: Vec<u32> = Vec::new();
    let begun = std::time::Instant::now();
    let mut round = 0u32;
    while begun.elapsed().as_secs_f64() < seconds {
        if round.is_multiple_of(200) {
            for id in thread_ids(process, me) {
                threads.entry(id).or_insert_with(|| {
                    // SAFETY: opens a thread of our own process.
                    unsafe {
                        OpenThread(
                            THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                            0,
                            id,
                        )
                    }
                });
            }
        }
        round += 1;
        if round == 20 {
            let mut distinct: HashMap<u32, u32> = HashMap::new();
            for (id, _) in seen.keys() {
                *distinct.entry(*id).or_default() += 1;
            }
            idle = threads
                .keys()
                .copied()
                .filter(|id| distinct.get(id).copied().unwrap_or(0) <= 1)
                .collect();
        }
        for (&id, &thread) in &threads {
            if thread.is_null() || idle.contains(&id) {
                continue;
            }
            let (mut frames, _) = frames_of(thread);
            if frames.is_empty() {
                continue;
            }
            frames.truncate(44);
            *seen.entry((id, frames)).or_default() += 1;
            *totals.entry(id).or_default() += 1;
        }
        std::thread::sleep(interval);
    }
    let mut by_thread: BTreeMap<u32, Vec<(u32, Vec<u64>)>> = BTreeMap::new();
    for ((id, frames), n) in seen {
        if !idle.contains(&id) {
            by_thread.entry(id).or_default().push((n, frames));
        }
    }
    let mut out = format!(
        "VIctory: profile of {round} rounds over {seconds} s; {} threads left out as waiting\n",
        idle.len()
    );
    for (id, mut stacks) in by_thread {
        stacks.sort_by(|a, b| b.0.cmp(&a.0));
        let total = totals[&id].max(1);
        let thread = threads[&id];
        out += &format!(
            "  thread {id} \"{}\" start {} samples {total}, {} distinct stacks\n",
            name_of(thread),
            locate(start_of(thread)),
            stacks.len()
        );
        for (n, frames) in stacks.into_iter().take(40) {
            if n * 200 < total {
                break;
            }
            out += &format!("    {:5.1}%", f64::from(n) * 100.0 / f64::from(total));
            for pc in frames {
                out += &format!(" {}", locate(pc as usize));
            }
            out.push('\n');
        }
    }
    for (_, thread) in threads {
        if !thread.is_null() {
            // SAFETY: we opened it.
            unsafe { CloseHandle(thread) };
        }
    }
    out
}

/// The ids of this process's threads other than `me`.

/// The host runtime's "D3D Background Thread"s (the driver compiling
/// pipelines' shaders again, flat out for the whole run: four threads on
/// a machine of four cores, run 105's profile) are put at idle priority,
/// so they take only what the title's threads leave. Looked for every two
/// seconds; `VICTORY_D3D12_BACKGROUND_IDLE=0` leaves them as they are.
/// (Switching the work off, `VICTORY_D3D12_BACKGROUND=0`, needs the
/// system's developer mode.)
fn calm_background_threads() {
    if std::env::var_os("VICTORY_D3D12_BACKGROUND_IDLE").is_some_and(|v| v == "0") {
        return;
    }
    std::thread::spawn(|| {
        let mut calmed = std::collections::HashSet::new();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(2));
            // SAFETY: plain system calls.
            let (process, me) = unsafe { (GetCurrentProcessId(), GetCurrentThreadId()) };
            for id in thread_ids(process, me) {
                if calmed.contains(&id) {
                    continue;
                }
                // SAFETY: a thread of ours, opened to name it and set its priority; closed below.
                let thread = unsafe { OpenThread(THREAD_QUERY_INFORMATION | THREAD_SET_INFORMATION, 0, id) };
                if thread.is_null() {
                    continue;
                }
                if name_of(thread).starts_with("D3D Background Thread") {
                    // SAFETY: an open thread handle with the right to set information.
                    let set = unsafe { SetThreadPriority(thread, THREAD_PRIORITY_IDLE) };
                    // And all on the last logical processor: at idle priority they
                    // still ran on every processor the title left free for an
                    // instant (run 110: each in 1,200-1,400 of 1,705 samples), and a
                    // compiler beside a title thread on the same core slows it.
                    let last = std::thread::available_parallelism().map_or(1, |n| n.get()).min(64) - 1;
                    // SAFETY: as above.
                    unsafe { SetThreadAffinityMask(thread, 1usize << last) };
                    eprintln!("victory: {id} (a D3D background thread) put at idle priority: {}", set != 0);
                    calmed.insert(id);
                }
                // SAFETY: we opened it.
                unsafe { CloseHandle(thread) };
            }
        }
    });
}

fn thread_ids(process: u32, me: u32) -> Vec<u32> {
    let mut ids = Vec::new();
    // SAFETY: a snapshot of the system's threads, closed below.
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
    if snapshot == INVALID_HANDLE_VALUE {
        return ids;
    }
    // SAFETY: zeroed is a valid THREADENTRY32; its size is set before use.
    let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<THREADENTRY32>() as u32;
    // SAFETY: `snapshot` is open; `entry` is sized.
    let mut more = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while more {
        if entry.th32OwnerProcessID == process && entry.th32ThreadID != me {
            ids.push(entry.th32ThreadID);
        }
        // SAFETY: as above.
        more = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    // SAFETY: we opened it.
    unsafe { CloseHandle(snapshot) };
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn another_threads_stack_unwinds_to_its_start() {
        let (tx, rx) = std::sync::mpsc::channel();
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let parked = std::thread::spawn(move || {
            // SAFETY: plain query.
            tx.send(unsafe { GetCurrentThreadId() }).unwrap();
            stop_rx.recv().ok();
        });
        let id = rx.recv().unwrap();
        // SAFETY: a thread of this process.
        let thread = unsafe {
            OpenThread(
                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                0,
                id,
            )
        };
        assert!(!thread.is_null());
        let (frames, _) = frames_of(thread);
        // Waiting in the kernel, several frames down from the thread start.
        assert!(frames.len() >= 3, "{frames:?}");
        assert!(start_of(thread) != 0);
        // SAFETY: opened above.
        unsafe { CloseHandle(thread) };
        dump_stacks();
        stop_tx.send(()).unwrap();
        parked.join().unwrap();
    }
}
