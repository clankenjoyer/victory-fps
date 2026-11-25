//! Lazy copy-back of what shaders wrote to host copies in the card's memory
//! (`VICTORY_LAZY_BACK=1`; off by default; needs the relay,
//! `VICTORY_RELAY=1` or `2`, and card buffers).
//!
//! Without it the thread that lands flights fetches every 64 KiB piece the
//! compare on the card flagged, waits for that fetch behind everything on
//! the queue (most of its ~270 ms a frame), and merges the pieces into
//! title memory before the title is told anything. Most of what shaders
//! write is read only by later shaders, from the same host copy, and
//! never by the title's CPU.
//!
//! With the switch, as console emulators do it (pages the GPU wrote are
//! protected; the CPU's first touch downloads them):
//!
//! - at landing the flagged pieces are copied down by a copy queue of ours
//!   (`Host::lazy_card_back`), which runs beside the title's work and is
//!   not waited for, and each piece is noted here as "GPU-ahead": the host
//!   copy is newer than title memory. The 64 KiB blocks of title memory
//!   that hold such pieces are made inaccessible (`PAGE_NOACCESS`,
//!   `crate::watch::lazy_hold`) before the title is told the flight is done;
//! - the first touch of such a block, by a title thread (a fault, taken by
//!   the guard handler, `crate::watch`) or by code of ours that reads or
//!   writes title memory (`crate::host::read_into`, `copy_mapped`,
//!   `crate::watch::writing`, a refresh of any host copy, a merge of other
//!   copies, a remap of the view), waits for that copy queue if it has not
//!   run yet, gives the block its access back and merges the fetched bytes
//!   as the relay did (`Shadow::lazy_merge`: the card's byte where it
//!   differs from the copy's CPU side, else the title's);
//! - fetches still held after `VICTORY_LAZY_BACK_AGE_MS` (4000) or past
//!   `VICTORY_LAZY_BACK_MB` (256) of fetched bytes are merged by the relay
//!   thread at its next landing (the send notes their merges need are kept
//!   until then).
//!
//! Copies in the PC's memory are left on the old path (compared and merged
//! at landing). A refresh of the same copy over a GPU-ahead block merges it
//! first (its CPU side takes the title's bytes and sends them up; without
//! the merge they would undo the card's writes); a refresh of another copy
//! over it merges it first too. Both are counted.
//!
//! What it assumes beyond the relay: a block given back is readable by
//! other threads a moment before its merge is in (the fault's thread
//! merges right after); a system call that reads title memory (a file
//! write from a buffer shaders wrote) fails on a held block unless its
//! caller tells us (`kernel_will_write` does it for kernel writes); title
//! memory's protection outside ours is read-write (or read-only by the
//! guards).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::{
    ID3D12CommandAllocator, ID3D12CommandQueue, ID3D12Fence, ID3D12GraphicsCommandList, ID3D12Resource,
};

use crate::shadow::Shadow;

/// The unit protected and fetched: the guards' block, 64 KiB.
pub(crate) const BLOCK: usize = crate::watch::PAGE;

/// Whether copy-back is lazy (see the module).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if std::env::var_os("VICTORY_LAZY_BACK").is_none_or(|v| v != "1") {
            return false;
        }
        if !crate::host::relay() || !crate::shadow::cards() {
            tracing::warn!(
                "VICTORY_LAZY_BACK=1 needs the relay (VICTORY_RELAY=1 or 2) and card buffers: copy-back stays as it is"
            );
            return false;
        }
        tracing::info!(
            budget_mb = budget() >> 20,
            age_ms = age_ms(),
            "what shaders write to copies in the card's memory goes back to title memory when the title touches it (VICTORY_LAZY_BACK=1)"
        );
        true
    })
}

/// `VICTORY_LAZY_BACK_MB` (256): fetched bytes held before the oldest
/// fetch is merged unasked.
fn budget() -> u64 {
    static MB: OnceLock<u64> = OnceLock::new();
    *MB.get_or_init(|| std::env::var("VICTORY_LAZY_BACK_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(256)) << 20
}

/// `VICTORY_LAZY_BACK_AGE_MS` (4000): how long a fetch is held before it is
/// merged unasked.
fn age_ms() -> u64 {
    static MS: OnceLock<u64> = OnceLock::new();
    *MS.get_or_init(|| std::env::var("VICTORY_LAZY_BACK_AGE_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4000))
}

/// The copy queue lazy fetches go to, and its fence.
pub(crate) struct Queue {
    pub(crate) queue: ID3D12CommandQueue,
    pub(crate) fence: ID3D12Fence,
    /// The last value signalled on `fence` (only the relay thread signals).
    pub(crate) sent: AtomicU64,
}

// SAFETY: D3D12 queues and fences are free-threaded.
unsafe impl Send for Queue {}
// SAFETY: as above.
unsafe impl Sync for Queue {}

static QUEUE: OnceLock<Option<Queue>> = OnceLock::new();

/// The copy queue, made with `make` the first time.
pub(crate) fn queue(make: impl FnOnce() -> Option<Queue>) -> Option<&'static Queue> {
    QUEUE.get_or_init(make).as_ref()
}

/// What one fetch uses: the memory the pieces come down to and the list
/// that copies them.
pub(crate) struct Room {
    pub(crate) buffer: ID3D12Resource,
    pub(crate) data: usize,
    pub(crate) size: u64,
    pub(crate) allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
}

// SAFETY: D3D12 objects; the list is recorded by one thread at a time
// (the relay thread), and the mapping is read after the copy has run.
unsafe impl Send for Room {}
// SAFETY: as above.
unsafe impl Sync for Room {}

/// Rooms of fetches no piece needs any more, with the fence value of
/// their last use.
static POOL: Mutex<Vec<(Room, u64)>> = Mutex::new(Vec::new());

/// A room whose last use has run and that holds `bytes`, if one is kept.
pub(crate) fn take_room(bytes: u64) -> Option<Room> {
    let completed = QUEUE.get()?.as_ref().map(|q| unsafe { q.fence.GetCompletedValue() })?;
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let at = pool.iter().position(|(room, value)| *value <= completed && room.size >= bytes)?;
    Some(pool.swap_remove(at).0)
}

/// One fetch on the copy queue.
struct Fetch {
    room: Option<Room>,
    /// The copy queue's fence value behind it.
    value: u64,
    /// The last send of title bytes to the card's copies that had run when
    /// the copy was sent (`crate::host`'s order of the queue): later sends
    /// may not have reached what it read.
    after: u64,
}

impl Drop for Fetch {
    fn drop(&mut self) {
        if let Some(room) = self.room.take() {
            LIVE_BYTES.fetch_sub(room.size, Relaxed);
            let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
            if pool.len() < 8 {
                pool.push((room, self.value));
            }
        }
    }
}

impl Fetch {
    /// The fetched bytes at `at`, `n` of them.
    fn bytes(&self, at: usize, n: usize) -> &[u8] {
        let room = self.room.as_ref().expect("a live fetch has its room");
        // SAFETY: the mapping of the fetch's buffer, which holds the pieces
        // at their offsets (`landed`); the copy has run (`wait`).
        unsafe { std::slice::from_raw_parts((room.data + at) as *const u8, n) }
    }
}

/// A piece of a copy in the card's memory that is newer than title memory.
struct Entry {
    shadow: Arc<Shadow>,
    /// Offsets in the copy.
    lo: usize,
    hi: usize,
    /// Where the bytes were fetched to; `None` for a copy in the PC's
    /// memory copied back on demand (`landed_direct`): read from the copy.
    fetch: Option<Arc<Fetch>>,
    /// Where its bytes are in the fetch.
    at: usize,
}

struct State {
    /// By title address of the piece and the copy's number.
    entries: BTreeMap<(usize, u64), Entry>,
    /// Live fetches by fence value: their `after`, how many entries hold
    /// them, when they were sent.
    fetches: BTreeMap<u64, (u64, usize, std::time::Instant)>,
}

static STATE: Mutex<State> = Mutex::new(State {
    entries: BTreeMap::new(),
    fetches: BTreeMap::new(),
});

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The blocks whose pieces a merge has taken and not yet put into title
/// memory. A merge waits for the copy queue and writes title memory (the
/// relay: 38% of its time, run 105) with the state's lock given up, so a
/// refresh asking which blocks are its own ([`own_clean`]) does not stand
/// behind it; a second merge waits ([`MERGE_DONE`]) only if it wants a block
/// of these (one lock for all merges still held the submitting thread up
/// for 11% of its time behind the relay's merges of other blocks, run
/// 111). Locked after the state's lock, never before it.
static MERGING: Mutex<BTreeSet<usize>> = Mutex::new(BTreeSet::new());

/// Told when a merge is done and its blocks leave [`MERGING`].
static MERGE_DONE: std::sync::Condvar = std::sync::Condvar::new();

fn merging() -> std::sync::MutexGuard<'static, BTreeSet<usize>> {
    MERGING.lock().unwrap_or_else(|e| e.into_inner())
}

/// The state's lock as a merger holds it: given up while it merges.
type Locked = Option<std::sync::MutexGuard<'static, State>>;

/// The blocks made inaccessible here. Changed only under [`STATE`]'s lock
/// and, inside it, the guards' (`crate::watch`): the order is `STATE`,
/// guards, `HELD`. Read by the fault handler and by the guards.
pub(crate) static HELD: Mutex<BTreeSet<usize>> = Mutex::new(BTreeSet::new());
/// `HELD`'s length, for the callers that only ask whether anything is held.
pub(crate) static HELD_COUNT: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn held() -> std::sync::MutexGuard<'static, BTreeSet<usize>> {
    HELD.lock().unwrap_or_else(|e| e.into_inner())
}

/// Bytes of rooms live fetches hold.
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Why a block was fetched.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Why {
    /// A title thread read it (a fault).
    TitleRead = 0,
    /// A title thread wrote it (a fault): fetched before the write goes on.
    TitleWrite = 1,
    /// Code of ours read or wrote it (a kernel read or write of title
    /// memory, a merge of a copy in the PC's memory).
    Own = 2,
    /// A refresh of a host copy over it (same copy or another: counted
    /// apart per piece).
    Refresh = 3,
    /// Held too long or past the budget.
    Retire = 4,
    /// Its block could not be protected.
    Eager = 5,
    /// The runtime unmaps or maps the view over it.
    Remap = 6,
}

/// Since the last frame line: fetches made by why ([`Why`]: 0..7); pieces
/// marked GPU-ahead and their bytes; pieces marked over an older mark;
/// bytes merged and of those written to title memory; microseconds in
/// fetches and of them waiting for the copy queue; pieces fetched by a
/// refresh of their own copy and of another copy; bytes sent down by the
/// copy queue; copy-queue fetches.
static STATS: [AtomicU64; 20] = [const { AtomicU64::new(0) }; 20];
/// Pieces not merged: their memory was freed and made anew while held.
const LOST: usize = 17;
/// Blocks of view memory merged at once instead of held.
const ALIASED: usize = 18;
/// Held blocks a refresh of their own copy left out: the title had not
/// written them (`own_clean`).
const OWN_CLEAN: usize = 19;
const MARKED: usize = 7;
const MARKED_BYTES: usize = 8;
const REPLACED: usize = 9;
const MERGED: usize = 10;
const WRITTEN: usize = 11;
const FETCH_US: usize = 12;
const WAIT_US: usize = 13;
const SAME_COPY: usize = 14;
const OTHER_COPY: usize = 15;
const DOWN_BYTES: usize = 16;

fn count(i: usize, n: u64) {
    STATS[i].fetch_add(n, Relaxed);
}

/// Waits until the copy queue has run to `value`.
fn wait(value: u64) {
    let Some(Some(q)) = QUEUE.get() else {
        return;
    };
    // SAFETY: a plain query; with no event the call returns once the
    // fence has reached the value.
    unsafe {
        if q.fence.GetCompletedValue() >= value {
            return;
        }
        let waiting = std::time::Instant::now();
        if q.fence.SetEventOnCompletion(value, HANDLE::default()).is_err() {
            while q.fence.GetCompletedValue() < value {
                std::thread::yield_now();
            }
        }
        count(WAIT_US, waiting.elapsed().as_micros() as u64);
    }
}

/// The relay landed a flight: `pieces` (copy, offsets in it, offset in
/// the fetch) were copied down into `room` by the copy queue, behind fence
/// value `value`, after send `after` had run. They are GPU-ahead from now
/// on and their blocks held; what cannot be held is merged at once. Then
/// fetches held too long or past the budget are merged.
pub(crate) fn landed(pieces: &[(Arc<Shadow>, usize, usize, u64)], room: Room, value: u64, after: u64) {
    LIVE_BYTES.fetch_add(room.size, Relaxed);
    let fetch = Arc::new(Fetch {
        room: Some(room),
        value,
        after,
    });
    let mut locked: Locked = Some(state());
    let st = locked.as_mut().expect("locked");
    let mut blocks = BTreeSet::new();
    for (shadow, lo, hi, at) in pieces {
        let (lo, hi) = (*lo, *hi);
        if lo >= hi {
            continue;
        }
        let address = shadow.span().0 as usize + lo;
        let entry = Entry {
            shadow: shadow.clone(),
            lo,
            hi,
            fetch: Some(fetch.clone()),
            at: *at as usize,
        };
        if let Some(old) = st.entries.insert((address, shadow.id()), entry) {
            count(REPLACED, 1);
            forget(st, &old);
        }
        st.fetches.entry(value).or_insert((after, 0, std::time::Instant::now())).1 += 1;
        blocks.extend(address / BLOCK..(address + (hi - lo)).div_ceil(BLOCK));
        count(MARKED, 1);
        count(MARKED_BYTES, (hi - lo) as u64);
    }
    drop(fetch);
    // A block of view memory (the title's physical pages) may be shown at
    // other addresses too, where a read finds no guard and sees the old
    // bytes: merged at once, never held. (Two runs with lazy copy-back
    // crashed in title code reading a pointer the card had rewritten.)
    let (blocks, aliased): (Vec<usize>, Vec<usize>) = blocks
        .into_iter()
        .partition(|b| !shown_elsewhere(b * BLOCK));
    if !aliased.is_empty() {
        count(ALIASED, aliased.len() as u64);
        settle_locked(&mut locked, aliased, Why::Eager, None);
    }
    let new: Vec<usize> = {
        let held = held();
        blocks.into_iter().filter(|b| !held.contains(b)).collect()
    };
    let failed = crate::watch::lazy_hold(&new);
    if !failed.is_empty() {
        settle_locked(&mut locked, failed, Why::Eager, None);
    }
    retire(&mut locked);
}

/// Copy-back on demand (`crate::shadow::on_demand`): the flight that wrote
/// `pieces` (offsets in `shadow`, a copy in the PC's memory, one block of
/// title memory each) has run. They are GPU-ahead from now on, read from
/// the copy when merged (no fetch), and their blocks held; with `hold`
/// false, or where a block cannot be held, merged now.
pub(crate) fn landed_direct(shadow: &Arc<Shadow>, pieces: &[(usize, usize)], hold: bool) {
    if pieces.is_empty() {
        return;
    }
    let mut locked: Locked = Some(state());
    let st = locked.as_mut().expect("locked");
    let mut blocks = BTreeSet::new();
    let start = shadow.span().0 as usize;
    for &(lo, hi) in pieces {
        if lo >= hi {
            continue;
        }
        let address = start + lo;
        let entry = Entry {
            shadow: shadow.clone(),
            lo,
            hi,
            fetch: None,
            at: 0,
        };
        if let Some(old) = st.entries.insert((address, shadow.id()), entry) {
            count(REPLACED, 1);
            forget(st, &old);
        }
        blocks.extend(address / BLOCK..(address + (hi - lo)).div_ceil(BLOCK));
        count(MARKED, 1);
        count(MARKED_BYTES, (hi - lo) as u64);
    }
    if !hold {
        settle_locked(&mut locked, blocks.into_iter().collect(), Why::Eager, None);
        return;
    }
    let new: Vec<usize> = {
        let held = held();
        blocks.into_iter().filter(|b| !held.contains(b)).collect()
    };
    let failed = crate::watch::lazy_hold(&new);
    if !failed.is_empty() {
        settle_locked(&mut locked, failed, Why::Eager, None);
    }
}

/// Blocks of title memory in ranges copied back on demand that the
/// flights just landed did not write (no write marks there,
/// `crate::shadow::Narrow`): nothing to merge or hold. Called after their
/// declarations are done with; each stops flying and gives back its
/// copy-on-write bytes ([`crate::watch::fault::landed_block`]), as a block
/// merged does, unless an entry still held (of an earlier flight) or a
/// merge under way covers it: that merge needs the bytes and lands it.
pub(crate) fn unwritten(blocks: &[usize]) {
    if blocks.is_empty() {
        return;
    }
    let free: Vec<usize> = {
        let st = state();
        let merging = merging();
        blocks
            .iter()
            .copied()
            .filter(|&b| {
                !merging.contains(&b)
                    && !st.entries.range((b * BLOCK, 0)..((b + 1) * BLOCK, 0)).any(|(_, e)| e.fetch.is_none())
            })
            .collect()
    };
    for b in free {
        crate::watch::fault::landed_block(b);
    }
}

/// An entry leaves: its fetch's count.
fn forget(st: &mut State, entry: &Entry) {
    let Some(value) = entry.fetch.as_ref().map(|f| f.value) else {
        return;
    };
    if let Some(f) = st.fetches.get_mut(&value) {
        f.1 = f.1.saturating_sub(1);
        if f.1 == 0 {
            st.fetches.remove(&value);
        }
    }
}

/// Merges the oldest fetches while they are older than [`age_ms`] or the
/// rooms of live fetches hold more than [`budget`].
fn retire(locked: &mut Locked) {
    loop {
        let st = locked.as_mut().expect("locked");
        let Some((&value, &(_, _, made))) = st.fetches.first_key_value() else {
            return;
        };
        let old = made.elapsed().as_millis() as u64 >= age_ms();
        if !old && LIVE_BYTES.load(Relaxed) <= budget() {
            return;
        }
        let blocks: BTreeSet<usize> = st
            .entries
            .iter()
            .filter(|(_, e)| e.fetch.as_ref().is_some_and(|f| f.value == value))
            .flat_map(|(&(address, _), e)| address / BLOCK..(address + (e.hi - e.lo)).div_ceil(BLOCK))
            .collect();
        if blocks.is_empty() {
            st.fetches.remove(&value);
            continue;
        }
        settle_locked(locked, blocks.into_iter().collect(), Why::Retire, None);
    }
}

/// With the relay: the number to tell `crate::shadow::relay_fetched`
/// instead of `number`, so that the notes of sends a held fetch's merge
/// still needs are kept.
pub(crate) fn floor(number: u64) -> u64 {
    if !on() {
        return number;
    }
    state().fetches.first_key_value().map_or(number, |(_, f)| f.0.min(number))
}

/// Title memory `[from, to)` is about to be read or written by code of
/// ours, or by the title: the GPU-ahead pieces in its blocks are merged
/// into it first, and the blocks given back.
pub(crate) fn settle(from: usize, to: usize, why: Why) {
    settle_for(from, to, why, None);
}

/// [`settle`] for a refresh of the host copy numbered `copy`.
pub(crate) fn settle_for(from: usize, to: usize, why: Why, copy: Option<u64>) {
    if HELD_COUNT.load(Relaxed) == 0 || from >= to {
        return;
    }
    let (first, last) = (from / BLOCK, to.div_ceil(BLOCK));
    let blocks: Vec<usize> = held().range(first..last).copied().collect();
    if blocks.is_empty() {
        return;
    }
    let mut locked: Locked = Some(state());
    settle_locked(&mut locked, blocks, why, copy);
}

/// With fault tracking: the held blocks of `[from, to)` (in order) whose
/// pieces all belong to the host copy numbered `copy` and that the title
/// had not written when they were held (`crate::watch::fault::cpu_ahead`;
/// inaccessible since, it cannot have written them after). A refresh of
/// that copy leaves them out: its card side is newer, and title memory has
/// nothing for it there. Fetching them (to merge, then copy the merge back
/// up) was 52 pieces a frame of "refreshes of the refreshed copy".
pub(crate) fn own_clean(from: usize, to: usize, copy: u64) -> Vec<usize> {
    if HELD_COUNT.load(Relaxed) == 0 || from >= to || !crate::watch::fault::on() {
        return Vec::new();
    }
    let (first, last) = (from / BLOCK, to.div_ceil(BLOCK));
    let st = state();
    let held = held();
    let mut own = Vec::new();
    for block in held.range(first..last).copied() {
        if crate::watch::fault::cpu_ahead(block) {
            continue;
        }
        // Entries are keyed by their first address: one starting before
        // the block may reach into it.
        let lo = (block * BLOCK).saturating_sub(64 << 20);
        let mut any = false;
        let mine = st
            .entries
            .range((lo, 0)..((block + 1) * BLOCK, 0))
            .filter(|&(&(address, _), e)| address + (e.hi - e.lo) > block * BLOCK)
            .all(|(&(_, id), _)| {
                any = true;
                id == copy
            });
        if mine && any {
            own.push(block);
        }
    }
    count(OWN_CLEAN, own.len() as u64);
    own
}

thread_local! {
    /// The blocks this thread has in `MERGING`.
    static MINE: std::cell::RefCell<BTreeSet<usize>> = const { std::cell::RefCell::new(BTreeSet::new()) };
}

/// [`settle`] of `blocks`, under the state's lock: their entries (and
/// those of any block an entry reaches into) are taken, the copy queue
/// waited for, the blocks given back, the pieces merged.
fn settle_locked(locked: &mut Locked, blocks: Vec<usize>, why: Why, copy: Option<u64>) {
    let started = std::time::Instant::now();
    let (taken, done) = loop {
        let st = locked.as_mut().expect("locked");
        // The blocks and every block an entry of theirs reaches into.
        let mut todo: BTreeSet<usize> = blocks.iter().copied().collect();
        let mut done: BTreeSet<usize> = BTreeSet::new();
        let mut keys: Vec<(usize, u64)> = Vec::new();
        while let Some(block) = todo.pop_first() {
            if !done.insert(block) {
                continue;
            }
            for (key, entry) in st.entries.range((block * BLOCK, 0)..((block + 1) * BLOCK, 0)) {
                keys.push(*key);
                let (address, n) = (key.0, entry.hi - entry.lo);
                todo.extend((address / BLOCK..(address + n).div_ceil(BLOCK)).filter(|b| !done.contains(b)));
            }
        }
        // Blocks this thread is merging already (a merge that reads title
        // memory comes back here for the block it holds: run 124 froze
        // waiting for itself) are its own: nothing more to do for them.
        MINE.with_borrow(|mine| done.retain(|b| !mine.contains(b)));
        let mut flying = merging();
        if done.iter().any(|b| flying.contains(b)) {
            // Another merge has some of them: when it is done, looked at
            // again (the state's lock given up meanwhile).
            *locked = None;
            drop(MERGE_DONE.wait(flying).unwrap_or_else(|e| e.into_inner()));
            *locked = Some(state());
            continue;
        }
        flying.extend(done.iter().copied());
        drop(flying);
        MINE.with_borrow_mut(|mine| mine.extend(done.iter().copied()));
        let taken: Vec<Entry> = keys.into_iter().filter_map(|key| st.entries.remove(&key)).collect();
        break (taken, done);
    };
    // The state's lock is given up for the wait and the merge (the blocks
    // are in `MERGING`); the fetches keep their counts until the merge is
    // done, so [`floor`] keeps the notes it needs.
    *locked = None;
    // The copy queue runs in order: the newest fetch's value covers all.
    if let Some(value) = taken.iter().filter_map(|e| e.fetch.as_ref().map(|f| f.value)).max() {
        wait(value);
    }
    let blocks: Vec<usize> = done.into_iter().collect();
    // Memory freed and made anew under us (its protection is not ours any
    // more): what shaders wrote to the old memory is not put there.
    let lost = crate::watch::lazy_release(&blocks);
    let mut title = Vec::new();
    let (mut merged, mut written) = (0u64, 0u64);
    for entry in &taken {
        let n = entry.hi - entry.lo;
        let address = entry.shadow.span().0 as usize + entry.lo;
        if !lost.is_empty() && (address / BLOCK..(address + n).div_ceil(BLOCK)).any(|b| lost.contains(&b)) {
            count(LOST, 1);
            continue;
        }
        merged += n as u64;
        written += match &entry.fetch {
            Some(fetch) => entry.shadow.lazy_merge(entry.lo, fetch.bytes(entry.at, n), fetch.after, &mut title),
            None => {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::BackDemandBlocks, 1);
                entry.shadow.demand_merge(entry.lo, entry.hi, &mut title)
            }
        } as u64;
        if why == Why::Refresh {
            count(if copy == Some(entry.shadow.id()) { SAME_COPY } else { OTHER_COPY }, 1);
        }
    }
    // Copies back on demand done: their blocks stop flying (if no other
    // declaration holds them) and give back their copy-on-write bytes.
    for entry in taken.iter().filter(|e| e.fetch.is_none()) {
        let address = entry.shadow.span().0 as usize + entry.lo;
        (address / BLOCK..(address + (entry.hi - entry.lo)).div_ceil(BLOCK)).for_each(crate::watch::fault::landed_block);
    }
    if !taken.is_empty() || !blocks.is_empty() {
        count(why as usize, 1);
    }
    let mut st = state();
    for entry in &taken {
        forget(&mut st, entry);
    }
    {
        let mut flying = merging();
        for block in &blocks {
            flying.remove(block);
        }
    }
    MINE.with_borrow_mut(|mine| blocks.iter().for_each(|b| {
        mine.remove(b);
    }));
    MERGE_DONE.notify_all();
    *locked = Some(st);
    count(MERGED, merged);
    count(WRITTEN, written);
    count(FETCH_US, started.elapsed().as_micros() as u64);
}

/// From the guard handler (`crate::watch`): an access violation at `at`
/// (`kind`: 0 a read, 1 a write, 8 an execution). Whether it was ours and
/// the access can be made again: a held block is fetched and given back; a
/// block another thread gave back since this one faulted is readable
/// again (a write there is the guards' to judge).
pub(crate) fn fault(at: usize, kind: usize) -> bool {
    if HELD_COUNT.load(Relaxed) == 0 && !EVER.load(Relaxed) {
        return false;
    }
    let block = at / BLOCK;
    if held().contains(&block) {
        settle_for(block * BLOCK, (block + 1) * BLOCK, if kind == 1 { Why::TitleWrite } else { Why::TitleRead }, None);
        return true;
    }
    kind == 0 && readable(at)
}

/// Whether anything was ever held (the fault handler's quick answer).
static EVER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Noted by `crate::watch::lazy_hold` when it holds a block.
pub(crate) fn held_some() {
    EVER.store(true, Relaxed);
}

/// Whether a byte at `at` can be read now.
fn readable(at: usize) -> bool {
    let mut byte = [0u8];
    crate::host::read_raw(at as *const u8, &mut byte)
}

/// The runtime is about to unmap (or has mapped) title memory `[at, at +
/// len)`: GPU-ahead pieces there are merged into what it shows before it
/// goes.
pub(crate) fn remapped(at: usize, len: usize) {
    settle(at, at.saturating_add(len), Why::Remap);
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Relaxed) as f64 / frames).collect();
    let (entries, fetches) = {
        let st = state();
        (st.entries.len(), st.fetches.len())
    };
    tracing::info!(
        "lazy copy-back per frame (VICTORY_LAZY_BACK=1): {:.0} pieces, {:.1} MB marked GPU-ahead ({:.0} over an older mark), \
         {:.1} MB sent down by the copy queue; fetched: {:.1} title reads, {:.1} title writes, {:.1} reads/writes of ours, \
         {:.1} refreshes ({:.1} pieces of the refreshed copy, {:.1} of another copy over the same memory), {:.1} retired (age or budget), \
         {:.1} not protectable, {:.1} remaps; {:.1} MB merged, {:.1} MB written to title memory, {:.1} ms in fetches \
         ({:.1} ms waiting for the copy queue), {:.1} pieces dropped (memory freed under them), {:.1} view-memory blocks merged at once, \
         {:.1} blocks a refresh of their own copy left held (the title had not written them); \
         held now: {} pieces of {} fetches in {} blocks, {} MB of fetch memory; {} merges so far found title memory guarded",
        s[MARKED],
        s[MARKED_BYTES] / 1e6,
        s[REPLACED],
        s[DOWN_BYTES] / 1e6,
        s[Why::TitleRead as usize],
        s[Why::TitleWrite as usize],
        s[Why::Own as usize],
        s[Why::Refresh as usize],
        s[SAME_COPY],
        s[OTHER_COPY],
        s[Why::Retire as usize],
        s[Why::Eager as usize],
        s[Why::Remap as usize],
        s[MERGED] / 1e6,
        s[WRITTEN] / 1e6,
        s[FETCH_US] / 1e3,
        s[WAIT_US] / 1e3,
        s[LOST],
        s[ALIASED],
        s[OWN_CLEAN],
        entries,
        fetches,
        HELD_COUNT.load(Relaxed),
        LIVE_BYTES.load(Relaxed) >> 20,
        crate::shadow::LAZY_GUARDED.load(Relaxed),
    );
}

/// The copy queue took `bytes` of pieces down.
pub(crate) fn sent_down(bytes: u64) {
    count(DOWN_BYTES, bytes);
}

/// Whether the title memory at `at` is a view whose physical page the
/// title shows at another address too: a read there finds no guard of
/// ours and sees the old bytes, so such a block is never held. A view
/// shown at this one address is held like any other memory
/// (only with `VICTORY_LAZY_VIEWS=1`; off by default: runs 115 and 117 with it
/// on froze and crashed in title code on a stale pointer at image+0x50ebb70,
/// the site of the first lazy crashes, so some view that is known at one
/// address is still read elsewhere; without it 72 MB a frame are merged
/// at landing in the world, run 96), and when another
/// view of its page is mapped later the block is merged first
/// (`crate::watch::views_mapped`).
fn shown_elsewhere(at: usize) -> bool {
    static VIEWS: OnceLock<bool> = OnceLock::new();
    if crate::inplace::cover(at, BLOCK) == crate::inplace::Cover::None {
        return false;
    }
    if !*VIEWS.get_or_init(|| std::env::var_os("VICTORY_LAZY_VIEWS").is_some_and(|v| v == "1")) {
        return true;
    }
    // The views are known by 64 KiB page.
    let shown = crate::inplace::pages_of(at & !0xFFFF);
    // Not a view the runtime knows (then nothing says where else it is
    // shown), or shown at more than one address.
    shown.len() != 1
}

/// Title memory `[from, to)` was overwritten by a write no guard of ours
/// saw (DirectStorage filling physical pages through the runtime's own
/// view, `crate::watch::title_memory_written`): what the card wrote there
/// before is older than the bytes now in title memory, as on the console
/// where the later write wins. The GPU-ahead pieces wholly inside are
/// dropped, not merged (merged, a piece put the card's old bytes over a
/// freshly loaded page: the stale pointers of runs 115 and 117), and the
/// blocks are given back; a piece reaching outside the range is merged as
/// usual.
pub(crate) fn overwritten(from: usize, to: usize) {
    if HELD_COUNT.load(Relaxed) == 0 || from >= to {
        return;
    }
    let blocks: Vec<usize> = held().range(from / BLOCK..to.div_ceil(BLOCK)).copied().collect();
    if blocks.is_empty() {
        return;
    }
    let mut locked: Locked = Some(state());
    let st = locked.as_mut().expect("locked");
    let inside: Vec<(usize, u64)> = st
        .entries
        .range((from, 0)..(to, 0))
        .filter(|&(&(address, _), e)| address + (e.hi - e.lo) <= to)
        .map(|(k, _)| *k)
        .collect();
    let mut flew = Vec::new();
    for key in inside {
        if let Some(entry) = st.entries.remove(&key) {
            forget(st, &entry);
            count(LOST, 1);
            if entry.fetch.is_none() {
                flew.push((key.0, entry.hi - entry.lo));
            }
        }
    }
    settle_locked(&mut locked, blocks, Why::Own, None);
    drop(locked);
    // Copies back on demand dropped: their blocks stop flying.
    for (address, n) in flew {
        (address / BLOCK..(address + n).div_ceil(BLOCK)).for_each(crate::watch::fault::landed_block);
    }
}

/// The runtime is about to write title memory `[at, at + len)` through a
/// view of its own (a load into physical pages): [`overwritten`], before
/// the first byte lands.
pub fn title_memory_overwriting(at: usize, len: usize) {
    if on() {
        overwritten(at, at.saturating_add(len));
    }
}
