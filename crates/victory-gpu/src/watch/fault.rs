//! Writes to private title memory found by faults, not by asking the kernel
//! (`VICTORY_FAULT_TRACK=1`; off by default).
//!
//! Without it every refresh asks `GetWriteWatch` about its range: some
//! 2,200 askings over 8 GB a world frame, ~49 ms on the submitting thread.
//! With it, a 64 KiB block of write-watched title memory that a refresh
//! takes is made read-only (it is "tracked" and "guarded"), and the
//! title's first write to it faults into our handler (`super::install`),
//! which notes the block written ("dirty") and gives the write access back.
//! An asking about tracked blocks reads the dirty bits (flat bitmaps, 64
//! blocks a word), takes them, makes the taken blocks read-only again, and
//! answers the journal as the kernel's answer would: no kernel call.
//!
//! - A block is adopted when the kernel is first asked about it and
//!   answers (it is write-watched memory): made read-only, then asked about
//!   once more, so that a write between the two is not lost. Blocks that
//!   cannot be made read-only stay with the kernel's askings.
//! - Order of a take: the dirty bits are cleared, then the blocks made
//!   read-only, then the caller copies (Xenia's `MakeRangeValid` before the
//!   upload): a write after the protection faults and is seen next time.
//! - Volatile blocks: dirty at [`volatile_after`] takes in a row, they are
//!   left writable and reported written at every take (the console's
//!   command and constant rings would fault at every submission). After
//!   [`RETRY`] more takes they are guarded again once (Ryujinx
//!   `RegionHandle` `VolatileThreshold`).
//! - Writes made for the title by the kernel or by us (`super::writing`,
//!   which `kernel_will_write` is) open the blocks and mark them dirty first.
//! - The lazy copy-back (`crate::lazy`) holds blocks inaccessible. One
//!   rule decides a block's protection, [`wanted`]: held is inaccessible,
//!   guarded is read-only, else read-write. A take does not guard a held
//!   block; a block given back by the lazy copy-back is a written block
//!   ([`released`]); a guarded block may be held (`super::lazy_hold`).
//!   Changes of protection here and in `lazy_hold`/`lazy_release` are made
//!   under the lazy copy-back's `HELD` lock (when it is on), so the two
//!   never undo each other.
//! - A check ([`verify`], every `VICTORY_FAULT_VERIFY_MS`, 250) asks the
//!   kernel about all tracked blocks: one it saw written while we held it
//!   read-only and nobody opened it has been freed and made anew (or
//!   re-protected by the title); it is reported written and no longer
//!   tracked.
//!
//! The fault path ([`fault`]) allocates nothing and takes no lock of ours
//! but the lazy copy-back's `HELD` (only when that is on; no thread holds
//! it while it touches title memory). The bitmaps are atomics in regions
//! that are made by the asking side and never freed.
//!
//! Model: Xenia `SharedMemory`/`MemoryInvalidationCallback` (BSD-3), Ryujinx
//! `RegionHandle`/`MultiRegionHandle` (MIT).

use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64};
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE};

/// The unit tracked: the journal's page, the guards' and the lazy
/// copy-back's block (64 KiB).
pub(crate) const BLOCK: usize = super::PAGE;

/// Blocks per region of the table (256 MiB of address space).
const REGION_BLOCKS: usize = 4096;
const WORDS: usize = REGION_BLOCKS / 64;
/// Regions in 47 bits of address space.
const TOP_LEN: usize = (1usize << 47) / (REGION_BLOCKS * BLOCK);

/// Takes a block stays volatile before it is guarded again once.
const RETRY: u8 = 64;

/// Whether fault tracking is on (see the module).
pub(crate) fn on() -> bool {
    if cfg!(test) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_FAULT_TRACK").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                volatile_after = volatile_after(),
                verify_ms = verify_ms(),
                "writes to write-watched title memory are found by faults on read-only blocks, not by kernel askings (VICTORY_FAULT_TRACK=1)"
            );
        }
        on
    })
}

/// `VICTORY_FAULT_VOLATILE` (5): takes in a row that find a block written
/// before it is left writable.
fn volatile_after() -> u8 {
    static N: OnceLock<u8> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_FAULT_VOLATILE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(5u8).clamp(1, 200)
    })
}

/// `VICTORY_FAULT_VERIFY_MS` (250): how often [`verify`] runs at most.
fn verify_ms() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_FAULT_VERIFY_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(250))
}

/// The protection a block of title memory is to have: inaccessible while
/// the lazy copy-back holds it, read-only while guarded (here, or by the
/// view guards), else read-write. The one rule both sides follow.
pub(crate) fn wanted(held: bool, guarded: bool) -> u32 {
    if held {
        PAGE_NOACCESS
    } else if guarded {
        PAGE_READONLY
    } else {
        PAGE_READWRITE
    }
}

struct Region {
    /// Followed here (not by the kernel's askings).
    tracked: [AtomicU64; WORDS],
    /// Read-only by our hand now.
    guarded: [AtomicU64; WORDS],
    /// Written since the last take.
    dirty: [AtomicU64; WORDS],
    /// Left writable and reported written at every take.
    volatile: [AtomicU64; WORDS],
    /// Writable by our hand at some time since the last [`verify`].
    opened: [AtomicU64; WORDS],
    /// Could not be made read-only: left to the kernel's askings until
    /// what the memory shows changes ([`forget`]).
    refused: [AtomicU64; WORDS],
    /// Not committed when adopted (view memory between mappings): covered,
    /// nothing to copy, looked at again by each take.
    absent: [AtomicU64; WORDS],
    /// View memory (not write-watched): the check ([`verify`]) leaves it
    /// to the runtime's remap notices.
    view: [AtomicU64; WORDS],
    /// The lazy copy-back holds the block and the title had written it
    /// since the last take when it was held ([`note_held`]).
    held_dirty: [AtomicU64; WORDS],
    /// Shaders of a submission not landed yet may be writing a host copy
    /// of the block, declared for copy-back on demand (`crate::shadow`):
    /// the title's first write takes a copy of the block first ([`cow`]).
    flying: [AtomicU64; WORDS],
    /// Made read-only by our hand at some time (never cleared): a write
    /// fault on such a block is ours to answer whatever the other bits say
    /// ([`fault`]), so that no race between a take, a fault, an opening
    /// and a drop leaves the title a read-only block nobody claims.
    ours: [AtomicU64; WORDS],
    /// The copy-on-write pool could not take the block: its first write
    /// is not known; the copy-back keeps the title's bytes there.
    cow_lost: [AtomicU64; WORDS],
    /// Declarations on demand not done with, per block.
    fly: [AtomicU16; REGION_BLOCKS],
    /// The pool slot (plus one) holding the block as it was before the
    /// title's first write while flying; 0 none.
    cow: [AtomicU32; REGION_BLOCKS],
    /// The [`EPOCH`] of the last change seen in the word (a take that found
    /// a block written, an adoption, our own write, a block dropped).
    epoch: [AtomicU64; WORDS],
    /// Blocks whose `streak` is not 0.
    hot: [AtomicU64; WORDS],
    /// Takes in a row that found the block written (or, volatile, takes
    /// since it became so).
    streak: [AtomicU8; REGION_BLOCKS],
    /// The frame (plus one) in which a take last asked the system whether
    /// the word's absent blocks are committed now ([`take`]).
    absent_asked: [AtomicU32; WORDS],
}

impl Region {
    fn new() -> Region {
        Region {
            absent_asked: [const { AtomicU32::new(0) }; WORDS],
            tracked: [const { AtomicU64::new(0) }; WORDS],
            guarded: [const { AtomicU64::new(0) }; WORDS],
            dirty: [const { AtomicU64::new(0) }; WORDS],
            volatile: [const { AtomicU64::new(0) }; WORDS],
            opened: [const { AtomicU64::new(0) }; WORDS],
            hot: [const { AtomicU64::new(0) }; WORDS],
            refused: [const { AtomicU64::new(0) }; WORDS],
            absent: [const { AtomicU64::new(0) }; WORDS],
            view: [const { AtomicU64::new(0) }; WORDS],
            held_dirty: [const { AtomicU64::new(0) }; WORDS],
            flying: [const { AtomicU64::new(0) }; WORDS],
            ours: [const { AtomicU64::new(0) }; WORDS],
            cow_lost: [const { AtomicU64::new(0) }; WORDS],
            fly: [const { AtomicU16::new(0) }; REGION_BLOCKS],
            cow: [const { AtomicU32::new(0) }; REGION_BLOCKS],
            epoch: [const { AtomicU64::new(0) }; WORDS],
            streak: [const { AtomicU8::new(0) }; REGION_BLOCKS],
        }
    }
}

/// Region pointers by region number, made on first use by the asking side.
static TOP: OnceLock<Box<[AtomicPtr<Region>]>> = OnceLock::new();
/// The region numbers made, for [`verify`].
static MADE: Mutex<Vec<usize>> = Mutex::new(Vec::new());
/// Blocks tracked now (for the line).
static TRACKED: AtomicU64 = AtomicU64::new(0);
/// Numbers the changes to tracked memory that the journal learns of (see
/// `Region::epoch`); a host copy checked at an epoch is current for a word
/// whose epoch is not newer and that holds no written block.
static EPOCH: AtomicU64 = AtomicU64::new(1);
/// `crate::lazy::on()`, noted by the asking side for the fault path (whose
/// first call would otherwise read the environment and log).
static LAZY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The region of `block`, if one was made (the fault path's lookup).
fn region(block: usize) -> Option<&'static Region> {
    let top = TOP.get()?;
    let p = top.get(block / REGION_BLOCKS)?.load(Acquire);
    // SAFETY: regions are leaked boxes, never freed.
    (!p.is_null()).then(|| unsafe { &*p })
}

/// The region of `block`, made if it is not yet; `None` past 47 bits.
fn region_made(block: usize) -> Option<&'static Region> {
    let top = TOP.get_or_init(|| (0..TOP_LEN).map(|_| AtomicPtr::new(std::ptr::null_mut())).collect());
    let slot = top.get(block / REGION_BLOCKS)?;
    let p = slot.load(Acquire);
    if !p.is_null() {
        // SAFETY: as in `region`.
        return Some(unsafe { &*p });
    }
    let new = Box::into_raw(Box::new(Region::new()));
    match slot.compare_exchange(std::ptr::null_mut(), new, AcqRel, Acquire) {
        Ok(_) => {
            MADE.lock().unwrap_or_else(|e| e.into_inner()).push(block / REGION_BLOCKS);
            // SAFETY: just leaked.
            Some(unsafe { &*new })
        }
        Err(other) => {
            // SAFETY: ours, never shared.
            drop(unsafe { Box::from_raw(new) });
            // SAFETY: as in `region`.
            Some(unsafe { &*other })
        }
    }
}

/// Calls `f(region, word, mask, first block of the word)` for the words
/// covering blocks `[first, last)`; `None` for words of regions not made.
fn words(first: usize, last: usize, make: bool, mut f: impl FnMut(Option<&'static Region>, usize, u64, usize)) {
    let mut at = first;
    while at < last {
        let in_region = at % REGION_BLOCKS;
        let (w, b) = (in_region / 64, in_region % 64);
        let n = (last - at).min(64 - b);
        let mask = if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << b };
        let r = if make { region_made(at) } else { region(at) };
        f(r, w, mask, at - b);
        at += n;
    }
}

/// What changes protection, and what the lazy copy-back holds: the system
/// in the program, a record in the tests.
pub(super) trait Os {
    /// Sets the protection of blocks `[from, to)`; the old protection (of
    /// the first page), or `None` if it failed.
    fn protect(&self, from: usize, to: usize, protection: u32) -> Option<u32>;
    /// Runs `f` with what tells whether a block is held by the lazy
    /// copy-back, its lock held.
    fn with_held(&self, f: &mut dyn FnMut(&dyn Fn(usize) -> bool));
    /// The end (exclusive) of the uncommitted blocks from `block` on
    /// (`block` itself if it is committed).
    fn uncommitted_end(&self, block: usize) -> usize;
    /// What the system says of the memory at `block`'s first byte: the
    /// end address (exclusive) of the pages from there that share its
    /// state, whether they are committed, and their protection. `None` if
    /// it says nothing.
    fn query(&self, block: usize) -> Option<(usize, bool, u32)>;
}

/// The system.
pub(super) struct Windows;

impl Os for Windows {
    fn protect(&self, from: usize, to: usize, protection: u32) -> Option<u32> {
        protect_blocks(from, to, protection)
    }

    fn with_held(&self, f: &mut dyn FnMut(&dyn Fn(usize) -> bool)) {
        if crate::lazy::on() {
            let held = crate::lazy::held();
            f(&|b| held.contains(&b));
        } else {
            f(&|_| false);
        }
    }

    fn uncommitted_end(&self, block: usize) -> usize {
        super::uncommitted_end(block)
    }

    fn query(&self, block: usize) -> Option<(usize, bool, u32)> {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
        (n != 0).then(|| {
            (
                (info.BaseAddress as usize).saturating_add(info.RegionSize),
                info.State == MEM_COMMIT,
                info.Protect,
            )
        })
    }
}

fn protect_blocks(from: usize, to: usize, protection: u32) -> Option<u32> {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    let mut old = 0;
    // SAFETY: title memory we track; only its protection changes.
    (unsafe { VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, protection, &mut old) } != 0).then_some(old)
}

#[cfg(feature = "diagnostics")]
fn count(c: crate::copystat::C, n: u64) {
    if n > 0 {
        crate::copystat::add(c, n);
    }
}

/// Runs of set bits in `bits` (relative to `base`), as block ranges.
fn runs(bits: u64, base: usize, out: &mut Vec<(usize, usize)>) {
    let mut v = bits;
    while v != 0 {
        let lo = v.trailing_zeros() as usize;
        let n = (v >> lo).trailing_ones() as usize;
        let (from, to) = (base + lo, base + lo + n);
        match out.last_mut() {
            Some(last) if last.1 == from => last.1 = to,
            _ => out.push((from, to)),
        }
        v = if lo + n >= 64 { 0 } else { v & !(((1u64 << n) - 1) << lo) };
    }
}

/// Whether blocks `[first, last)` are all covered here: tracked, or noted
/// absent (not committed), so that a take answers for them.
pub(crate) fn all_tracked(first: usize, last: usize) -> bool {
    let mut all = first < last;
    words(first, last, false, |r, w, mask, _| {
        all &= r.is_some_and(|r| (r.tracked[w].load(Acquire) | r.absent[w].load(Acquire)) & mask == mask);
    });
    all
}

/// The current epoch (see [`EPOCH`]).
pub(crate) fn epoch() -> u64 {
    EPOCH.load(Acquire)
}

/// The journal learns of changes to blocks `[first, last)` (our own write,
/// a kernel answer): their words' epochs move on.
pub(crate) fn touched(first: usize, last: usize) {
    if TOP.get().is_none() || first >= last {
        return;
    }
    let e = EPOCH.fetch_add(1, AcqRel) + 1;
    words(first, last, false, |r, w, _, _| {
        if let Some(r) = r {
            r.epoch[w].fetch_max(e, AcqRel);
        }
    });
}

/// Whether the blocks `mask` of global word `word` (blocks `word * 64..`)
/// are all covered, none written since the last take nor volatile, and the
/// word unchanged since epoch `since`: a host copy checked then is current
/// there.
pub(crate) fn word_clean(word: usize, mask: u64, since: u64) -> bool {
    let block = word * 64;
    let Some(r) = region(block) else {
        return false;
    };
    let w = (block % REGION_BLOCKS) / 64;
    (r.tracked[w].load(Acquire) | r.absent[w].load(Acquire)) & mask == mask
        && (r.dirty[w].load(Acquire) | r.volatile[w].load(Acquire)) & mask == 0
        && r.epoch[w].load(Acquire) <= since
}

/// The epoch of global word `word`'s last change (0 if never).
pub(crate) fn word_epoch(word: usize) -> u64 {
    region(word * 64).map_or(u64::MAX, |r| r.epoch[(word * 64 % REGION_BLOCKS) / 64].load(Acquire))
}

/// For `super::lazy_hold`, under the held lock: `block` is made
/// inaccessible. Noted whether the title had written it since the last
/// take (or it is not tracked, so that nobody knows): then its bytes in
/// title memory are newer than the host copy's CPU side in places.
pub(crate) fn note_held(block: usize) {
    if let Some(r) = region(block) {
        let (w, bit) = ((block % REGION_BLOCKS) / 64, 1u64 << (block % 64));
        let dirty = r.dirty[w].load(Acquire) & bit != 0 || r.tracked[w].load(Acquire) & bit == 0;
        if dirty {
            r.held_dirty[w].fetch_or(bit, AcqRel);
        } else {
            r.held_dirty[w].fetch_and(!bit, AcqRel);
        }
    }
}

/// Whether the title may have written held `block` since a refresh of the
/// copies over it took it ([`note_held`]): if not, a refresh of the copy
/// the lazy copy-back holds it for has nothing to take from it (the card's
/// side is newer), and need not fetch it.
pub(crate) fn cpu_ahead(block: usize) -> bool {
    region(block).is_none_or(|r| {
        let (w, bit) = ((block % REGION_BLOCKS) / 64, 1u64 << (block % 64));
        r.tracked[w].load(Acquire) & bit == 0 || r.held_dirty[w].load(Acquire) & bit != 0
    })
}

/// Pushes the blocks of `[first, last)` that could not be made read-only
/// to `out`: written as far as anyone knows, at every asking.
pub(crate) fn refused(first: usize, last: usize, out: &mut Vec<usize>) {
    words(first, last, false, |r, w, mask, base| {
        if let Some(r) = r {
            let mut v = r.refused[w].load(Acquire) & mask;
            while v != 0 {
                out.push(base + v.trailing_zeros() as usize);
                v &= v - 1;
            }
        }
    });
}

/// Block `b` is not committed: dropped, noted absent (covered, nothing to
/// copy) until a take finds it committed.
fn make_absent(b: usize) {
    let Some(r) = region(b) else {
        return;
    };
    let (w, bit) = ((b % REGION_BLOCKS) / 64, 1u64 << (b % 64));
    let view = r.view[w].load(Acquire) & bit;
    drop_block(b);
    r.view[w].fetch_or(view, AcqRel);
    r.absent[w].fetch_or(bit, AcqRel);
}

/// Makes the runs read-only (guarded, as [`wanted`] says) under the held
/// lock, leaving out held blocks. Their guarded and `ours` bits are set
/// before the protection, so a fault sees them.
///
/// With `check` (blocks taken on, whose protection is not known), the
/// system is asked first and only blocks wholly committed and read-write
/// are protected; uncommitted ones are noted absent, others refused, and
/// none of them is touched. (Protecting a run and putting back what the
/// call reported, the first page's protection, over all of it left the
/// rest of the run read-only with nothing tracking it.) Without `check`
/// (blocks written since a take: read-write by our hand) a run is
/// protected in one call, block by block if that fails.
fn guard_runs(os: &dyn Os, list: &[(usize, usize)], check: bool) -> u64 {
    let mut protected = 0u64;
    let mut protect = |at: usize, end: usize| {
        set_bits(at, end, |r, w, m| {
            r.ours[w].fetch_or(m, AcqRel);
            r.guarded[w].fetch_or(m, AcqRel)
        });
        if os.protect(at, end, wanted(false, true)).is_some() {
            protected += (end - at) as u64;
            return;
        }
        // A long run may reach over more than one allocation, which the
        // system refuses in one call: by 64 blocks first, then by block.
        let mut piece = at;
        while piece < end {
            let stop = ((piece / 64 + 1) * 64).min(end);
            if stop - piece > 1 && end - at > 64 && os.protect(piece, stop, wanted(false, true)).is_some() {
                protected += (stop - piece) as u64;
                piece = stop;
                continue;
            }
            for b in piece..stop {
                if os.protect(b, b + 1, wanted(false, true)).is_some() {
                    protected += 1;
                } else if os.uncommitted_end(b) > b {
                    make_absent(b);
                } else {
                    refuse(b);
                }
            }
            piece = stop;
        }
    };
    os.with_held(&mut |held| {
        for &(from, to) in list {
            // Cut around held blocks: they stay inaccessible.
            let mut at = from;
            while at < to {
                if held(at) {
                    at += 1;
                    continue;
                }
                let mut end = at + 1;
                while end < to && !held(end) {
                    end += 1;
                }
                if !check {
                    protect(at, end);
                    at = end;
                    continue;
                }
                // By runs of one state, as the system reports them.
                let mut b = at;
                while b < end {
                    let Some((until, committed, protection)) = os.query(b) else {
                        refuse(b);
                        b += 1;
                        continue;
                    };
                    let whole = (until / BLOCK).min(end);
                    if whole <= b {
                        // The state changes inside the block: not ours.
                        refuse(b);
                        b += 1;
                        continue;
                    }
                    if !committed {
                        (b..whole).for_each(make_absent);
                    } else if protection != PAGE_READWRITE {
                        (b..whole).for_each(refuse);
                    } else {
                        protect(b, whole);
                    }
                    b = whole;
                }
                at = end;
            }
        }
    });
    protected
}

/// Calls `f(region, word, mask)` for the made regions' words of `[from, to)`.
fn set_bits(from: usize, to: usize, mut f: impl FnMut(&Region, usize, u64) -> u64) {
    words(from, to, false, |r, w, mask, _| {
        if let Some(r) = r {
            f(r, w, mask);
        }
    });
}

/// Block `b` could not be made read-only: dropped, and not adopted again.
fn refuse(b: usize) {
    drop_block(b);
    if let Some(r) = region(b) {
        r.refused[(b % REGION_BLOCKS) / 64].fetch_or(1u64 << (b % 64), AcqRel);
    }
}

/// Block `b` is no longer tracked (its protection is left as it is).
fn drop_block(b: usize) {
    if let Some(r) = region(b) {
        let (w, bit) = ((b % REGION_BLOCKS) / 64, 1u64 << (b % 64));
        if r.tracked[w].fetch_and(!bit, AcqRel) & bit != 0 {
            TRACKED.fetch_sub(1, Relaxed);
        }
        r.guarded[w].fetch_and(!bit, AcqRel);
        r.volatile[w].fetch_and(!bit, AcqRel);
        r.dirty[w].fetch_and(!bit, AcqRel);
        r.hot[w].fetch_and(!bit, AcqRel);
        r.absent[w].fetch_and(!bit, AcqRel);
        r.view[w].fetch_and(!bit, AcqRel);
        r.held_dirty[w].fetch_and(!bit, AcqRel);
        r.streak[b % REGION_BLOCKS].store(0, Relaxed);
    }
}

/// Takes the written blocks of `[first, last)`, all covered: pushes them
/// (and the volatile ones, and absent ones now committed) to `out`, and
/// guards again those written that are not volatile. The words where it
/// found any move to a new epoch. Under the journal's lock.
fn absent_every_take() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(test) || std::env::var_os("VICTORY_ABSENT_EVERY_TAKE").is_some_and(|v| v == "1"))
}

pub(super) fn take(first: usize, last: usize, out: &mut Vec<usize>, os: &dyn Os) {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let after = volatile_after();
    let mut guard: Vec<(usize, usize)> = Vec::new();
    let mut revived: Vec<(usize, usize)> = Vec::new();
    let mut changed: Vec<(&'static Region, usize)> = Vec::new();
    #[cfg(feature = "diagnostics")]
    let mut turned = 0u64;
    words(first, last, false, |r, w, whole, base| {
        let Some(r) = r else {
            return;
        };
        // Absent blocks committed since (ordinary commits need no notice
        // from the runtime): one question per run of them.
        // Once a frame for each word, not at every take: a submission
        // takes its ranges, some thirty a frame, and each asked the
        // system about every run of absent blocks again (14 ms a frame of
        // `VirtualQuery` in the world, profile of run fork7). What the
        // runtime commits or maps it tells us at once (`forget`); this
        // asking is for what it does not. `VICTORY_ABSENT_EVERY_TAKE=1`:
        // at every take, as before.
        let absent = r.absent[w].load(Acquire) & whole;
        // (And not every frame: every `VICTORY_ABSENT_EVERY`th, 4 unless set;
        // a run of absent blocks was one system call a frame each, 30 ms a
        // frame of them on the submit thread, profile of run fork19. Memory
        // the title commits without telling the runtime is taken up that
        // many frames late at most, then as written whole. 1: every frame.)
        static EVERY: OnceLock<u32> = OnceLock::new();
        let every = *EVERY.get_or_init(|| {
            if cfg!(test) {
                return 1;
            }
            std::env::var("VICTORY_ABSENT_EVERY").ok().and_then(|v| v.trim().parse::<u32>().ok()).unwrap_or(4).max(1)
        });
        let asked = (crate::frame::current() as u32 + w as u32) / every + 1;
        if absent != 0 && (absent_every_take() || r.absent_asked[w].swap(asked, AcqRel) != asked) {
            let mut list = Vec::new();
            runs(absent, base, &mut list);
            let before = revived.len();
            for (from, to) in list {
                let mut at = from;
                while at < to {
                    let end = os.uncommitted_end(at).clamp(at, to);
                    if end > at {
                        at = end;
                        continue;
                    }
                    // Committed from `at` on: up to the next uncommitted one.
                    // (By the system's regions: one question for all the
                    // blocks a committed region begins, where each block
                    // was asked for by itself.)
                    let mut stop = at + 1;
                    while stop < to {
                        match os.query(stop) {
                            Some((end, true, _)) => stop = end.div_ceil(BLOCK).max(stop + 1).min(to),
                            Some((_, false, _)) => break,
                            None => stop += 1,
                        }
                    }
                    match revived.last_mut() {
                        Some(last) if last.1 == at => last.1 = stop,
                        _ => revived.push((at, stop)),
                    }
                    at = stop;
                }
            }
            if revived.len() > before {
                changed.push((r, w));
            }
        }
        let mask = whole & r.tracked[w].load(Acquire);
        let vol = r.volatile[w].load(Acquire) & mask;
        // Taken before the protection (see the module).
        let d = r.dirty[w].fetch_and(!(mask & !vol), AcqRel) & mask & !vol;
        let hot = r.hot[w].load(Relaxed) & mask;
        if d | vol | hot == 0 {
            return;
        }
        if d | vol != 0 {
            changed.push((r, w));
        }
        let mut again = 0u64;
        let (mut new_hot, mut cold, mut to_vol, mut from_vol) = (0u64, 0u64, 0u64, 0u64);
        let mut v = d | vol | hot;
        while v != 0 {
            let b = v.trailing_zeros() as usize;
            v &= v - 1;
            let bit = 1u64 << b;
            let s = &r.streak[w * 64 + b];
            if d & bit != 0 {
                let n = s.load(Relaxed).saturating_add(1);
                s.store(n, Relaxed);
                new_hot |= bit;
                out.push(base + b);
                if n >= after && r.flying[w].load(Acquire) & bit == 0 {
                    to_vol |= bit;
                    s.store(0, Relaxed);
                } else {
                    again |= bit;
                }
            } else if vol & bit != 0 {
                out.push(base + b);
                let n = s.load(Relaxed).saturating_add(1);
                if n >= RETRY {
                    from_vol |= bit;
                    again |= bit;
                    s.store(0, Relaxed);
                    cold |= bit;
                } else {
                    s.store(n, Relaxed);
                    new_hot |= bit;
                }
            } else {
                s.store(0, Relaxed);
                cold |= bit;
            }
        }
        if to_vol != 0 {
            r.volatile[w].fetch_or(to_vol, AcqRel);
            r.opened[w].fetch_or(to_vol, AcqRel);
            #[cfg(feature = "diagnostics")]
            {
                turned += u64::from(to_vol.count_ones());
            }
        }
        if from_vol != 0 {
            r.volatile[w].fetch_and(!from_vol, AcqRel);
        }
        r.hot[w].fetch_or(new_hot, Relaxed);
        r.hot[w].fetch_and(!(cold & !new_hot), Relaxed);
        runs(again, base, &mut guard);
    });
    // Absent blocks now committed: new memory, written as a whole.
    for &(from, to) in &revived {
        set_bits(from, to, |r, w, m| {
            r.absent[w].fetch_and(!m, AcqRel);
            r.opened[w].fetch_or(m, AcqRel);
            TRACKED.fetch_add(u64::from((m & !r.tracked[w].load(Acquire)).count_ones()), Relaxed);
            r.tracked[w].fetch_or(m, AcqRel)
        });
        out.extend(from..to);
    }
    // Runs that touch (they are cut at every 64 blocks) are one run: one
    // change of protection each, where 14 ms a frame went to those calls.
    guard.sort_unstable();
    let mut joined: Vec<(usize, usize)> = Vec::with_capacity(guard.len());
    for (from, to) in guard {
        match joined.last_mut() {
            Some(last) if last.1 >= from => last.1 = last.1.max(to),
            _ => joined.push((from, to)),
        }
    }
    let guard = joined;
    let mut protected = if guard.is_empty() { 0 } else { guard_runs(os, &guard, false) };
    if !revived.is_empty() {
        protected += guard_runs(os, &revived, true);
    }
    if !changed.is_empty() {
        let e = EPOCH.fetch_add(1, AcqRel) + 1;
        for (r, w) in changed {
            r.epoch[w].fetch_max(e, AcqRel);
        }
    }
    let _ = protected;
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::C;
        count(C::FaultTakes, 1);
        count(C::FaultProtected, protected);
        count(C::FaultVolatile, turned);
        count(C::FaultTakeUs, started.elapsed().as_micros() as u64);
    }
}

/// Blocks `[first, last)` were just answered for by the kernel (they are
/// write-watched), or are view memory (`view`): those not covered yet are
/// made read-only and tracked, or noted absent if not committed. Pushes
/// them to `fresh_out` (a view's are all written as far as anyone knows).
/// Returns how many were made read-only. Under the journal's lock; for
/// write-watched memory the caller asks the kernel once more if any were,
/// for writes made between its asking and now.
pub(super) fn adopt(first: usize, last: usize, view: bool, fresh_out: &mut Vec<usize>, os: &dyn Os) -> usize {
    super::install();
    LAZY.store(crate::lazy::on(), Relaxed);
    let mut new: Vec<(usize, usize)> = Vec::new();
    let mut changed: Vec<(&'static Region, usize)> = Vec::new();
    words(first, last, true, |r, w, mask, base| {
        let Some(r) = r else {
            return;
        };
        let fresh = mask & !r.tracked[w].load(Acquire) & !r.refused[w].load(Acquire) & !r.absent[w].load(Acquire);
        if fresh == 0 {
            return;
        }
        r.tracked[w].fetch_or(fresh, AcqRel);
        if view {
            r.view[w].fetch_or(fresh, AcqRel);
        }
        // Writable until now: a check must not take its writes for a
        // freed block's.
        r.opened[w].fetch_or(fresh, AcqRel);
        TRACKED.fetch_add(u64::from(fresh.count_ones()), Relaxed);
        runs(fresh, base, &mut new);
        changed.push((r, w));
    });
    if new.is_empty() {
        return 0;
    }
    for &(from, to) in &new {
        fresh_out.extend(from..to);
    }
    let protected = guard_runs(os, &new, true);
    let e = EPOCH.fetch_add(1, AcqRel) + 1;
    for (r, w) in changed {
        r.epoch[w].fetch_max(e, AcqRel);
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::C;
        count(C::FaultAdopted, protected);
        count(C::FaultProtected, protected);
    }
    protected as usize
}

/// After [`open`] for a kernel write: the blocks of `[at, at + len)` are
/// tracked no longer and never adopted again (asked of the kernel's write
/// watch, as before tracking). An overlapped read completes after the call
/// that started it, and a block guarded again in between makes that read
/// fail, which the title does not survive (loading-screen crashes at
/// title+0x164a298 in two of four runs with tracking).
pub(super) fn exclude(at: usize, len: usize) {
    if len == 0 || TOP.get().is_none() {
        return;
    }
    // Kept tracked but volatile: left writable and taken as written at
    // every take, until `volatile_after` takes guard it again once (an
    // overlapped read has long completed by then).
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    for b in first..last {
        if let Some(r) = region(b) {
            let (w, bit) = ((b % REGION_BLOCKS) / 64, 1u64 << (b % 64));
            if r.tracked[w].load(Acquire) & bit != 0 {
                r.volatile[w].fetch_or(bit, AcqRel);
                r.dirty[w].fetch_or(bit, AcqRel);
            }
        }
    }
}

/// Before the kernel or we write title memory `[at, at + len)` for the
/// title: its guarded blocks are made writable, and its tracked blocks
/// noted written. Blocks the lazy copy-back holds are its to give back.
pub(super) fn open(at: usize, len: usize, os: &dyn Os) {
    if len == 0 || TOP.get().is_none() {
        return;
    }
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    let mut list: Vec<(usize, usize)> = Vec::new();
    os.with_held(&mut |held| {
        list.clear();
        words(first, last, false, |r, w, mask, base| {
            let Some(r) = r else {
                return;
            };
            let t = r.tracked[w].load(Acquire) & mask;
            if t == 0 {
                return;
            }
            let mut g = r.guarded[w].load(Acquire) & t;
            // Held blocks keep their guarded bit and protection.
            let mut v = g;
            while v != 0 {
                let b = v.trailing_zeros() as usize;
                v &= v - 1;
                if held(base + b) {
                    g &= !(1u64 << b);
                }
            }
            let g = r.guarded[w].fetch_and(!g, AcqRel) & g;
            // A flying block keeps its bytes from before the write.
            let mut v = g & r.flying[w].load(Acquire);
            while v != 0 {
                let b = v.trailing_zeros() as usize;
                v &= v - 1;
                cow(r, base + b);
            }
            runs(g, base, &mut list);
            r.opened[w].fetch_or(t, AcqRel);
        });
        for &(from, to) in &list {
            if os.protect(from, to, wanted(false, false)).is_none() {
                for b in from..to {
                    os.protect(b, b + 1, wanted(false, false));
                }
            }
        }
    });
    // Noted written after the access is back (see `fault`).
    words(first, last, false, |r, w, mask, _| {
        if let Some(r) = r {
            let t = r.tracked[w].load(Acquire) & mask;
            r.dirty[w].fetch_or(t, AcqRel);
        }
    });
    #[cfg(feature = "diagnostics")]
    count(crate::copystat::C::FaultUnprotected, list.iter().map(|&(a, b)| (b - a) as u64).sum());
}

/// For `super::lazy_release`, under the held lock: `block` is given its
/// write access back by the lazy copy-back, about to be written.
pub(crate) fn released(block: usize) {
    if let Some(r) = region(block) {
        let (w, bit) = ((block % REGION_BLOCKS) / 64, 1u64 << (block % 64));
        if r.tracked[w].load(Acquire) & bit != 0 {
            r.guarded[w].fetch_and(!bit, AcqRel);
            r.opened[w].fetch_or(bit, AcqRel);
            r.dirty[w].fetch_or(bit, AcqRel);
        }
        r.held_dirty[w].fetch_and(!bit, AcqRel);
    }
}

/// For `super::lazy_hold`: whether `block` is read-only by our hand.
pub(crate) fn guarded(block: usize) -> bool {
    region(block).is_some_and(|r| r.guarded[(block % REGION_BLOCKS) / 64].load(Acquire) & (1u64 << (block % 64)) != 0)
}

/// What title memory `[at, at + len)` shows changes (mapped, unmapped,
/// freed): nothing there is tracked any more. Guarded blocks are given
/// their write access back (held ones stay the lazy copy-back's).
pub(super) fn forget(at: usize, len: usize, os: &dyn Os) {
    if len == 0 || TOP.get().is_none() {
        return;
    }
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    let mut list: Vec<(usize, usize)> = Vec::new();
    os.with_held(&mut |held| {
        list.clear();
        words(first, last, false, |r, w, mask, base| {
            let Some(r) = r else {
                return;
            };
            r.refused[w].fetch_and(!mask, AcqRel);
            let t = (r.tracked[w].load(Acquire) | r.absent[w].load(Acquire)) & mask;
            if t == 0 {
                return;
            }
            let g = r.guarded[w].load(Acquire) & t;
            let mut v = t;
            while v != 0 {
                let b = v.trailing_zeros() as usize;
                v &= v - 1;
                drop_block(base + b);
            }
            let mut open = g;
            let mut v = g;
            while v != 0 {
                let b = v.trailing_zeros() as usize;
                v &= v - 1;
                if held(base + b) {
                    open &= !(1u64 << b);
                }
            }
            runs(open, base, &mut list);
        });
        for &(from, to) in &list {
            if os.protect(from, to, wanted(false, false)).is_none() {
                for b in from..to {
                    os.protect(b, b + 1, wanted(false, false));
                }
            }
        }
    });
    // Copies checked over it are not current any more.
    touched(first, last);
}

/// The fault path, from the guard handler: a write at `at`. `None` if the
/// block was never made read-only by us (not ours); `Some(true)` to make
/// the write again. Every block we ever made read-only is claimed: one
/// still guarded is opened and noted written; one another thread is
/// opening is waited for; one left read-only by a race (a take guarding it
/// again while it was dropped, excluded or forgotten) is opened here and
/// noted written. Allocates nothing; takes the lazy copy-back's `HELD`
/// lock if that is on.
pub(super) fn fault(at: usize) -> Option<bool> {
    let block = at / BLOCK;
    let r = region(block)?;
    let (w, bit) = ((block % REGION_BLOCKS) / 64, 1u64 << (block % 64));
    if (r.ours[w].load(Acquire) | r.guarded[w].load(Acquire)) & bit == 0 {
        return None;
    }
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let count_fault = || {
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::C;
            count(C::FaultFaults, 1);
            count(C::FaultUnprotected, 1);
            count(C::FaultUs, started.elapsed().as_micros() as u64);
        }
    };
    for round in 0..64u32 {
        {
            // Held across the bit and the protection: `lazy_hold` changes
            // both under it too.
            let held = LAZY.load(Relaxed).then(crate::lazy::held);
            if held.as_ref().is_some_and(|h| h.contains(&block)) {
                // The lazy copy-back took it meanwhile: its fault now.
                return Some(true);
            }
            if r.guarded[w].fetch_and(!bit, AcqRel) & bit != 0 {
                // Flying: its bytes before the write are kept first.
                if r.flying[w].load(Acquire) & bit != 0 {
                    cow(r, block);
                }
                let opened = protect_blocks(block, block + 1, wanted(false, false)).is_some();
                r.opened[w].fetch_or(bit, AcqRel);
                // After the access is back: a take between the two sees
                // the block neither guarded nor written, and leaves it.
                r.dirty[w].fetch_or(bit, AcqRel);
                drop(held);
                count_fault();
                // Not there any more (freed under us): the write fails as
                // it would have.
                return Some(opened || super::writable(at));
            }
            // Not guarded but read-only: another thread is opening it, or
            // a race left it so. After a few looks, opened here: we made it
            // read-only, and nothing else of ours wants it so.
            if super::writable(at) {
                return Some(true);
            }
            if round >= 4 {
                let opened = protect_blocks(block, block + 1, wanted(false, false)).is_some();
                r.opened[w].fetch_or(bit, AcqRel);
                if r.tracked[w].load(Acquire) & bit != 0 {
                    r.dirty[w].fetch_or(bit, AcqRel);
                }
                drop(held);
                count_fault();
                return Some(opened || super::writable(at));
            }
        }
        std::thread::yield_now();
    }
    Some(super::writable(at))
}

/// The copy-on-write pool for flying blocks: slots of a block each, made
/// once ([`cow_init`]) and never freed, so that the fault path takes one
/// without allocating.
struct Pool {
    memory: Box<[u8]>,
    /// One bit per slot, set while taken.
    taken: Box<[AtomicU64]>,
}

static POOL: OnceLock<Pool> = OnceLock::new();

/// `VICTORY_BACK_COW_MB` (64): the copy-on-write pool's size.
pub(crate) fn cow_init() {
    POOL.get_or_init(|| {
        let mb: usize = std::env::var("VICTORY_BACK_COW_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(64);
        let slots = (mb << 20) / BLOCK;
        let slots = slots.next_multiple_of(64).max(64);
        Pool {
            memory: vec![0u8; slots * BLOCK].into_boxed_slice(),
            taken: (0..slots / 64).map(|_| AtomicU64::new(0)).collect(),
        }
    });
}

/// The fault path (or [`open`]), before a flying guarded block becomes
/// writable: its bytes go to a pool slot, once. Allocates nothing.
fn cow(r: &Region, block: usize) {
    let (i, w, bit) = (block % REGION_BLOCKS, (block % REGION_BLOCKS) / 64, 1u64 << (block % 64));
    if r.cow[i].load(Acquire) != 0 || r.cow_lost[w].load(Acquire) & bit != 0 {
        return;
    }
    let Some(pool) = POOL.get() else {
        r.cow_lost[w].fetch_or(bit, AcqRel);
        count_lost();
        return;
    };
    for (k, word) in pool.taken.iter().enumerate() {
        let mut v = word.load(Acquire);
        while v != u64::MAX {
            let free = (!v).trailing_zeros() as usize;
            match word.compare_exchange(v, v | (1u64 << free), AcqRel, Acquire) {
                Ok(_) => {
                    let slot = k * 64 + free;
                    // SAFETY: the block is read-only (guarded) and
                    // committed; the slot is ours alone (claimed above).
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (block * BLOCK) as *const u8,
                            pool.memory.as_ptr().add(slot * BLOCK).cast_mut(),
                            BLOCK,
                        );
                    }
                    r.cow[i].store(slot as u32 + 1, Release);
                    #[cfg(feature = "diagnostics")]
                    count(crate::copystat::C::BackCows, 1);
                    return;
                }
                Err(now) => v = now,
            }
        }
    }
    r.cow_lost[w].fetch_or(bit, AcqRel);
    count_lost();
}

fn count_lost() {
    #[cfg(feature = "diagnostics")]
    count(crate::copystat::C::BackCowsLost, 1);
}

/// For a declaration of copy-back on demand (`crate::shadow`): whether
/// blocks `[first, last)` are all tracked, read-only by our hand, not
/// written since the last take and not volatile, so that title memory there
/// equals what a host copy refreshed just now holds, and any later write
/// faults first. If so they are noted flying (counted per declaration).
pub(crate) fn declare_flying(first: usize, last: usize) -> bool {
    if POOL.get().is_none() || first >= last {
        return false;
    }
    let mut ok = true;
    words(first, last, false, |r, w, mask, _| {
        ok &= r.is_some_and(|r| {
            r.tracked[w].load(Acquire) & mask == mask
                && r.guarded[w].load(Acquire) & mask == mask
                && (r.dirty[w].load(Acquire) | r.volatile[w].load(Acquire) | r.view[w].load(Acquire)) & mask == 0
        });
    });
    if !ok {
        return false;
    }
    for b in first..last {
        if let Some(r) = region(b) {
            r.fly[b % REGION_BLOCKS].fetch_add(1, AcqRel);
            r.flying[(b % REGION_BLOCKS) / 64].fetch_or(1u64 << (b % 64), AcqRel);
        }
    }
    true
}

/// A declaration on demand of blocks `[first, last)` is done with (its
/// last flight landed): the count falls. The flying bit and the block's
/// copy stay until its copy-back ([`landed_block`]).
pub(crate) fn undeclare_flying(first: usize, last: usize) {
    for b in first..last {
        if let Some(r) = region(b) {
            let f = &r.fly[b % REGION_BLOCKS];
            let _ = f.fetch_update(AcqRel, Acquire, |n| Some(n.saturating_sub(1)));
        }
    }
}

/// What the copy-back on demand compares a host copy's bytes of `block`
/// with: `Some(Some(bytes))`, the block as it was before the title's first
/// write since the declaration; `Some(None)`, title memory (not written
/// since); `None`, not known (the pool was full): the title's bytes stay.
pub(crate) fn before(block: usize) -> Option<Option<&'static [u8]>> {
    let Some(r) = region(block) else {
        return Some(None);
    };
    let (i, w, bit) = (block % REGION_BLOCKS, (block % REGION_BLOCKS) / 64, 1u64 << (block % 64));
    if r.cow_lost[w].load(Acquire) & bit != 0 {
        return None;
    }
    let slot = r.cow[i].load(Acquire);
    if slot == 0 {
        return Some(None);
    }
    let pool = POOL.get()?;
    // SAFETY: a slot of the pool, kept until `landed_block` frees it.
    Some(Some(unsafe { std::slice::from_raw_parts(pool.memory.as_ptr().add((slot as usize - 1) * BLOCK), BLOCK) }))
}

/// Puts `bytes` at `offset` of `block`'s copy-on-write bytes, if it has any
/// ([`before`]): the title's bytes there are known to every copy now
/// (`crate::shadow::Shadow::merge_under_writes`).
pub(crate) fn before_put(block: usize, offset: usize, bytes: &[u8]) {
    let Some(r) = region(block) else { return };
    let slot = r.cow[block % REGION_BLOCKS].load(Acquire);
    let Some(pool) = POOL.get() else { return };
    if slot == 0 || offset + bytes.len() > BLOCK {
        return;
    }
    // SAFETY: a slot of the pool, ours until `landed_block` frees it; the
    // pool's memory is written through this cast everywhere (`cow`).
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), pool.memory.as_ptr().add((slot as usize - 1) * BLOCK + offset).cast_mut(), bytes.len());
    }
}

/// The copy-back on demand of `block` is done: if no declaration holds it
/// any more, it stops flying and its copy goes back to the pool.
pub(crate) fn landed_block(block: usize) {
    let Some(r) = region(block) else {
        return;
    };
    let (i, w, bit) = (block % REGION_BLOCKS, (block % REGION_BLOCKS) / 64, 1u64 << (block % 64));
    if r.fly[i].load(Acquire) != 0 {
        return;
    }
    r.flying[w].fetch_and(!bit, AcqRel);
    r.cow_lost[w].fetch_and(!bit, AcqRel);
    let slot = r.cow[i].swap(0, AcqRel);
    if slot != 0
        && let Some(pool) = POOL.get()
    {
        let s = slot as usize - 1;
        pool.taken[s / 64].fetch_and(!(1u64 << (s % 64)), AcqRel);
    }
}

/// When [`verify`] last ran (`crate::host::now_ms`).
static VERIFIED: AtomicU64 = AtomicU64::new(0);

/// Whether [`verify`] is due (and notes it run).
pub(super) fn verify_due() -> bool {
    let now = crate::host::now_ms();
    let last = VERIFIED.load(Relaxed);
    now >= last + verify_ms() && VERIFIED.compare_exchange(last, now, Relaxed, Relaxed).is_ok()
}

/// Asks the kernel about every tracked block (`ask(first, last)`: the
/// written blocks of a range, resetting, or `None` if the range is not
/// write-watched). A block it saw written that was guarded and opened by
/// nobody since the last check was freed and made anew, or re-protected,
/// under us: it is pushed to `out` and no longer tracked. A block the
/// kernel no longer answers for is pushed and dropped too. Under the
/// journal's lock.
pub(super) fn verify(out: &mut Vec<usize>, ask: &mut dyn FnMut(usize, usize) -> Option<Vec<usize>>, os: &dyn Os) {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let regions: Vec<usize> = MADE.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let mut list: Vec<(usize, usize)> = Vec::new();
    for n in regions {
        let base = n * REGION_BLOCKS;
        let Some(r) = region(base) else {
            continue;
        };
        list.clear();
        for w in 0..WORDS {
            runs(r.tracked[w].load(Acquire) & !r.view[w].load(Acquire), base + w * 64, &mut list);
        }
        for &(from, to) in &list {
            verify_run(from, to, out, ask, os);
        }
    }
    #[cfg(feature = "diagnostics")]
    count(crate::copystat::C::FaultVerifyUs, started.elapsed().as_micros() as u64);
}

fn verify_run(from: usize, to: usize, out: &mut Vec<usize>, ask: &mut dyn FnMut(usize, usize) -> Option<Vec<usize>>, os: &dyn Os) {
    match ask(from, to) {
        Some(written) => {
            let mut stale = 0u64;
            for b in written {
                let Some(r) = region(b) else {
                    continue;
                };
                let (w, bit) = ((b % REGION_BLOCKS) / 64, 1u64 << (b % 64));
                let guarded = r.guarded[w].load(Acquire) & bit != 0;
                let opened = r.opened[w].load(Acquire) & bit != 0;
                if guarded && !opened {
                    out.push(b);
                    // Given its access back if it is read-only by us after all.
                    forget(b * BLOCK, BLOCK, os);
                    stale += 1;
                }
            }
            let _ = stale;
            #[cfg(feature = "diagnostics")]
            count(crate::copystat::C::FaultStale, stale);
            // Opened from now on counts from here.
            words(from, to, false, |r, w, mask, _| {
                if let Some(r) = r {
                    let keep = r.guarded[w].load(Acquire) & mask;
                    // Blocks writable now stay opened.
                    r.opened[w].fetch_and(!(mask & keep), AcqRel);
                }
            });
        }
        None if to - from > 1 => {
            let mid = from + (to - from) / 2;
            verify_run(from, mid, out, ask, os);
            verify_run(mid, to, out, ask, os);
        }
        None => {
            out.push(from);
            forget(from * BLOCK, BLOCK, os);
            #[cfg(feature = "diagnostics")]
            count(crate::copystat::C::FaultStale, 1);
        }
    }
}

/// For tests: a race's outcome, the block left read-only by us with its
/// tracking bits gone.
#[cfg(test)]
pub(super) fn lose_tracking(block: usize) {
    drop_block(block);
}

/// Blocks tracked now.
#[cfg(feature = "diagnostics")]
pub(crate) fn tracked() -> u64 {
    TRACKED.load(Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Protection by block, and held blocks.
    #[derive(Default)]
    struct Fake {
        prot: RefCell<std::collections::HashMap<usize, u32>>,
        held: std::collections::HashSet<usize>,
        fail: std::collections::HashSet<usize>,
        uncommitted: RefCell<std::collections::HashSet<usize>>,
    }

    impl Os for Fake {
        fn protect(&self, from: usize, to: usize, protection: u32) -> Option<u32> {
            if (from..to).any(|b| self.fail.contains(&b) || self.uncommitted.borrow().contains(&b)) {
                return None;
            }
            let mut p = self.prot.borrow_mut();
            let old = *p.get(&from).unwrap_or(&PAGE_READWRITE);
            for b in from..to {
                p.insert(b, protection);
            }
            Some(old)
        }
        fn uncommitted_end(&self, block: usize) -> usize {
            let u = self.uncommitted.borrow();
            let mut b = block;
            while u.contains(&b) {
                b += 1;
            }
            b
        }
        fn query(&self, block: usize) -> Option<(usize, bool, u32)> {
            // One block at a time.
            let committed = !self.uncommitted.borrow().contains(&block);
            Some(((block + 1) * BLOCK, committed, self.of(block)))
        }
        fn with_held(&self, f: &mut dyn FnMut(&dyn Fn(usize) -> bool)) {
            f(&|b| self.held.contains(&b));
        }
    }

    impl Fake {
        fn of(&self, b: usize) -> u32 {
            *self.prot.borrow().get(&b).unwrap_or(&PAGE_READWRITE)
        }
    }

    /// A block range of its own per test (regions are global).
    fn blocks(test: usize) -> usize {
        (0x7100_0000_0000usize / BLOCK) + test * REGION_BLOCKS
    }

    /// A write the fault path would note (without the system's protection).
    fn write(b: usize, os: &Fake) {
        let r = region(b).unwrap();
        let (w, bit) = ((b % REGION_BLOCKS) / 64, 1u64 << (b % 64));
        if r.guarded[w].fetch_and(!bit, AcqRel) & bit != 0 {
            os.protect(b, b + 1, PAGE_READWRITE);
        }
        r.opened[w].fetch_or(bit, AcqRel);
        r.dirty[w].fetch_or(bit, AcqRel);
    }

    #[test]
    fn adopted_blocks_are_guarded_and_writes_are_taken_once() {
        let os = Fake::default();
        let b = blocks(0) + 60; // across a word
        assert!(!all_tracked(b, b + 10));
        assert_eq!(adopt(b, b + 10, false, &mut Vec::new(), &os), 10);
        assert!(all_tracked(b, b + 10));
        assert!((b..b + 10).all(|x| os.of(x) == PAGE_READONLY && guarded(x)));
        let mut out = Vec::new();
        take(b, b + 10, &mut out, &os);
        assert!(out.is_empty());
        write(b + 3, &os);
        write(b + 5, &os);
        assert_eq!(os.of(b + 3), PAGE_READWRITE);
        take(b, b + 10, &mut out, &os);
        assert_eq!(out, vec![b + 3, b + 5]);
        // Guarded again by the take.
        assert_eq!(os.of(b + 3), PAGE_READONLY);
        assert!(guarded(b + 5));
        out.clear();
        take(b, b + 10, &mut out, &os);
        assert!(out.is_empty());
    }

    #[test]
    fn a_block_written_at_every_take_turns_volatile_and_is_tried_again() {
        let os = Fake::default();
        let b = blocks(1);
        adopt(b, b + 2, false, &mut Vec::new(), &os);
        let mut out = Vec::new();
        for _ in 0..volatile_after() {
            write(b, &os);
            out.clear();
            take(b, b + 2, &mut out, &os);
            assert_eq!(out, vec![b]);
        }
        // Volatile: left writable, reported at every take.
        assert_eq!(os.of(b), PAGE_READWRITE);
        assert!(!guarded(b));
        for _ in 0..RETRY - 1 {
            out.clear();
            take(b, b + 2, &mut out, &os);
            assert_eq!(out, vec![b]);
            assert_eq!(os.of(b), PAGE_READWRITE);
        }
        // Then guarded once more (and still reported this time).
        out.clear();
        take(b, b + 2, &mut out, &os);
        assert_eq!(out, vec![b]);
        assert_eq!(os.of(b), PAGE_READONLY);
        out.clear();
        take(b, b + 2, &mut out, &os);
        assert!(out.is_empty());
    }

    #[test]
    fn a_clean_take_ends_the_streak() {
        let os = Fake::default();
        let b = blocks(2);
        adopt(b, b + 1, false, &mut Vec::new(), &os);
        let mut out = Vec::new();
        for _ in 0..3 * u32::from(volatile_after()) {
            write(b, &os);
            take(b, b + 1, &mut out, &os);
            // One clean take between writes.
            take(b, b + 1, &mut out, &os);
        }
        assert!(guarded(b), "never written twice in a row: never volatile");
    }

    #[test]
    fn held_blocks_are_not_guarded_and_kernel_writes_open_and_mark() {
        let mut os = Fake::default();
        let b = blocks(3);
        os.held.insert(b + 1);
        os.prot.borrow_mut().insert(b + 1, PAGE_NOACCESS);
        adopt(b, b + 3, false, &mut Vec::new(), &os);
        // The held block keeps its protection (and was not read-write, so
        // it is not taken on).
        assert_eq!(os.of(b + 1), PAGE_NOACCESS);
        assert_eq!(os.of(b), PAGE_READONLY);
        assert_eq!(os.of(b + 2), PAGE_READONLY);
        open((b * BLOCK) + 10, 2 * BLOCK, &os);
        assert_eq!(os.of(b), PAGE_READWRITE);
        assert_eq!(os.of(b + 1), PAGE_NOACCESS);
        assert_eq!(os.of(b + 2), PAGE_READWRITE);
        let mut out = Vec::new();
        take(b, b + 3, &mut out, &os);
        assert!(out.contains(&b) && out.contains(&(b + 2)));
    }

    #[test]
    fn blocks_that_cannot_be_protected_stay_with_the_kernel() {
        let mut os = Fake::default();
        let b = blocks(4);
        os.fail.insert(b + 2);
        adopt(b, b + 4, false, &mut Vec::new(), &os);
        assert!(all_tracked(b, b + 2));
        assert!(!all_tracked(b + 2, b + 3));
        assert!(all_tracked(b + 3, b + 4));
        // Read-only for another reason: given its protection back, not
        // tracked.
        let os2 = Fake::default();
        let c = blocks(4) + 100;
        os2.prot.borrow_mut().insert(c, PAGE_READONLY);
        adopt(c, c + 1, false, &mut Vec::new(), &os2);
        assert_eq!(os2.of(c), PAGE_READONLY);
        assert!(!all_tracked(c, c + 1));
    }

    #[test]
    fn a_check_drops_blocks_written_behind_our_back() {
        let os = Fake::default();
        let b = blocks(5);
        adopt(b, b + 4, false, &mut Vec::new(), &os);
        let mut out = Vec::new();
        // First check: all opened at adoption, nothing is stale.
        verify_run(b, b + 4, &mut out, &mut |_, _| Some(vec![b, b + 1]), &os);
        assert!(out.is_empty());
        write(b + 2, &os);
        // b+3: written while we hold it read-only (memory made anew); b+2
        // opened by a fault: legitimate.
        verify_run(b, b + 4, &mut out, &mut |_, _| Some(vec![b + 2, b + 3]), &os);
        assert_eq!(out, vec![b + 3]);
        assert!(!all_tracked(b + 3, b + 4));
        assert!(all_tracked(b, b + 3));
        // A range the kernel no longer answers for, block by block.
        out.clear();
        verify_run(b, b + 3, &mut out, &mut |f, t| if (f..t).contains(&(b + 1)) { None } else { Some(Vec::new()) }, &os);
        assert_eq!(out, vec![b + 1]);
    }

    #[test]
    fn forgetting_gives_access_back() {
        let os = Fake::default();
        let b = blocks(6);
        adopt(b, b + 2, false, &mut Vec::new(), &os);
        forget(b * BLOCK, 2 * BLOCK, &os);
        assert!(!all_tracked(b, b + 1) && !all_tracked(b + 1, b + 2));
        assert_eq!(os.of(b), PAGE_READWRITE);
        assert!(!guarded(b));
    }

    #[test]
    fn runs_cover_whole_words() {
        let mut out = Vec::new();
        runs(u64::MAX, 0, &mut out);
        runs(0b1011, 64, &mut out);
        assert_eq!(out, vec![(0, 66), (67, 68)]);
    }

    #[test]
    fn uncommitted_view_blocks_are_covered_until_committed() {
        let os = Fake::default();
        let b = blocks(7);
        os.uncommitted.borrow_mut().insert(b + 1);
        let mut fresh = Vec::new();
        adopt(b, b + 3, true, &mut fresh, &os);
        // All three are written as far as anyone knows (copied once).
        assert_eq!(fresh, vec![b, b + 1, b + 2]);
        assert!(all_tracked(b, b + 3), "the absent block is covered");
        let mut out = Vec::new();
        take(b, b + 3, &mut out, &os);
        assert!(out.is_empty());
        // Committed: the next take reports it and guards it.
        os.uncommitted.borrow_mut().clear();
        take(b, b + 3, &mut out, &os);
        assert_eq!(out, vec![b + 1]);
        assert_eq!(os.of(b + 1), PAGE_READONLY);
        // A check of the kernel leaves view blocks alone.
        let mut stale = Vec::new();
        verify_run(b, b, &mut stale, &mut |_, _| None, &os);
        let mut asked = false;
        verify(&mut stale, &mut |f, t| {
            asked |= (f..t).contains(&b);
            Some(Vec::new())
        }, &os);
        assert!(!asked);
    }

    #[test]
    fn a_word_is_clean_until_a_take_finds_a_write_in_it() {
        let os = Fake::default();
        let b = blocks(8);
        let word = b / 64;
        adopt(b, b + 64, false, &mut Vec::new(), &os);
        let checked = epoch();
        assert!(word_clean(word, u64::MAX, checked));
        write(b + 5, &os);
        assert!(!word_clean(word, 1 << 5, checked), "written, not taken yet");
        assert!(word_clean(word, 1, checked), "other blocks: still clean");
        let mut out = Vec::new();
        take(b, b + 64, &mut out, &os);
        assert!(!word_clean(word, 1, checked), "the word moved on");
        assert!(word_clean(word, 1, epoch()));
        touched(b, b + 1);
        assert!(!word_clean(word, u64::MAX, epoch() - 1));
    }

    #[test]
    fn held_blocks_remember_whether_the_title_had_written_them() {
        let os = Fake::default();
        let b = blocks(9);
        adopt(b, b + 2, false, &mut Vec::new(), &os);
        write(b, &os);
        note_held(b);
        note_held(b + 1);
        assert!(cpu_ahead(b));
        assert!(!cpu_ahead(b + 1));
        released(b);
        assert!(!cpu_ahead(b));
        assert!(cpu_ahead(blocks(10) + 3), "untracked: nobody knows");
    }
}
