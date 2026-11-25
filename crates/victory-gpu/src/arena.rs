//! One host backing per title byte: an arena.
//!
//! Ported from Xenia's shared memory (`src/xenia/gpu/shared_memory.cc` and
//! `src/xenia/gpu/d3d12/d3d12_shared_memory.cc`, Copyright 2020 Ben Vanik,
//! BSD 3-clause licence). Changed for this project: a slice of title
//! address space instead of the console's 512 MiB, 64 KiB pages, tiles from
//! pooled heaps, and no command recording besides the tile mappings.
//!
//! An arena is one reserved host buffer over title addresses `[base, base +
//! size)`: title address `a` is byte `a - base` of the buffer, so title
//! buffers that overlap share their host bytes and each title byte is
//! uploaded once, not once per host copy that holds it. Nothing of the
//! buffer is backed until used: the 64 KiB tiles a request touches are
//! mapped then, from 64 MiB heaps made as they fill up (never given back).
//!
//! Per 64 KiB page the arena keeps two bits:
//! - `valid`: the host buffer holds the title's current bytes. A request
//!   uploads only the runs of pages that are not valid, then marks them.
//! - `gpu_written`: shaders wrote the page, so the host bytes are newer
//!   than the title's. Such a page is valid and a CPU write does not make
//!   it invalid (nothing uploads over it) until [`Arena::clear_gpu_written`].
//!
//! The page logic ([`Pages`], [`Tiles`]) has no D3D in it, for the tests;
//! [`Arena`] adds the reserved buffer, the heaps and the tile mappings.
//! The arena takes `&mut self` and does no locking: the caller holds it in
//! a lock if CPU-write invalidation comes from another thread.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::Graphics::Dxgi::DXGI_ERROR_UNSUPPORTED;

/// The page and tile size: a D3D12 tile, and the write-watch page.
pub(crate) const PAGE: u64 = 64 << 10;
const PAGE_SHIFT: u32 = 16;
const _: () = assert!(PAGE == D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES as u64);
const _: () = assert!(PAGE == crate::watch::PAGE as u64);
const _: () = assert!(1 << PAGE_SHIFT == PAGE);

/// Tiles per heap: 64 MiB heaps.
const HEAP_TILES: u64 = 1024;

/// The bits of 64-page block `block` that lie in pages `[first, last]`.
fn block_mask(block: u64, first: u64, last: u64) -> u64 {
    let mut mask = u64::MAX;
    if block == first >> 6 {
        mask &= u64::MAX << (first & 63);
    }
    if block == last >> 6 {
        mask &= u64::MAX >> (63 - (last & 63));
    }
    mask
}

/// Appends to `out` the runs `(first page, page count)` of pages in
/// `[first, last]` whose bit in `bits` is clear, in order.
///
/// Xenia's `SharedMemory::TryFindUploadRange` and `TryGetNextUploadRange`:
/// per 64-page block, pages outside the range count as set, and the scan
/// alternates between the next clear bit (a run opens) and the next set bit
/// after the run's start (it closes), with `trailing_zeros`.
fn clear_runs(bits: &[u64], first: u64, last: u64, out: &mut Vec<(u64, u64)>) {
    let mut start: Option<u64> = None;
    for block in first >> 6..=last >> 6 {
        let mut set = bits[block as usize] | !block_mask(block, first, last);
        loop {
            match start {
                None => {
                    let clear = !set;
                    if clear == 0 {
                        break;
                    }
                    start = Some((block << 6) + u64::from(clear.trailing_zeros()));
                }
                Some(s) => {
                    // Set bits before the run's start do not close it.
                    let mut closing = set;
                    if block == s >> 6 {
                        closing &= u64::MAX << (s & 63);
                    }
                    if closing == 0 {
                        break;
                    }
                    let bit = closing.trailing_zeros();
                    out.push((s, (block << 6) + u64::from(bit) - s));
                    // The run is taken: its pages count as set for the rest
                    // of the block's scan.
                    set |= (1u64 << bit) - 1;
                    start = None;
                }
            }
        }
    }
    if let Some(s) = start {
        out.push((s, last + 1 - s));
    }
}

/// Sets (`on`) or clears the bits of pages `[first, last]`.
fn set_bits(bits: &mut [u64], first: u64, last: u64, on: bool) {
    for block in first >> 6..=last >> 6 {
        let mask = block_mask(block, first, last);
        if on {
            bits[block as usize] |= mask;
        } else {
            bits[block as usize] &= !mask;
        }
    }
}

/// Per 64 KiB page of an arena: whether the host holds the title's bytes,
/// and whether shaders wrote it. Pages are numbered from the arena's base.
pub(crate) struct Pages {
    valid: Vec<u64>,
    /// Always inside `valid`.
    gpu_written: Vec<u64>,
    count: u64,
}

impl Pages {
    pub(crate) fn new(count: u64) -> Pages {
        let words = count.div_ceil(64) as usize;
        Pages { valid: vec![0; words], gpu_written: vec![0; words], count }
    }

    /// Appends the runs of pages in `[first, last]` that are not valid,
    /// as `(first page, page count)`.
    pub(crate) fn upload_runs(&self, first: u64, last: u64, out: &mut Vec<(u64, u64)>) {
        clear_runs(&self.valid, first, last, out);
    }

    /// Whether every page in `[first, last]` is valid (Xenia's
    /// `IsRangeValid`).
    pub(crate) fn all_valid(&self, first: u64, last: u64) -> bool {
        (first >> 6..=last >> 6).all(|b| {
            let mask = block_mask(b, first, last);
            self.valid[b as usize] & mask == mask
        })
    }

    /// Pages `[first, last]` hold the title's bytes (`gpu` false: just
    /// uploaded) or bytes shaders wrote (`gpu` true). Xenia's
    /// `MakeRangeValid`: an upload also ends the pages' GPU-written state.
    pub(crate) fn make_valid(&mut self, first: u64, last: u64, gpu: bool) {
        set_bits(&mut self.valid, first, last, true);
        set_bits(&mut self.gpu_written, first, last, gpu);
    }

    /// Pages `[first, last]` are no longer GPU-written (their bytes reached
    /// the title, or the title's bytes win); they stay valid.
    pub(crate) fn clear_gpu_written(&mut self, first: u64, last: u64) {
        set_bits(&mut self.gpu_written, first, last, false);
    }

    /// The CPU wrote pages `[first, last]`: they stop being valid, except
    /// GPU-written ones. With `widen`, the range first grows within its
    /// first and last 64-page block up to the nearest GPU-written page
    /// (Xenia's non-exact `MemoryInvalidationCallback`: some excess upload
    /// is cheaper than another write fault). Returns the pages covered.
    ///
    /// Xenia clears GPU-written pages inside the exact range too (the CPU
    /// write wins); here that takes [`Pages::clear_gpu_written`] first.
    pub(crate) fn invalidate(&mut self, mut first: u64, mut last: u64, widen: bool) -> (u64, u64) {
        if widen {
            if first & 63 != 0 {
                let below = self.gpu_written[(first >> 6) as usize] & ((1u64 << (first & 63)) - 1);
                first = (first & !63) + u64::from(64 - below.leading_zeros());
            }
            if last & 63 != 63 {
                let above = self.gpu_written[(last >> 6) as usize] & !(u64::MAX >> (63 - (last & 63)));
                last = (last & !63) + u64::from(above.trailing_zeros().max(1) - 1);
                last = last.min(self.count - 1);
            }
        }
        for block in first >> 6..=last >> 6 {
            let b = block as usize;
            self.valid[b] &= !(block_mask(block, first, last) & !self.gpu_written[b]);
        }
        (first, last)
    }
}

/// One mapping of arena tiles onto heap tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Mapping {
    /// The first arena tile (from the arena's base).
    pub(crate) tile: u64,
    /// Which heap, and its first tile.
    pub(crate) heap: usize,
    pub(crate) heap_tile: u32,
    pub(crate) tiles: u32,
}

/// Which arena tiles are mapped, and the next free heap tile. Heap tiles
/// are handed out in order and never given back, so heap `h` holds heap
/// tiles `[h * HEAP_TILES, (h + 1) * HEAP_TILES)` of the pool.
pub(crate) struct Tiles {
    mapped: Vec<u64>,
    next: u64,
}

impl Tiles {
    pub(crate) fn new(count: u64) -> Tiles {
        Tiles { mapped: vec![0; count.div_ceil(64) as usize], next: 0 }
    }

    /// The mappings that would map every tile of `[first, last]`, in heap
    /// order; a run of unmapped tiles is split where a heap ends. Nothing
    /// changes until [`Tiles::apply`] (Xenia's
    /// `EnsureHostGpuMemoryAllocated`, which maps whole unmapped runs).
    pub(crate) fn plan(&self, first: u64, last: u64) -> Vec<Mapping> {
        let mut runs = Vec::new();
        clear_runs(&self.mapped, first, last, &mut runs);
        let mut next = self.next;
        let mut plan = Vec::new();
        for (mut tile, mut left) in runs {
            while left > 0 {
                let in_heap = next % HEAP_TILES;
                let n = left.min(HEAP_TILES - in_heap);
                plan.push(Mapping {
                    tile,
                    heap: (next / HEAP_TILES) as usize,
                    heap_tile: in_heap as u32,
                    tiles: n as u32,
                });
                tile += n;
                left -= n;
                next += n;
            }
        }
        plan
    }

    /// How many heaps `plan` needs in all.
    pub(crate) fn heaps_for(&self, plan: &[Mapping]) -> usize {
        plan.last().map_or(0, |m| m.heap + 1)
    }

    /// `plan` (from [`Tiles::plan`]) was mapped.
    pub(crate) fn apply(&mut self, plan: &[Mapping]) {
        for m in plan {
            set_bits(&mut self.mapped, m.tile, m.tile + u64::from(m.tiles) - 1, true);
            self.next += u64::from(m.tiles);
        }
    }

    /// Heap tiles handed out.
    pub(crate) fn used(&self) -> u64 {
        self.next
    }
}

/// A reserved host buffer over title addresses `[base, base + size)`, with
/// its page state and tile heaps.
pub(crate) struct Arena {
    device: ID3D12Device,
    resource: ID3D12Resource,
    gpu: u64,
    base: u64,
    size: u64,
    heaps: Vec<ID3D12Heap>,
    heap_flags: D3D12_HEAP_FLAGS,
    pages: Pages,
    tiles: Tiles,
    /// Reused for upload runs.
    runs: Vec<(u64, u64)>,
}

// SAFETY: D3D12 devices, resources and heaps are free-threaded; the page
// state changes only through `&mut self`.
unsafe impl Send for Arena {}
// SAFETY: as above; `&self` methods only read.
unsafe impl Sync for Arena {}

impl Arena {
    /// Reserves the buffer for `[base, base + size)` (`base` and `size` on
    /// 64 KiB lines). Nothing is backed yet. Fails without tiled resources
    /// or if the device cannot address `size` in one resource.
    pub(crate) fn new(device: &ID3D12Device, base: u64, size: u64) -> windows::core::Result<Arena> {
        assert!(size > 0 && base.is_multiple_of(PAGE) && size.is_multiple_of(PAGE), "arena not on 64 KiB lines");
        // Tile coordinates are 32-bit.
        assert!(size / PAGE <= u64::from(u32::MAX), "arena too big");
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        // SAFETY: a plain query into a struct of the asked size.
        unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut options).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
            )?;
        }
        if options.TiledResourcesTier == D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED {
            return Err(windows::core::Error::new(DXGI_ERROR_UNSUPPORTED, "no tiled resources"));
        }
        let bits = options.MaxGPUVirtualAddressBitsPerResource;
        if bits < 64 && size > 1u64 << bits {
            return Err(windows::core::Error::new(DXGI_ERROR_UNSUPPORTED, "arena beyond the resource address bits"));
        }
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: plain object creation with a valid description. Common
        // state: a buffer is promoted to what each use needs.
        unsafe { device.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut resource)? };
        let resource = resource.ok_or_else(|| windows::core::Error::from(DXGI_ERROR_UNSUPPORTED))?;
        // SAFETY: a plain query.
        let gpu = unsafe { resource.GetGPUVirtualAddress() };
        let pages = size / PAGE;
        tracing::info!(
            base = format_args!("{base:#x}"),
            mb = size >> 20,
            tier = options.TiledResourcesTier.0,
            "arena reserved"
        );
        Ok(Arena {
            device: device.clone(),
            resource,
            gpu,
            base,
            size,
            heaps: Vec::new(),
            heap_flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED,
            pages: Pages::new(pages),
            tiles: Tiles::new(pages),
            runs: Vec::new(),
        })
    }

    /// The reserved buffer.
    pub(crate) fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }

    /// The title addresses the arena covers, `(base, size)`.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.base, self.size)
    }

    /// Whether `[address, address + len)` lies wholly in the arena.
    pub(crate) fn contains(&self, address: u64, len: u64) -> bool {
        address >= self.base && address.checked_add(len).is_some_and(|end| end <= self.base + self.size)
    }

    /// The host GPU address of title address `address` (in the arena).
    pub(crate) fn gpu_address(&self, address: u64) -> u64 {
        debug_assert!(self.contains(address, 0));
        self.gpu + (address - self.base)
    }

    /// Bytes of heap memory mapped into the arena.
    pub(crate) fn committed_bytes(&self) -> u64 {
        self.tiles.used() * PAGE
    }

    /// `[address, address + len)` cut to the arena, as pages `[first,
    /// last]`; `None` if nothing is left.
    fn pages_of(&self, address: u64, len: u64) -> Option<(u64, u64)> {
        let from = address.max(self.base);
        let to = address.saturating_add(len).min(self.base + self.size);
        (from < to).then(|| ((from - self.base) >> PAGE_SHIFT, (to - 1 - self.base) >> PAGE_SHIFT))
    }

    /// Makes the host bytes of `[address, address + len)` (cut to the
    /// arena) current: maps the tiles not yet mapped (on `queue`, ahead of
    /// lists later executed on it), then calls `upload(address, len)` for
    /// each run of whole pages that is not valid, which the caller copies
    /// from title memory (an upload ring and [`Arena::record_copy`]). Runs
    /// are marked valid before `upload` is called, as in Xenia, so a CPU
    /// write racing the copy invalidates them again.
    ///
    /// Xenia's `RequestRange`. Runs cover whole pages, which may reach
    /// outside the asked range: the caller copies what of them is title
    /// memory.
    pub(crate) fn request_range(
        &mut self,
        queue: &ID3D12CommandQueue,
        address: u64,
        len: u64,
        mut upload: impl FnMut(u64, u64),
    ) -> windows::core::Result<()> {
        let Some((first, last)) = self.pages_of(address, len) else {
            return Ok(());
        };
        self.map(queue, first, last)?;
        let mut runs = std::mem::take(&mut self.runs);
        runs.clear();
        self.pages.upload_runs(first, last, &mut runs);
        for &(page, n) in &runs {
            self.pages.make_valid(page, page + n - 1, false);
        }
        for &(page, n) in &runs {
            upload(self.base + (page << PAGE_SHIFT), n << PAGE_SHIFT);
        }
        self.runs = runs;
        Ok(())
    }

    /// Whether every page of `[address, address + len)` is valid.
    pub(crate) fn is_valid(&self, address: u64, len: u64) -> bool {
        self.pages_of(address, len).is_none_or(|(first, last)| self.pages.all_valid(first, last))
    }

    /// The title wrote `[address, address + len)`: its pages stop being
    /// valid except GPU-written ones (see [`Pages::invalidate`] for
    /// `widen`). Returns the title range covered, `(address, len)`, or
    /// `None` outside the arena.
    pub(crate) fn invalidate(&mut self, address: u64, len: u64, widen: bool) -> Option<(u64, u64)> {
        let (first, last) = self.pages_of(address, len)?;
        let (first, last) = self.pages.invalidate(first, last, widen);
        Some((self.base + (first << PAGE_SHIFT), (last - first + 1) << PAGE_SHIFT))
    }

    /// Shaders wrote `[address, address + len)`: its pages are valid and
    /// GPU-written (Xenia's `RangeWrittenByGpu`). Whole pages: a page
    /// partly written counts as written.
    pub(crate) fn written_by_gpu(&mut self, address: u64, len: u64) {
        if let Some((first, last)) = self.pages_of(address, len) {
            self.pages.make_valid(first, last, true);
        }
    }

    /// The pages of `[address, address + len)` are no longer GPU-written
    /// (their bytes went back to title memory, or the title's win): a CPU
    /// write invalidates them again. They stay valid.
    pub(crate) fn clear_gpu_written(&mut self, address: u64, len: u64) {
        if let Some((first, last)) = self.pages_of(address, len) {
            self.pages.clear_gpu_written(first, last);
        }
    }

    /// Records into `list` the copy of an upload run (`address`, `len`, as
    /// `request_range` gave it) from `upload` at `upload_offset` into the
    /// arena. The arena must be in (or promotable to) copy-destination
    /// state.
    pub(crate) fn record_copy(
        &self,
        list: &ID3D12GraphicsCommandList,
        upload: &ID3D12Resource,
        upload_offset: u64,
        address: u64,
        len: u64,
    ) {
        debug_assert!(self.contains(address, len));
        // SAFETY: both are buffers and the range lies in each (the caller's
        // upload holds `len` bytes at `upload_offset`).
        unsafe { list.CopyBufferRegion(&self.resource, address - self.base, upload, upload_offset, len) };
    }

    /// Maps every tile of pages `[first, last]` not yet mapped: heaps made
    /// as needed, then one `UpdateTileMappings` per heap with all of its
    /// new regions (Xenia's `AllocateSparseHostGpuMemoryRange`, which
    /// makes one heap per run instead).
    fn map(&mut self, queue: &ID3D12CommandQueue, first: u64, last: u64) -> windows::core::Result<()> {
        let plan = self.tiles.plan(first, last);
        if plan.is_empty() {
            return Ok(());
        }
        while self.heaps.len() < self.tiles.heaps_for(&plan) {
            let heap = self.new_heap()?;
            self.heaps.push(heap);
        }
        for group in plan.chunk_by(|a, b| a.heap == b.heap) {
            let n = group.len() as u32;
            let coordinates: Vec<D3D12_TILED_RESOURCE_COORDINATE> = group
                .iter()
                .map(|m| D3D12_TILED_RESOURCE_COORDINATE { X: m.tile as u32, Y: 0, Z: 0, Subresource: 0 })
                .collect();
            let sizes: Vec<D3D12_TILE_REGION_SIZE> = group
                .iter()
                .map(|m| D3D12_TILE_REGION_SIZE { NumTiles: m.tiles, UseBox: false.into(), ..Default::default() })
                .collect();
            let flags = vec![D3D12_TILE_RANGE_FLAG_NONE; group.len()];
            let offsets: Vec<u32> = group.iter().map(|m| m.heap_tile).collect();
            let counts: Vec<u32> = group.iter().map(|m| m.tiles).collect();
            // SAFETY: each array holds `n` entries; the regions lie in the
            // reserved buffer and the ranges in the heap.
            unsafe {
                queue.UpdateTileMappings(
                    &self.resource,
                    n,
                    Some(coordinates.as_ptr()),
                    Some(sizes.as_ptr()),
                    &self.heaps[group[0].heap],
                    n,
                    Some(flags.as_ptr()),
                    Some(offsets.as_ptr()),
                    Some(counts.as_ptr()),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
        }
        self.tiles.apply(&plan);
        tracing::debug!(regions = plan.len(), mb = self.committed_bytes() >> 20, "arena tiles mapped");
        Ok(())
    }

    /// A 64 MiB heap for buffer tiles, not zeroed if the system allows.
    fn new_heap(&mut self) -> windows::core::Result<ID3D12Heap> {
        let mut desc = D3D12_HEAP_DESC {
            SizeInBytes: HEAP_TILES * PAGE,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: self.heap_flags,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain object creation with a valid description.
        let made = unsafe { self.device.CreateHeap(&desc, &mut heap) };
        if made.is_err() && self.heap_flags.contains(D3D12_HEAP_FLAG_CREATE_NOT_ZEROED) {
            // Older systems refuse the flag.
            self.heap_flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
            desc.Flags = self.heap_flags;
            // SAFETY: as above.
            unsafe { self.device.CreateHeap(&desc, &mut heap)? };
        } else {
            made?;
        }
        heap.ok_or_else(|| windows::core::Error::from(DXGI_ERROR_UNSUPPORTED))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(pages: &Pages, first: u64, last: u64) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        pages.upload_runs(first, last, &mut out);
        out
    }

    #[test]
    fn nothing_valid_is_one_run() {
        let pages = Pages::new(512);
        assert_eq!(runs(&pages, 3, 200), vec![(3, 198)]);
        assert_eq!(runs(&pages, 64, 127), vec![(64, 64)]);
        assert_eq!(runs(&pages, 5, 5), vec![(5, 1)]);
    }

    #[test]
    fn runs_skip_valid_pages_across_blocks() {
        let mut pages = Pages::new(512);
        pages.make_valid(10, 20, false);
        pages.make_valid(60, 70, false);
        pages.make_valid(130, 130, false);
        assert_eq!(runs(&pages, 0, 199), vec![(0, 10), (21, 39), (71, 59), (131, 69)]);
        // Starting inside a valid stretch, ending inside a hole.
        assert_eq!(runs(&pages, 15, 100), vec![(21, 39), (71, 30)]);
        assert!(runs(&pages, 10, 20).is_empty());
        assert!(pages.all_valid(60, 70));
        assert!(!pages.all_valid(59, 70));
    }

    #[test]
    fn runs_up_to_the_last_page() {
        let mut pages = Pages::new(128);
        pages.make_valid(0, 126, false);
        assert_eq!(runs(&pages, 0, 127), vec![(127, 1)]);
        pages.make_valid(127, 127, false);
        assert!(runs(&pages, 0, 127).is_empty());
    }

    #[test]
    fn invalidate_keeps_gpu_written_pages() {
        let mut pages = Pages::new(256);
        pages.make_valid(0, 255, false);
        pages.make_valid(40, 41, true);
        assert_eq!(pages.invalidate(30, 50, false), (30, 50));
        assert_eq!(runs(&pages, 0, 255), vec![(30, 10), (42, 9)]);
        // Once the GPU-written state ends, a CPU write invalidates them.
        pages.clear_gpu_written(40, 41);
        pages.invalidate(40, 41, false);
        assert_eq!(runs(&pages, 0, 255), vec![(30, 21)]);
        // An upload ends the GPU-written state too.
        pages.make_valid(100, 100, true);
        pages.make_valid(100, 100, false);
        pages.invalidate(100, 100, false);
        assert_eq!(runs(&pages, 100, 100), vec![(100, 1)]);
    }

    #[test]
    fn widening_stops_at_gpu_written_pages() {
        let mut pages = Pages::new(256);
        pages.make_valid(0, 255, false);
        pages.make_valid(70, 70, true);
        pages.make_valid(100, 100, true);
        // Grows to the GPU-written pages' neighbours, within the block.
        assert_eq!(pages.invalidate(80, 90, true), (71, 99));
        // No GPU-written page nearby: the whole block.
        assert_eq!(pages.invalidate(130, 131, true), (128, 191));
        // The last block is cut to the page count.
        let mut short = Pages::new(100);
        assert_eq!(short.invalidate(70, 70, true), (64, 99));
    }

    #[test]
    fn tiles_split_at_heap_ends() {
        let mut tiles = Tiles::new(4096);
        let plan = tiles.plan(10, 19);
        assert_eq!(plan, vec![Mapping { tile: 10, heap: 0, heap_tile: 0, tiles: 10 }]);
        tiles.apply(&plan);
        // Mapped tiles are skipped; the pool runs over into heap 1.
        let plan = tiles.plan(0, 1100);
        assert_eq!(
            plan,
            vec![
                Mapping { tile: 0, heap: 0, heap_tile: 10, tiles: 10 },
                Mapping { tile: 20, heap: 0, heap_tile: 20, tiles: 1004 },
                Mapping { tile: 1024, heap: 1, heap_tile: 0, tiles: 77 },
            ]
        );
        assert_eq!(tiles.heaps_for(&plan), 2);
        tiles.apply(&plan);
        assert!(tiles.plan(0, 1100).is_empty());
        assert_eq!(tiles.used(), 1101);
    }
}
