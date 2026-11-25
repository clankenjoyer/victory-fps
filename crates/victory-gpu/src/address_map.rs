//! The address map translated shaders read title memory through when they
//! compute its address (a pointer they load, the BVH a ray-tracing pass
//! walks): see `victory_shader::emit::address_map` for its layout. It
//! lists the host copies of title memory made so far ([`crate::shadow`]),
//! each through raw views in dynamic slots of the view heap a list binds.
//!
//! shortcut: every copy is brought up to date (the pages the title wrote
//! since) once per list that binds the map, and what shaders write through
//! it stays in the copies (it is not copied back to title memory: the BVH
//! the title builds on the GPU is only read on the GPU). A list of copies
//! that grows into the thousands makes that refresh the cost to watch.

use std::sync::Arc;

use victory_shader::emit::address_map::{ENTRY, HEADER, Range, encode};

use crate::descriptor::host::HostHeap;
use crate::host::HostList;
use crate::shadow::Shadow;

/// The most bytes one raw view holds: D3D12 buffer views have at most 2^27
/// elements.
const VIEW_BYTES: u64 = 4 << 27;

/// Pieces of title memory the spans `(start, len)` cover, sorted and not
/// overlapping, each with the span that holds it: where spans overlap, the
/// one starting last (the innermost buffer, as [`crate::shadow::find`]
/// picks) holds the piece.
pub(crate) fn pieces(spans: &[(u64, u64)]) -> Vec<(u64, u64, usize)> {
    // (address, whether a span starts there, span): ends before starts.
    let mut points: Vec<(u64, bool, usize)> = Vec::with_capacity(2 * spans.len());
    for (i, &(start, len)) in spans.iter().enumerate() {
        if len > 0 {
            points.push((start, true, i));
            points.push((start.saturating_add(len), false, i));
        }
    }
    points.sort_unstable();
    let mut active = std::collections::BTreeSet::new();
    let mut out: Vec<(u64, u64, usize)> = Vec::new();
    let mut at = 0;
    for (x, starts, i) in points {
        if x > at
            && let Some(&(_, holder)) = active.iter().next_back()
        {
            match out.last_mut() {
                Some(last) if last.2 == holder && last.0 + last.1 == at => last.1 += x - at,
                _ => out.push((at, x - at, holder)),
            }
        }
        if starts {
            active.insert((spans[i].0, i));
        } else {
            active.remove(&(spans[i].0, i));
        }
        at = x;
    }
    out
}

/// The map's entries for `pieces` of the spans: each piece cut where its
/// span's views ([`VIEW_BYTES`] each) end, with the view (span, view number
/// within the span) and the piece's offset in that view.
fn entries(
    spans: &[(u64, u64)],
    pieces: &[(u64, u64, usize)],
) -> Vec<(u64, u32, (usize, u64), u32)> {
    let mut out = Vec::new();
    for &(start, len, span) in pieces {
        let mut at = start;
        let end = start + len;
        while at < end {
            let rel = at - spans[span].0;
            let view = rel / VIEW_BYTES;
            let view_end = spans[span].0 + (view + 1) * VIEW_BYTES;
            let next = end.min(view_end);
            out.push((
                at,
                (next - at) as u32,
                (span, view),
                (rel - view * VIEW_BYTES) as u32,
            ));
            at = next;
        }
    }
    out
}

/// The address map for draws of `list` while `heap` is its view heap: its
/// host GPU address, and the dynamic slots its views take (to give back
/// when the list is reset). Each copy is brought up to date when the list
/// is submitted.
pub(crate) fn make(list: &HostList, heap: &HostHeap) -> Option<(u64, (u32, u32))> {
    let copies: Vec<Arc<Shadow>> = crate::shadow::all();
    let spans: Vec<(u64, u64)> = copies.iter().map(|c| c.span()).collect();
    let pieces = pieces(&spans);
    let entries = entries(&spans, &pieces);
    let mut views: Vec<(usize, u64)> = entries.iter().map(|e| e.2).collect();
    views.sort_unstable();
    views.dedup();
    let n = views.len() as u32;
    let first = if n == 0 { 0 } else { heap.dynamic(n)? };
    for (k, &(span, view)) in views.iter().enumerate() {
        let copy = &copies[span];
        let offset = view * VIEW_BYTES;
        let bytes = (copy.span().1 - offset).min(VIEW_BYTES).next_multiple_of(4);
        heap.raw_uav_at(
            first + k as u32,
            &copy.buffer,
            offset,
            bytes,
        );
        // Shaders writing through the map mark the pages they write
        // (`victory_shader::emit`, "Write marks"), as through the title's
        // views: a range of the copy declared for those may hold them.
        let (start, _) = copy.span();
        heap.set_mark(first + k as u32, copy.mark_entry(start + offset, 0));
    }
    for c in &copies {
        if list.first_use(Arc::as_ptr(c) as usize ^ MAP_KEY) {
            let (start, len) = c.span();
            c.refresh(list, start, len);
        }
    }
    let ranges: Vec<Range> = entries
        .iter()
        .map(|&(start, len, view, view_offset)| Range {
            start,
            len,
            view: first + views.binary_search(&view).expect("listed") as u32,
            view_offset,
        })
        .collect();
    let words = encode(&ranges);
    // Room for one entry even when there is none: the shader reads it.
    let bytes = u64::from(HEADER) + u64::from(ENTRY) * ranges.len().max(1) as u64;
    let (gpu, to) = list.scratch(bytes)?;
    // SAFETY: `bytes` of the list's upload memory, which outlives the
    // list's run; the words fit.
    unsafe {
        std::ptr::write_bytes(to, 0, bytes as usize);
        std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), to, words.len() * 4);
    }
    Some((gpu, (first, n)))
}

/// Keeps the map's refreshes apart from other first uses the list asks
/// about (texture keys are addresses too).
const MAP_KEY: usize = 0x5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_buffers_win_and_pieces_do_not_overlap() {
        // A heap-sized buffer with two placed inside it, one past its end,
        // and an empty one.
        let spans = [
            (0x1000, 0x1000),
            (0x1200, 0x100),
            (0x1800, 0x100),
            (0x3000, 0x40),
            (0x5000, 0),
        ];
        assert_eq!(
            pieces(&spans),
            [
                (0x1000, 0x200, 0),
                (0x1200, 0x100, 1),
                (0x1300, 0x500, 0),
                (0x1800, 0x100, 2),
                (0x1900, 0x700, 0),
                (0x3000, 0x40, 3),
            ]
        );
    }

    #[test]
    fn the_same_span_twice_is_one_piece() {
        assert_eq!(
            pieces(&[(0x100, 0x100), (0x100, 0x100)]),
            [(0x100, 0x100, 1)]
        );
    }

    #[test]
    fn pieces_are_cut_where_views_end() {
        let spans = [(0x10, 2 * VIEW_BYTES + 8)];
        let e = entries(&spans, &pieces(&spans));
        assert_eq!(
            e,
            [
                (0x10, VIEW_BYTES as u32, (0, 0), 0),
                (0x10 + VIEW_BYTES, VIEW_BYTES as u32, (0, 1), 0),
                (0x10 + 2 * VIEW_BYTES, 8, (0, 2), 0),
            ]
        );
    }
}
