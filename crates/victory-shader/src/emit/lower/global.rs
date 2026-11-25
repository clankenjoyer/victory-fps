//! Title memory at addresses a program computes ([`GLOBAL`]): pointers it
//! loads, picks by control flow or builds, and the nodes of a BVH it walks.
//! Such an address names no root parameter, so the host binds the address
//! map ([`crate::emit::address_map`]) and a read looks the address up in it
//! (a binary search over its ranges) and loads through the raw view the
//! range names. An address no range covers reads zero.

use super::*;
use crate::emit::address_map::{ENTRY, HEADER};
use crate::ir::Value;
use crate::resource::GLOBAL;

/// A raw UAV (`RWByteAddressBuffer`): the map's views are UAVs so that the
/// same view serves the program's stores.
const RAW_UAV: (u32, u32) = (11 | 0x1000, 0);

impl Lower<'_> {
    /// Whether the access at `site` reads title memory through the map.
    pub(super) fn is_global(&self, site: Site) -> bool {
        self.scalar.get(&site).is_some_and(|a| a.param == GLOBAL)
    }

    /// `lo | hi << 32`, an i64.
    pub(super) fn wide(&mut self, lo: V, hi: V) -> V {
        let c = self.cur;
        let t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let lo = self.s.m.cast(c, Cast::ZExt, lo, t);
        let hi = self.s.m.cast(c, Cast::ZExt, hi, t);
        let k = self.s.m.int_const(t, 32);
        let hi = self.s.m.bin(c, Bin::Shl, hi, k);
        self.s.m.bin(c, Bin::Or, hi, lo)
    }

    /// `x` (an i32) widened to i64, sign-extended when `signed`.
    pub(super) fn widen(&mut self, x: V, signed: bool) -> V {
        let t = self.s.m.int(64);
        let op = if signed { Cast::SExt } else { Cast::ZExt };
        self.s.m.cast(self.cur, op, x, t)
    }

    /// `n` (at most 4) dwords of the address map from byte `at`.
    fn map_dwords(&mut self, at: V, n: u32) -> Vec<V> {
        let h = self.s.address_map_handle();
        self.raw_dwords(h, at, n)
    }

    /// `n` (at most 4) dwords of raw buffer `h` from byte `at`.
    pub(super) fn raw_dwords(&mut self, h: V, at: V, n: u32) -> Vec<V> {
        debug_assert!((1..=4).contains(&n));
        let c = self.cur;
        let (u, mask, al) = (
            self.s.m.undef(self.s.t.i32),
            self.s.m.i8((1 << n) - 1),
            self.s.m.i32(4),
        );
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[h, at, u, mask, al],
        );
        (0..n)
            .map(|k| {
                let x = self.s.m.extract(self.cur, r, k);
                self.as_int(x)
            })
            .collect()
    }

    /// Looks up title address `va` (an i64) in the address map: whether a
    /// range covers it, the heap index of the range's view, and the byte
    /// in that view `va` is at.
    pub(super) fn map_lookup(&mut self, va: V) -> (V, V, V) {
        let i32t = self.s.t.i32;
        let zero = self.s.m.i32(0);
        let count = self.map_dwords(zero, 1)[0];
        // The last range starting at or below `va`: halve the candidates
        // [lo, lo + len) until one is left.
        let pre = self.cur;
        let (head, body, done) = (
            self.s.m.new_block(),
            self.s.m.new_block(),
            self.s.m.new_block(),
        );
        self.s.m.terminate(pre, Term::Br(head));
        let lo = self.s.m.phi(head, i32t);
        let len = self.s.m.phi(head, i32t);
        let one = self.s.m.i32(1);
        let more = self.s.m.cmp(head, pred::IUGT, len, one);
        self.s.m.terminate(head, Term::CondBr(more, body, done));
        self.cur = body;
        let half = self.s.m.bin(body, Bin::LShr, len, one);
        let mid = self.s.m.bin(body, Bin::Add, lo, half);
        let start = self.entry_dwords(mid, 0, 2);
        let start = self.wide(start[0], start[1]);
        let below = self.s.m.cmp(self.cur, pred::IULE, start, va);
        let next_lo = self.s.m.select(self.cur, below, mid, lo);
        let next_len = self.s.m.bin(self.cur, Bin::Sub, len, half);
        let tail = self.cur;
        self.s.m.terminate(tail, Term::Br(head));
        self.s.m.set_phi(lo, vec![(zero, pre), (next_lo, tail)]);
        self.s.m.set_phi(len, vec![(count, pre), (next_len, tail)]);
        self.cur = done;
        let e = self.entry_dwords(lo, 0, 4);
        let view_offset = self.entry_dwords(lo, 16, 1)[0];
        let c = self.cur;
        let start = self.wide(e[0], e[1]);
        let rel = self.s.m.bin(c, Bin::Sub, va, start);
        let size = self.widen(e[2], false);
        let some = self.s.m.cmp(c, pred::NE, count, zero);
        let after = self.s.m.cmp(c, pred::IUGE, va, start);
        let inside = self.s.m.cmp(c, pred::IULT, rel, size);
        let hit = self.s.m.bin(c, Bin::And, after, inside);
        let hit = self.s.m.bin(c, Bin::And, hit, some);
        let rel = self.s.m.cast(c, Cast::Trunc, rel, i32t);
        let byte = self.s.m.bin(c, Bin::Add, rel, view_offset);
        (hit, e[3], byte)
    }

    /// Dwords `first / 4..` (`n`, at most 4) of map entry `index`.
    fn entry_dwords(&mut self, index: V, first: u32, n: u32) -> Vec<V> {
        let c = self.cur;
        let size = self.s.m.i32(ENTRY);
        let at = self.s.m.bin(c, Bin::Mul, index, size);
        let base = self.s.m.i32(HEADER + first);
        let at = self.s.m.bin(c, Bin::Add, at, base);
        self.map_dwords(at, n)
    }

    /// `n` dwords of title memory from address `va` (an i64), zero where
    /// no range of the map covers `va`. `uniform`: the same address in
    /// every lane (a scalar load). The dwords are taken to lie in the range
    /// that holds the first.
    pub(super) fn global_read(&mut self, va: V, n: u32, uniform: bool) -> Vec<V> {
        let (hit, view, byte) = self.map_lookup(va);
        let at = self.begin_guarded(Some(hit));
        let h = self.s.heap_handle(self.cur, view, false, !uniform, RAW_UAV);
        let mut out = Vec::with_capacity(n as usize);
        let mut k = 0;
        while k < n {
            let chunk = (n - k).min(4);
            let off = self.s.m.i32(4 * k);
            let from = self.s.m.bin(self.cur, Bin::Add, byte, off);
            out.extend(self.raw_dwords(h, from, chunk));
            k += chunk;
        }
        let zero = self.s.m.i32(0);
        let pairs: Vec<(V, V)> = out.iter().map(|&x| (x, zero)).collect();
        self.end_guarded(at, &pairs)
    }

    /// `s_load_dword*` from an address the program computes.
    pub(super) fn global_smem(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Smem {
            base,
            data,
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar access to title memory"),
        };
        let (Operand::Scalar { code: b, .. }, Some(Operand::Scalar { code: d, .. })) =
            (*base, *data)
        else {
            return Err("a scalar load from title memory");
        };
        let lo = inst.read(Reg::S(b)).ok_or("an unread address")?;
        let hi = inst.read(Reg::S(b + 1)).ok_or("an unread address")?;
        let (lo, hi) = (self.i32_of(lo)?, self.i32_of(hi)?);
        let mut va = self.wide(lo, hi);
        let c = self.cur;
        let imm = self.s.m.i32(*offset as u32);
        let imm = self.widen(imm, true);
        va = self.s.m.bin(c, Bin::Add, va, imm);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, soffset)?;
            let so = self.widen(so, false);
            va = self.s.m.bin(self.cur, Bin::Add, va, so);
        }
        // Scalar loads ignore the address's two low bits.
        let t = self.s.m.int(64);
        let aligned = self.s.m.int_const(t, !3);
        va = self.s.m.bin(self.cur, Bin::And, va, aligned);
        let vals = self.global_read(va, count, true);
        for (k, x) in vals.into_iter().enumerate() {
            self.write(inst, Reg::S(d + k as u8), x, None);
        }
        Ok(())
    }

    /// The address a `global_*` instruction accesses: its SGPR base plus
    /// its VGPR offset, or its VGPR pair, plus its immediate (an i64).
    pub(super) fn global_address(&mut self, inst: &Inst) -> Result<V, Why> {
        let Fields::Flat {
            addr,
            saddr,
            offset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = *addr else {
            return Err("a global address not in VGPRs");
        };
        let vreg = |l: &mut Self, r: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(r)).ok_or("an unread address")?;
            l.i32_of(v)
        };
        let va = match *saddr {
            Some(Operand::Scalar { code, .. }) if code != NULL => {
                let lo = inst.read(Reg::S(code)).ok_or("an unread address")?;
                let hi = inst.read(Reg::S(code + 1)).ok_or("an unread address")?;
                let (lo, hi) = (self.i32_of(lo)?, self.i32_of(hi)?);
                let base = self.wide(lo, hi);
                let v = vreg(self, reg)?;
                let v = self.widen(v, false);
                self.s.m.bin(self.cur, Bin::Add, base, v)
            }
            _ => {
                let (lo, hi) = (vreg(self, reg)?, vreg(self, reg + 1)?);
                self.wide(lo, hi)
            }
        };
        let imm = self.s.m.i32(*offset as i32 as u32);
        let imm = self.widen(imm, true);
        Ok(self.s.m.bin(self.cur, Bin::Add, va, imm))
    }

    /// `global_load_*` from an address the program computes.
    pub(super) fn global_load(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, .. } = &inst.fields else {
            unreachable!()
        };
        let (n, sub) = match inst.name().strip_prefix("global_load_") {
            Some("dword") => (1, None),
            Some("dwordx2") => (2, None),
            Some("dwordx3") => (3, None),
            Some("dwordx4") => (4, None),
            Some("ubyte") => (1, Some((8, false))),
            Some("sbyte") => (1, Some((8, true))),
            Some("ushort") => (1, Some((16, false))),
            Some("sshort") => (1, Some((16, true))),
            _ => return Err("a global access to title memory"),
        };
        let Some(Operand::Vector { reg, .. }) = *dst else {
            return Err("a global load result");
        };
        let guard = self.guard(b, i)?;
        let va = self.global_address(inst)?;
        let vals = match sub {
            None => self.global_read(va, n, false),
            Some((width, signed)) => {
                // shortcut: the field is taken not to cross a dword.
                let t = self.s.m.int(64);
                let aligned = self.s.m.int_const(t, !3);
                let at = self.s.m.bin(self.cur, Bin::And, va, aligned);
                let w = self.global_read(at, 1, false)[0];
                let c = self.cur;
                let i32t = self.s.t.i32;
                let low = self.s.m.cast(c, Cast::Trunc, va, i32t);
                let three = self.s.m.i32(3);
                let byte = self.s.m.bin(c, Bin::And, low, three);
                let sh = self.s.m.bin(c, Bin::Shl, byte, three);
                let x = self.s.m.bin(c, Bin::LShr, w, sh);
                let left = self.s.m.i32(32 - width);
                let x = self.s.m.bin(c, Bin::Shl, x, left);
                let op = if signed { Bin::AShr } else { Bin::LShr };
                vec![self.s.m.bin(c, op, x, left)]
            }
        };
        for (k, x) in vals.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// The VGPR dwords `n` from `reg` that `inst` reads.
    fn data_dwords(&mut self, inst: &Inst, reg: u8, n: u32) -> Result<Vec<V>, Why> {
        (0..n)
            .map(|k| {
                let v = inst
                    .read(Reg::V(reg + k as u8))
                    .ok_or("an unread store operand")?;
                self.i32_of(v)
            })
            .collect()
    }

    /// `global_store_dword*` to an address the program computes. Lanes
    /// whose address no range covers store nothing.
    pub(super) fn global_store(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { data, .. } = &inst.fields else {
            unreachable!()
        };
        let n = match inst.name().strip_prefix("global_store_") {
            Some("dword") => 1,
            Some("dwordx2") => 2,
            Some("dwordx3") => 3,
            Some("dwordx4") => 4,
            _ => return Err("a byte or half store to title memory"),
        };
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a global store's data");
        };
        let vals = self.data_dwords(inst, reg, n)?;
        let guard = self.guard(b, i)?;
        let va = self.global_address(inst)?;
        let (hit, view, byte) = self.map_lookup(va);
        let go = match guard {
            Some(g) => self.s.m.bin(self.cur, Bin::And, hit, g),
            None => hit,
        };
        let at = self.begin_guarded(Some(go));
        let h = self.s.heap_handle(self.cur, view, false, true, RAW_UAV);
        let iu = self.s.m.undef(self.s.t.i32);
        let mut args = vec![h, byte, iu];
        args.extend(&vals);
        args.resize(3 + 4, iu);
        args.push(self.s.m.i8((1u8 << n) - 1));
        args.push(self.s.m.i32(4));
        let void = self.s.t.void;
        self.s.op(
            self.cur,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            void,
            140,
            &args,
        );
        // The map's views are of host copies too (`emit`, "Write marks").
        self.mark_view_bytes(h, byte, 4 * n, 1);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// `global_atomic_*` (32-bit) at an address the program computes. A
    /// lane whose address no range covers changes nothing and gets 0.
    pub(super) fn global_atomic(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { data, dst, .. } = &inst.fields else {
            unreachable!()
        };
        // DXIL's atomic operations; cmpswap is u32::MAX.
        let op = match inst.name().strip_prefix("global_atomic_") {
            Some("add") => 0,
            Some("and") => 1,
            Some("or") => 2,
            Some("xor") => 3,
            Some("smin") => 4,
            Some("smax") => 5,
            Some("umin") => 6,
            Some("umax") => 7,
            Some("swap") => 8,
            Some("sub") => u32::MAX - 1,
            Some("cmpswap") => u32::MAX,
            _ => return Err("a global atomic of that kind"),
        };
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a global atomic's data");
        };
        let vals = self.data_dwords(inst, reg, if op == u32::MAX { 2 } else { 1 })?;
        let guard = self.guard(b, i)?;
        let va = self.global_address(inst)?;
        let (hit, view, byte) = self.map_lookup(va);
        let go = match guard {
            Some(g) => self.s.m.bin(self.cur, Bin::And, hit, g),
            None => hit,
        };
        let at = self.begin_guarded(Some(go));
        let h = self.s.heap_handle(self.cur, view, false, true, RAW_UAV);
        let iu = self.s.m.undef(self.s.t.i32);
        let i32t = self.s.t.i32;
        let r = if op == u32::MAX {
            // Data is (new, compare).
            let args = [h, byte, iu, iu, vals[1], vals[0]];
            self.s.op(
                self.cur,
                "atomicCompareExchange.i32",
                Attr::NoUnwind,
                i32t,
                79,
                &args,
            )
        } else {
            let (op, x) = if op == u32::MAX - 1 {
                let zero = self.s.m.i32(0);
                (0, self.s.m.bin(self.cur, Bin::Sub, zero, vals[0]))
            } else {
                (op, vals[0])
            };
            let opv = self.s.m.i32(op);
            let args = [h, opv, byte, iu, iu, x];
            self.s
                .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
        };
        self.mark_view_bytes(h, byte, 4, 1);
        let zero = self.s.m.i32(0);
        let r = self.end_guarded(at, &[(r, zero)])[0];
        if let Some(Operand::Vector { reg, .. }) = *dst {
            self.write(inst, Reg::V(reg), r, guard);
        }
        Ok(())
    }

    /// Whether any instruction or phi of the program reads `v`.
    pub(super) fn value_read(&self, v: Value) -> bool {
        self.x.p.blocks.iter().any(|b| {
            b.phis.iter().any(|p| p.args.contains(&v))
                || b.insts.iter().any(|i| i.reads.iter().any(|r| r.1 == v))
        })
    }

    /// The console words of the `count` dwords the descriptor load at
    /// `site` reads, from the descriptor words the host binds: the load's
    /// address is the byte in the view heap as the title lays it out (32
    /// bytes a descriptor; a table's user data is the byte offset of its
    /// first descriptor, the heap's own base 0), which is the byte in the
    /// descriptor words as well. Computed in the load's own block, as
    /// [`Lower::heap_index`] computes the slot. With a heap guard, a load
    /// past the slots programs name reads zero (root descriptors are not
    /// bounds-checked; the host keeps two slots of room after the last
    /// for a load that runs over).
    pub(super) fn descriptor_words(&mut self, site: Site, count: u32) -> Result<Vec<V>, Why> {
        let a = *self
            .scalar
            .get(&site)
            .ok_or("an untracked descriptor load")?;
        let rp = *self
            .x
            .native
            .layout
            .get(a.param)
            .ok_or("a descriptor read as data from no table")?;
        let samplers = self
            .x
            .native
            .used
            .iter()
            .any(|u| u.slot == rp.slot && u.kind == ResourceType::Sampler);
        match rp.root {
            RootType::Table if !samplers => {}
            RootType::ResourceHeap => {}
            _ => return Err("a sampler descriptor read as data"),
        }
        let li: &Inst = &self.x.p.blocks[site.0].insts[site.1];
        let Fields::Smem {
            base: Operand::Scalar { code, .. },
            offset,
            soffset,
            ..
        } = &li.fields
        else {
            return Err("a descriptor load");
        };
        let base = li.read(Reg::S(*code)).ok_or("a descriptor load")?;
        let mut byte = self.i32_of(base)?;
        let k = self.s.m.i32(*offset as u32);
        byte = self.s.m.bin(self.cur, Bin::Add, byte, k);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(li, soffset)?;
            byte = self.s.m.bin(self.cur, Bin::Add, byte, so);
        }
        let inside = crate::emit::view_heap_guard().map(|g| {
            let c = self.cur;
            let (five, limit) = (self.s.m.i32(5), self.s.m.i32(g.limit));
            let slot = self.s.m.bin(c, Bin::LShr, byte, five);
            self.s.m.cmp(c, pred::IULT, slot, limit)
        });
        if let Some(inside) = inside {
            let zero = self.s.m.i32(0);
            byte = self.s.m.select(self.cur, inside, byte, zero);
        }
        let h = self.s.descriptor_words_handle();
        let mut out = Vec::with_capacity(count as usize);
        let mut k = 0;
        while k < count {
            let chunk = (count - k).min(4);
            let four = self.s.m.i32(4 * k);
            let at = self.s.m.bin(self.cur, Bin::Add, byte, four);
            out.extend(self.raw_dwords(h, at, chunk));
            k += chunk;
        }
        if let Some(inside) = inside {
            let zero = self.s.m.i32(0);
            for x in &mut out {
                *x = self.s.m.select(self.cur, inside, *x, zero);
            }
        }
        Ok(out)
    }
}
