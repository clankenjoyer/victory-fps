//! The console's ray intersection instructions (`image_bvh_intersect_ray`,
//! `image_bvh64_intersect_ray`), which no host shader model has: the
//! program walks the BVH itself and asks the hardware to test one node.
//! The node is read from title memory through the address map
//! ([`super::global`]) and tested here, as Microsoft's software model of
//! the instruction does (`SoftwareBvhIntersectRay` in the XDXR traversal
//! headers): a box node gives its four children ordered by distance, a
//! leaf gives the hit's distance as a fraction.
//!
//! Node pointer: the type in bits 0-2 (0-3 the triangles of a leaf, 4 a
//! box with half-float bounds, 5 one with float bounds; the model treats
//! 6 and 7 as float boxes too), and the node's address over 64 in the
//! rest, added to the descriptor's base. Descriptor: base address over
//! 256 in bits 0-39, box sorting at bit 63, the triangle return mode at
//! bit 120 (1: the barycentric numerators, 0: the leaf's id dword and a
//! hit flag).
//!
//! Layouts (`XboxBvh4Node16`, `XboxBvh4Node32`, `XboxBvh4Leaf`): four
//! child pointers, then four boxes of six halves (16) or floats (32); a
//! leaf holds five vertices (TRI0 uses 0, 1, 2; TRI1 1, 3, 2; TRI2 2, 3,
//! 4; TRI3 2, 4, 0) and in its last dword two bits of barycentric remap
//! per triangle.
//!
//! shortcut: the triangle test's products are not marked precise; a host
//! compiler that fuses them into FMAs changes results on edges, where the
//! test is meant to be watertight.

use super::*;

/// What a node of each kind gives back when the ray misses it.
const INVALID: u32 = 0xFFFF_FFFF;
const INF: u32 = 0x7F80_0000;
const ONE: u32 = 0x3F80_0000;
const FLT_MAX: u32 = 0x7F7F_FFFF;
/// Dwords read per node: a float box node's children and bounds.
const NODE_DWORDS: u32 = 28;

impl Lower<'_> {
    pub(super) fn bvh_map(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            rsrc,
            a16,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let Operand::Vector { reg: out, .. } = *data else {
            return Err("a ray intersection result");
        };
        let Operand::Scalar { code: desc, .. } = *rsrc else {
            return Err("a BVH descriptor not in SGPRs");
        };
        let wide = inst.name() == "image_bvh64_intersect_ray";
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        let guard = self.guard(b, i)?;
        let dword = |l: &mut Self, k: usize| -> Result<V, Why> {
            let r = *regs.get(k).ok_or("a ray intersection address")?;
            let v = inst.read(Reg::V(r)).ok_or("a ray intersection address")?;
            l.i32_of(v)
        };
        let mut k = 0;
        let mut next = |l: &mut Self| {
            k += 1;
            dword(l, k - 1)
        };
        let node_lo = next(self)?;
        let node_hi = if wide { Some(next(self)?) } else { None };
        let extent = next(self)?;
        let extent = self.as_float(extent);
        let mut origin = [self.s.m.i32(0); 3];
        for o in &mut origin {
            let x = next(self)?;
            *o = self.as_float(x);
        }
        let (dir, inv) = if *a16 {
            // Halves, low first: dir x y, dir z and inverse x, inverse y z.
            let w = [next(self)?, next(self)?, next(self)?];
            let h: Vec<V> = (0..6).map(|n| self.half(w[n / 2], n % 2 == 1)).collect();
            ([h[0], h[1], h[2]], [h[3], h[4], h[5]])
        } else {
            let mut d = [self.s.m.i32(0); 6];
            for x in &mut d {
                let y = next(self)?;
                *x = self.as_float(y);
            }
            ([d[0], d[1], d[2]], [d[3], d[4], d[5]])
        };
        let mut words = [self.s.m.i32(0); 4];
        for (k, w) in words.iter_mut().enumerate() {
            let v = inst
                .read(Reg::S(desc + k as u8))
                .ok_or("a BVH descriptor")?;
            *w = self.i32_of(v)?;
        }

        let c = self.cur;
        // The node's address: the descriptor's base plus the pointer's
        // index of 64-byte units.
        let ff = self.s.m.i32(0xFF);
        let base_hi = self.s.m.bin(c, Bin::And, words[1], ff);
        let base = self.wide(words[0], base_hi);
        let t = self.s.m.int(64);
        let eight = self.s.m.int_const(t, 8);
        let base = self.s.m.bin(c, Bin::Shl, base, eight);
        let node = match node_hi {
            Some(hi) => self.wide(node_lo, hi),
            None => self.widen(node_lo, false),
        };
        let three = self.s.m.int_const(t, 3);
        let six = self.s.m.int_const(t, 6);
        let units = self.s.m.bin(c, Bin::LShr, node, three);
        let offset = self.s.m.bin(c, Bin::Shl, units, six);
        let va = self.s.m.bin(c, Bin::Add, base, offset);
        let seven = self.s.m.i32(7);
        let kind = self.s.m.bin(c, Bin::And, node_lo, seven);
        let n = self.global_read(va, NODE_DWORDS, false);

        let boxes = self.box_node(&n, kind, extent, origin, dir, inv, words[1]);
        let leaf = self.leaf_node(&n, kind, origin, dir, words[3]);
        let c = self.cur;
        let three = self.s.m.i32(3);
        let is_leaf = self.s.m.cmp(c, pred::IULE, kind, three);
        for k in 0..4 {
            let x = self.s.m.select(c, is_leaf, leaf[k], boxes[k]);
            self.write(inst, Reg::V(out + k as u8), x, guard);
        }
        Ok(())
    }

    /// Half `hi` or the low half of `w`, as a float.
    fn half(&mut self, w: V, hi: bool) -> V {
        let c = self.cur;
        let h = if hi {
            let sh = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, w, sh)
        } else {
            w
        };
        let mask = self.s.m.i32(0xFFFF);
        let h = self.s.m.bin(c, Bin::And, h, mask);
        self.s
            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h])
    }

    fn fconst(&mut self, bits: u32) -> V {
        self.s.m.f32(bits)
    }

    fn fmin(&mut self, a: V, b: V) -> V {
        self.binary(36, a, b)
    }

    fn fmax(&mut self, a: V, b: V) -> V {
        self.binary(35, a, b)
    }

    fn fop(&mut self, op: Bin, a: V, b: V) -> V {
        self.s.m.bin(self.cur, op, a, b)
    }

    fn fcmp(&mut self, p: u8, a: V, b: V) -> V {
        self.s.m.cmp(self.cur, p, a, b)
    }

    fn or(&mut self, a: V, b: V) -> V {
        self.s.m.bin(self.cur, Bin::Or, a, b)
    }

    fn and(&mut self, a: V, b: V) -> V {
        self.s.m.bin(self.cur, Bin::And, a, b)
    }

    fn not(&mut self, a: V) -> V {
        let t = self.s.m.bool(true);
        self.s.m.bin(self.cur, Bin::Xor, a, t)
    }

    /// `XdxrDecompressBox16`'s clamp of a widened half to the finite range.
    fn finite(&mut self, x: V) -> V {
        let (max, min) = (self.fconst(FLT_MAX), self.fconst(FLT_MAX | 0x8000_0000));
        let x = self.fmin(x, max);
        self.fmax(x, min)
    }

    /// A box node's result: the four child pointers whose boxes the ray
    /// meets within `extent`, the others invalid, ordered by entry
    /// distance when the descriptor asks (`TestXboxBvh4Node`).
    #[allow(clippy::too_many_arguments)]
    fn box_node(
        &mut self,
        n: &[V],
        kind: V,
        extent: V,
        origin: [V; 3],
        dir: [V; 3],
        inv: [V; 3],
        desc1: V,
    ) -> [V; 4] {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let half_box = self.s.m.cmp(c, pred::EQ, kind, four);
        let mut dist = [self.fconst(0); 4];
        let mut ptrs = [self.s.m.i32(0); 4];
        for k in 0..4 {
            // Float bounds: six floats from dword 4 + 6k.
            let full: Vec<V> = (0..6).map(|j| self.as_float(n[4 + 6 * k + j])).collect();
            // Half bounds: three dwords from 4 + 3k, min x y, min z max x,
            // max y z.
            let w = [n[4 + 3 * k], n[5 + 3 * k], n[6 + 3 * k]];
            let small: Vec<V> = (0..6)
                .map(|j| {
                    let h = self.half(w[j / 2], j % 2 == 1);
                    self.finite(h)
                })
                .collect();
            let c = self.cur;
            let b: Vec<V> = (0..6)
                .map(|j| self.s.m.select(c, half_box, small[j], full[j]))
                .collect();
            let t = self.ray_box(
                extent,
                origin,
                dir,
                inv,
                [b[0], b[1], b[2]],
                [b[3], b[4], b[5]],
            );
            let c = self.cur;
            let within = self.s.m.cmp(c, pred::OLE, t, extent);
            let invalid = self.s.m.i32(INVALID);
            ptrs[k] = self.s.m.select(c, within, n[k], invalid);
            dist[k] = t;
        }
        let c = self.cur;
        let one = self.s.m.i32(1);
        let shifted = self.s.m.i32(31);
        let sort = self.s.m.bin(c, Bin::LShr, desc1, shifted);
        let sort = self.s.m.bin(c, Bin::And, sort, one);
        let nil = self.s.m.i32(0);
        let sort = self.s.m.cmp(c, pred::NE, sort, nil);
        let (mut d, mut p) = (dist, ptrs);
        for (x, y) in [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)] {
            let c = self.cur;
            let later = self.s.m.cmp(c, pred::OGT, d[x], d[y]);
            let swap = self.s.m.bin(c, Bin::And, later, sort);
            let (dx, dy) = (
                self.s.m.select(c, swap, d[y], d[x]),
                self.s.m.select(c, swap, d[x], d[y]),
            );
            let (px, py) = (
                self.s.m.select(c, swap, p[y], p[x]),
                self.s.m.select(c, swap, p[x], p[y]),
            );
            (d[x], d[y], p[x], p[y]) = (dx, dy, px, py);
        }
        p
    }

    /// The distance at which the ray enters the box, within
    /// `[0, extent]`, or infinity (`RayBoxTest`).
    fn ray_box(
        &mut self,
        extent: V,
        origin: [V; 3],
        dir: [V; 3],
        inv: [V; 3],
        min: [V; 3],
        max: [V; 3],
    ) -> V {
        let mut miss = self.s.m.bool(false);
        for a in 0..3 {
            let flipped = self.fcmp(pred::OGT, min[a], max[a]);
            miss = self.or(miss, flipped);
        }
        let mut near = self.fconst(0);
        let mut far = extent;
        for a in 0..3 {
            let zero = self.fconst(0);
            let flat = self.fcmp(pred::OEQ, dir[a], zero);
            let below = self.fcmp(pred::OLT, origin[a], min[a]);
            let above = self.fcmp(pred::OGT, origin[a], max[a]);
            let outside = self.or(below, above);
            let out = self.and(flat, outside);
            miss = self.or(miss, out);
            let d1 = self.fop(Bin::Sub, min[a], origin[a]);
            let t1 = self.fop(Bin::Mul, d1, inv[a]);
            let d2 = self.fop(Bin::Sub, max[a], origin[a]);
            let t2 = self.fop(Bin::Mul, d2, inv[a]);
            let lo = self.fmin(t1, t2);
            let hi = self.fmax(t1, t2);
            let n = self.fmax(near, lo);
            let f = self.fmin(far, hi);
            let c = self.cur;
            near = self.s.m.select(c, flat, near, n);
            far = self.s.m.select(c, flat, far, f);
        }
        let crossed = self.fcmp(pred::OLT, far, near);
        miss = self.or(miss, crossed);
        let inf = self.fconst(INF);
        self.s.m.select(self.cur, miss, inf, near)
    }

    /// A leaf's result for triangle `kind` (`TestXboxBvh4Leaf`): the hit
    /// distance's numerator and denominator, then the barycentric
    /// numerators (return mode 1, descriptor bit 120) or the leaf's id
    /// dword and a hit flag (mode 0); a miss gives infinity over one.
    fn leaf_node(&mut self, n: &[V], kind: V, origin: [V; 3], dir: [V; 3], desc3: V) -> [V; 4] {
        let c = self.cur;
        let verts: Vec<[V; 3]> = (0..5)
            .map(|v| {
                [
                    self.as_float(n[3 * v]),
                    self.as_float(n[3 * v + 1]),
                    self.as_float(n[3 * v + 2]),
                ]
            })
            .collect();
        let is = |l: &mut Self, k: u32| {
            let kk = l.s.m.i32(k);
            l.s.m.cmp(c, pred::EQ, kind, kk)
        };
        let (t0, t1, t2) = (is(self, 0), is(self, 1), is(self, 2));
        let pick3 = |l: &mut Self, sel: &[(V, usize)], last: usize| -> [V; 3] {
            let mut out = verts[last];
            for &(cond, v) in sel.iter().rev() {
                for j in 0..3 {
                    out[j] = l.s.m.select(c, cond, verts[v][j], out[j]);
                }
            }
            out
        };
        // TRI0 (0, 1, 2), TRI1 (1, 3, 2), TRI2 (2, 3, 4), TRI3 (2, 4, 0).
        let mut va = pick3(self, &[(t0, 0), (t1, 1)], 2);
        let t12 = self.or(t1, t2);
        let mut vb = pick3(self, &[(t0, 1), (t12, 3)], 4);
        let t01 = self.or(t0, t1);
        let mut vc = pick3(self, &[(t01, 2), (t2, 4)], 0);
        let eight = self.s.m.i32(8);
        let sh = self.s.m.bin(c, Bin::Mul, kind, eight);
        let remap = self.s.m.bin(c, Bin::LShr, n[15], sh);

        // The axis the ray moves most along becomes z.
        let mag: Vec<V> = dir.iter().map(|&d| self.unary(6, d)).collect();
        let m = self.fmax(mag[1], mag[2]);
        let m = self.fmax(mag[0], m);
        let on_x = self.fcmp(pred::OEQ, mag[0], m);
        let on_y = self.fcmp(pred::OEQ, mag[1], m);
        let not_x = self.not(on_x);
        let on_y = self.and(not_x, on_y);
        let c = self.cur;
        let swizzle = |l: &mut Self, v: [V; 3]| -> [V; 3] {
            // On x: (y, z, x); on y: (z, x, y).
            let yzx = [v[1], v[2], v[0]];
            let zxy = [v[2], v[0], v[1]];
            let mut out = v;
            for j in 0..3 {
                out[j] = l.s.m.select(c, on_y, zxy[j], out[j]);
                out[j] = l.s.m.select(c, on_x, yzx[j], out[j]);
            }
            out
        };
        let d = swizzle(self, dir);
        let o = swizzle(self, origin);
        va = swizzle(self, va);
        vb = swizzle(self, vb);
        vc = swizzle(self, vc);
        let rel = |l: &mut Self, v: [V; 3]| -> [V; 3] {
            [
                l.fop(Bin::Sub, v[0], o[0]),
                l.fop(Bin::Sub, v[1], o[1]),
                l.fop(Bin::Sub, v[2], o[2]),
            ]
        };
        let (va, vb, vc) = (rel(self, va), rel(self, vb), rel(self, vc));
        let shear = |l: &mut Self, v: [V; 3]| -> [V; 3] {
            let x1 = l.fop(Bin::Mul, v[0], d[2]);
            let x2 = l.fop(Bin::Mul, d[0], v[2]);
            let y1 = l.fop(Bin::Mul, v[1], d[2]);
            let y2 = l.fop(Bin::Mul, d[1], v[2]);
            [l.fop(Bin::Sub, x1, x2), l.fop(Bin::Sub, y1, y2), v[2]]
        };
        let (a, bb, cc) = (shear(self, va), shear(self, vb), shear(self, vc));
        let cross = |l: &mut Self, p: [V; 3], q: [V; 3]| {
            let m1 = l.fop(Bin::Mul, p[0], q[1]);
            let m2 = l.fop(Bin::Mul, p[1], q[0]);
            l.fop(Bin::Sub, m1, m2)
        };
        let u = cross(self, cc, bb);
        let v = cross(self, a, cc);
        let w = cross(self, bb, a);
        let dot = |l: &mut Self, x: [V; 3]| {
            let p = l.fop(Bin::Mul, u, x[0]);
            let q = l.fop(Bin::Mul, v, x[1]);
            let r = l.fop(Bin::Mul, w, x[2]);
            let pq = l.fop(Bin::Add, p, q);
            l.fop(Bin::Add, pq, r)
        };
        let t_num = dot(self, [a[2], bb[2], cc[2]]);
        let t_den = dot(self, [d[2], d[2], d[2]]);

        let zero = self.fconst(0);
        let sign = |l: &mut Self, x: V, p: u8| l.fcmp(p, x, zero);
        let (un, vn, wn) = (
            sign(self, u, pred::OLT),
            sign(self, v, pred::OLT),
            sign(self, w, pred::OLT),
        );
        let (up, vp, wp) = (
            sign(self, u, pred::OGT),
            sign(self, v, pred::OGT),
            sign(self, w, pred::OGT),
        );
        let any_neg = self.or(un, vn);
        let any_neg = self.or(any_neg, wn);
        let any_pos = self.or(up, vp);
        let any_pos = self.or(any_pos, wp);
        let mut missed = self.and(any_neg, any_pos);
        let den_abs = self.unary(6, t_den);
        let flat = self.fcmp(pred::OEQ, den_abs, zero);
        missed = self.or(missed, flat);
        let c = self.cur;
        let den_bits = self.as_int(t_den);
        let sign_bit = self.s.m.i32(0x8000_0000);
        let den_sign = self.s.m.bin(c, Bin::And, den_bits, sign_bit);
        let one_bits = self.s.m.i32(ONE);
        let unit = self.s.m.bin(c, Bin::Or, den_sign, one_bits);
        let unit = self.as_float(unit);
        let ahead = self.fop(Bin::Mul, t_num, unit);
        let behind = self.fcmp(pred::OLE, ahead, zero);
        missed = self.or(missed, behind);
        let nan = self.fcmp(pred::UNO, t_num, t_num);
        missed = self.or(missed, nan);
        let winding = self.fcmp(pred::OGT, t_den, zero);
        // `BreakTieOnEdge`: which of two triangles sharing an edge owns it.
        let tie = |l: &mut Self, ay: V, by: V, cy: V| -> V {
            let (ba, ca) = (l.unary(6, by), l.unary(6, cy));
            let b0 = l.fcmp(pred::OEQ, ba, zero);
            let c0 = l.fcmp(pred::OEQ, ca, zero);
            let both = l.and(b0, c0);
            let bottom = l.fcmp(pred::OGT, ay, zero);
            let b_neg = l.fcmp(pred::OLT, by, zero);
            let c_pos = l.fcmp(pred::OGT, cy, zero);
            let b0c = l.and(b0, c_pos);
            let below = l.or(b_neg, b0c);
            let right = l.s.m.bin(l.cur, Bin::Xor, below, winding);
            l.s.m.select(l.cur, both, bottom, right)
        };
        for (e, (p, q, r)) in [(u, (a, bb, cc)), (v, (bb, cc, a)), (w, (cc, a, bb))] {
            let ea = self.unary(6, e);
            let on = self.fcmp(pred::OEQ, ea, zero);
            let lost = tie(self, p[1], q[1], r[1]);
            let lost = self.and(on, lost);
            missed = self.or(missed, lost);
        }

        let c = self.cur;
        let pick = |l: &mut Self, sel: V| -> V {
            let (one, two) = (l.s.m.i32(1), l.s.m.i32(2));
            let is1 = l.s.m.cmp(c, pred::EQ, sel, one);
            let is2 = l.s.m.cmp(c, pred::EQ, sel, two);
            let x = l.s.m.select(c, is1, v, u);
            l.s.m.select(c, is2, w, x)
        };
        let three = self.s.m.i32(3);
        let i_sel = self.s.m.bin(c, Bin::And, remap, three);
        let two = self.s.m.i32(2);
        let j_sel = self.s.m.bin(c, Bin::LShr, remap, two);
        let j_sel = self.s.m.bin(c, Bin::And, j_sel, three);
        let i_num = pick(self, i_sel);
        let j_num = pick(self, j_sel);
        let i_neg = self.fcmp(pred::OLT, i_num, zero);
        let i_pos = self.fcmp(pred::OGT, i_num, zero);
        let den_neg = self.fcmp(pred::OLT, t_den, zero);
        let f1 = self.and(i_neg, winding);
        let f2 = self.and(i_pos, den_neg);
        let flip = self.or(f1, f2);
        let c = self.cur;
        let negate = |l: &mut Self, x: V| {
            let bits = l.as_int(x);
            let s = l.s.m.i32(0x8000_0000);
            let bits = l.s.m.bin(c, Bin::Xor, bits, s);
            let flipped = l.as_float(bits);
            l.s.m.select(c, flip, flipped, x)
        };
        let i_num = negate(self, i_num);
        let j_num = negate(self, j_num);

        let c = self.cur;
        let shift = self.s.m.i32(24);
        let mode = self.s.m.bin(c, Bin::LShr, desc3, shift);
        let one = self.s.m.i32(1);
        let mode = self.s.m.bin(c, Bin::And, mode, one);
        let nil = self.s.m.i32(0);
        let ij = self.s.m.cmp(c, pred::NE, mode, nil);
        let (tn, td, ii, jj) = (
            self.as_int(t_num),
            self.as_int(t_den),
            self.as_int(i_num),
            self.as_int(j_num),
        );
        let c = self.cur;
        // shortcut: in mode 0 the id dword is the leaf's last (the remap
        // bytes) and the flag 1; what the hardware returns there is not
        // modelled by the XDXR headers, which always ask for mode 1.
        let z = self.s.m.select(c, ij, ii, n[15]);
        let w_out = self.s.m.select(c, ij, jj, one);
        let (inf, unit, nil) = (self.s.m.i32(INF), self.s.m.i32(ONE), self.s.m.i32(0));
        [
            self.s.m.select(c, missed, inf, tn),
            self.s.m.select(c, missed, unit, td),
            self.s.m.select(c, missed, nil, z),
            self.s.m.select(c, missed, nil, w_out),
        ]
    }
}
