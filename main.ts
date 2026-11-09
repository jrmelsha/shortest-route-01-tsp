// shortest-route-01: Euclidean TSP (TSPLIB EUC_2D) by 2-opt + Or-opt local search with neighbour lists and a
// work queue (don't-look bits), then iterated local search (Martin, Otto & Felten style ILS) with localized
// segment-swap (double-bridge) kicks, undone through a reversal journal when they do not pay.
// Deterministic: every search is bounded by iteration counts and seeded with constants; nothing reads the clock.
//
// Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output.

import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// ILS kicks per city; the 35,000 cities of one run share the 120 s budget.
const KICKS_PER_CITY = 50;
const K = 8; // nearest candidate neighbours per city (plus quadrant neighbours)

function readInstance(path: string): { xs: Float64Array; ys: Float64Array } {
  const lines = readFileSync(path, 'utf8').trim().split('\n');
  const n = Number(lines[0]);
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const line = lines[k + 1]!;
    const sp = line.indexOf(' ');
    xs[k] = Number(line.slice(0, sp));
    ys[k] = Number(line.slice(sp + 1));
  }
  return { xs, ys };
}

function solve(xs: Float64Array, ys: Float64Array, kicks: number, seed: number): Int32Array {
  const n = xs.length;
  if (n <= 3) return Int32Array.from({ length: n }, (_, i) => i);

  const dist = (a: number, b: number): number => {
    const dx = xs[a]! - xs[b]!;
    const dy = ys[a]! - ys[b]!;
    return Math.round(Math.sqrt(dx * dx + dy * dy));
  };

  // Candidate lists by brute force (n <= 10,000): the K nearest plus the Q nearest in each of the four
  // quadrants around the city (quadrant neighbours reach across the gaps between clusters), merged and
  // sorted nearest first, at most kk per city.
  const Q = 2;
  const kk = K + 4 * Q;
  const nbr = new Int32Array(n * kk);
  const ncnt = new Int32Array(n);
  {
    // slot 0 holds the K nearest overall, slots 1..4 the Q nearest per quadrant
    const cap = [K, Q, Q, Q, Q];
    const bd = [new Float64Array(K), new Float64Array(Q), new Float64Array(Q), new Float64Array(Q), new Float64Array(Q)];
    const bi = [new Int32Array(K), new Int32Array(Q), new Int32Array(Q), new Int32Array(Q), new Int32Array(Q)];
    const cnt = new Int32Array(5);
    const insert = (s: number, d: number, j: number): void => {
      const D = bd[s]!;
      const I = bi[s]!;
      const c = cap[s]!;
      let p: number;
      if (cnt[s]! < c) p = cnt[s]!++;
      else if (d < D[c - 1]!) p = c - 1;
      else return;
      while (p > 0 && D[p - 1]! > d) {
        D[p] = D[p - 1]!;
        I[p] = I[p - 1]!;
        p--;
      }
      D[p] = d;
      I[p] = j;
    };
    const md = new Float64Array(kk);
    const mi = new Int32Array(kk);
    for (let i = 0; i < n; i++) {
      cnt.fill(0);
      const xi = xs[i]!;
      const yi = ys[i]!;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dx = xs[j]! - xi;
        const dy = ys[j]! - yi;
        const d = dx * dx + dy * dy;
        if (cnt[0]! < K || d < bd[0]![K - 1]!) insert(0, d, j);
        const q = 1 + (dx >= 0 ? 0 : 1) + (dy >= 0 ? 0 : 2);
        if (cnt[q]! < Q || d < bd[q]![Q - 1]!) insert(q, d, j);
      }
      // merge, dropping duplicates, nearest first
      let m = 0;
      for (let s = 0; s < 5; s++) {
        for (let t = 0; t < cnt[s]!; t++) {
          const j = bi[s]![t]!;
          const d = bd[s]![t]!;
          let dup = false;
          for (let u = 0; u < m; u++) if (mi[u] === j) dup = true;
          if (dup) continue;
          let p = m++;
          while (p > 0 && md[p - 1]! > d) {
            md[p] = md[p - 1]!;
            mi[p] = mi[p - 1]!;
            p--;
          }
          md[p] = d;
          mi[p] = j;
        }
      }
      ncnt[i] = m;
      for (let k = 0; k < m; k++) nbr[i * kk + k] = mi[k]!;
    }
  }

  // Start: nearest neighbour from city 0.
  const tour = new Int32Array(n);
  const pos = new Int32Array(n);
  {
    const visited = new Uint8Array(n);
    let at = 0;
    visited[0] = 1;
    for (let step = 1; step < n; step++) {
      let best = -1;
      // try the candidate list first
      let bestD = Infinity;
      for (let k = 0; k < ncnt[at]!; k++) {
        const c = nbr[at * kk + k]!;
        if (visited[c] === 0) {
          best = c;
          break;
        }
      }
      if (best < 0) {
        const xa = xs[at]!;
        const ya = ys[at]!;
        for (let c = 0; c < n; c++) {
          if (visited[c] === 1) continue;
          const dx = xs[c]! - xa;
          const dy = ys[c]! - ya;
          const d = dx * dx + dy * dy;
          if (d < bestD) {
            bestD = d;
            best = c;
          }
        }
      }
      tour[step] = best;
      visited[best] = 1;
      at = best;
    }
    for (let i = 0; i < n; i++) pos[tour[i]!] = i;
  }

  const succ = (c: number): number => {
    const p = pos[c]! + 1;
    return tour[p === n ? 0 : p]!;
  };
  const pred = (c: number): number => {
    const p = pos[c]! - 1;
    return tour[p < 0 ? n - 1 : p]!;
  };

  // Reversal journal (pairs of positions) so a rejected kick can be undone exactly.
  let journal = new Int32Array(1 << 16);
  let jlen = 0;
  let journaling = false;

  const doRev = (i: number, j: number): void => {
    let len = j - i;
    if (len < 0) len += n;
    len += 1;
    for (let s = len >> 1; s > 0; s--) {
      const a = tour[i]!;
      const b = tour[j]!;
      tour[i] = b;
      pos[b] = i;
      tour[j] = a;
      pos[a] = j;
      i++;
      if (i === n) i = 0;
      j--;
      if (j < 0) j = n - 1;
    }
  };

  // Reverse the forward path of positions i..j, or its complement when that is shorter (same cycle).
  const revPath = (i: number, j: number): void => {
    let len = j - i;
    if (len < 0) len += n;
    len += 1;
    if (len * 2 > n) {
      const ni = j + 1 === n ? 0 : j + 1;
      const nj = i === 0 ? n - 1 : i - 1;
      i = ni;
      j = nj;
    }
    if (journaling) {
      if (jlen + 2 > journal.length) {
        const bigger = new Int32Array(journal.length * 2);
        bigger.set(journal);
        journal = bigger;
      }
      journal[jlen++] = i;
      journal[jlen++] = j;
    }
    doRev(i, j);
  };

  // 2-opt move: tour edges (a,b) and (c,d), both forward or both backward, become (a,c) and (b,d).
  const make2opt = (a: number, b: number, c: number, d: number): void => {
    if (succ(a) === b) revPath(pos[b]!, pos[c]!);
    else revPath(pos[a]!, pos[d]!);
  };

  // Work queue of cities to examine (don't-look bits).
  const queue = new Int32Array(n);
  const inQ = new Uint8Array(n);
  let qh = 0;
  let qt = 0;
  let qsize = 0;
  const push = (c: number): void => {
    if (inQ[c] === 1) return;
    inQ[c] = 1;
    queue[qt] = c;
    qt++;
    if (qt === n) qt = 0;
    qsize++;
  };
  const pop = (): number => {
    const c = queue[qh]!;
    qh++;
    if (qh === n) qh = 0;
    qsize--;
    inQ[c] = 0;
    return c;
  };

  // One improving move from city a, applied; returns its (negative) delta or 0.
  const improveCity = (a: number): number => {
    for (let dir = 0; dir < 2; dir++) {
      const fwd = dir === 0;
      // 2-opt
      const b = fwd ? succ(a) : pred(a);
      const dab = dist(a, b);
      for (let k = 0, ka = ncnt[a]!; k < ka; k++) {
        const c = nbr[a * kk + k]!;
        const dac = dist(a, c);
        if (dac >= dab) break;
        const d = fwd ? succ(c) : pred(c);
        if (c === b || d === a) continue;
        const delta = dac + dist(b, d) - dab - dist(c, d);
        if (delta < 0) {
          make2opt(a, b, c, d);
          push(a);
          push(b);
          push(c);
          push(d);
          return delta;
        }
      }
      // Or-opt: move the segment s1=a .. s2 (1 to 3 cities, in direction dir) elsewhere, either way round.
      let s2 = a;
      let mid = a;
      for (let L = 1; L <= 3; L++) {
        if (L > 1) {
          mid = L === 3 ? s2 : a;
          s2 = fwd ? succ(s2) : pred(s2);
        }
        const s1 = a;
        const p = fwd ? pred(s1) : succ(s1);
        const nx = fwd ? succ(s2) : pred(s2);
        if (p === s2 || nx === s1 || p === nx) break;
        const G = dist(p, s1) + dist(s2, nx) - dist(p, nx);
        if (G <= 0) continue;
        for (let e = 0; e < 2; e++) {
          const sEnd = e === 0 ? s1 : s2;
          for (let k = 0, ke = ncnt[sEnd]!; k < ke; k++) {
            const c = nbr[sEnd * kk + k]!;
            const dsc = dist(sEnd, c);
            if (dsc >= G) break;
            if (c === s1 || c === s2 || c === mid) continue;
            for (let side = 0; side < 2; side++) {
              // insert between x and y = next(x) (orientation of dir)
              const x = side === 0 ? c : fwd ? pred(c) : succ(c);
              const y = side === 0 ? (fwd ? succ(c) : pred(c)) : c;
              if (x === s1 || x === s2 || x === mid || y === s1 || y === s2 || y === mid) continue;
              const dxy = dist(x, y);
              // reversed insertion: x-s2 ... s1-y ; straight: x-s1 ... s2-y
              const addRev = dist(x, s2) + dist(s1, y) - dxy;
              const addStr = dist(x, s1) + dist(s2, y) - dxy;
              const straight = addStr < addRev;
              const add = straight ? addStr : addRev;
              const delta = add - G;
              if (delta < 0) {
                make2opt(p, s1, x, y);
                if (x !== nx) make2opt(p, x, nx, s2);
                if (straight) make2opt(x, s2, s1, y);
                push(p);
                push(nx);
                push(s1);
                push(s2);
                push(x);
                push(y);
                return delta;
              }
            }
          }
        }
      }
    }
    return 0;
  };

  const localSearch = (): number => {
    let total = 0;
    while (qsize > 0) total += improveCity(pop());
    return total;
  };

  let rng = seed >>> 0 || 0x9e3779b9;
  const rand = (): number => {
    let x = rng;
    x ^= x << 13;
    x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    rng = x;
    return x;
  };

  for (let i = 0; i < n; i++) push(tour[i]!);
  localSearch();

  // ILS: kick = swap two adjacent short segments B and C (three reversals), re-optimise around it,
  // keep it when the tour got no longer, else undo every reversal in reverse order.
  journaling = true;
  const lmax = Math.max(3, Math.min(50, (n / 8) | 0));
  for (let it = 0; it < kicks; it++) {
    jlen = 0;
    const i = rand() % n;
    const l1 = 1 + (rand() % lmax);
    const l2 = 1 + (rand() % lmax);
    const ib1 = (i + 1) % n;
    const ib2 = (i + l1) % n;
    const ic1 = (i + l1 + 1) % n;
    const ic2 = (i + l1 + l2) % n;
    const ie = (i + l1 + l2 + 1) % n;
    const a = tour[i]!;
    const b1 = tour[ib1]!;
    const b2 = tour[ib2]!;
    const c1 = tour[ic1]!;
    const c2 = tour[ic2]!;
    const e = tour[ie]!;
    if (e === a || l1 + l2 + 2 > n) continue;
    const kickDelta = dist(a, c1) + dist(c2, b1) + dist(b2, e) - dist(a, b1) - dist(b2, c1) - dist(c2, e);
    // B C -> rev(C) rev(B) -> C B
    journal[jlen++] = ib1;
    journal[jlen++] = ic2;
    doRev(ib1, ic2);
    const m = (i + l2) % n;
    journal[jlen++] = ib1;
    journal[jlen++] = m;
    doRev(ib1, m);
    const m1 = (m + 1) % n;
    journal[jlen++] = m1;
    journal[jlen++] = ic2;
    doRev(m1, ic2);
    push(a);
    push(b1);
    push(b2);
    push(c1);
    push(c2);
    push(e);
    const delta = kickDelta + localSearch();
    if (delta > 0) {
      for (let q = jlen - 2; q >= 0; q -= 2) doRev(journal[q]!, journal[q + 1]!);
    }
  }
  return tour;
}

function tourLength(xs: Float64Array, ys: Float64Array, tour: Int32Array): number {
  let s = 0;
  const n = tour.length;
  for (let i = 0; i < n; i++) {
    const a = tour[i]!;
    const b = tour[i + 1 === n ? 0 : i + 1]!;
    const dx = xs[a]! - xs[b]!;
    const dy = ys[a]! - ys[b]!;
    s += Math.round(Math.sqrt(dx * dx + dy * dy));
  }
  return s;
}

const names = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.txt'))
  .sort();
for (const name of names) {
  const { xs, ys } = readInstance(join(INPUT_DIR, name));
  const n = xs.length;
  // small instances are cheap per kick: give them a floor of kicks
  const tour = solve(xs, ys, Math.max(KICKS_PER_CITY * n, 150_000), 0x2545f491 ^ n);
  if (process.env.MAKEFUN_DEBUG) process.stderr.write(`${name} n=${n} len=${tourLength(xs, ys, tour)}\n`);
  writeFileSync(join(OUTPUT_DIR, name.replace(/\.txt$/, '.tour')), `${Array.from(tour).join('\n')}\n`);
}
