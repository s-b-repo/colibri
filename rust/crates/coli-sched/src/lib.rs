//! colibrì concurrent MoE scheduler (M4 core).
//!
//! The throughput lever from the plan: instead of the C engine's phased
//! "stream-then-compute", this overlaps the **I/O lane** (io_uring streaming of
//! disk-resident experts) with the **CPU lane** (computing RAM-resident experts)
//! on the same MoE layer, then merges. Output is identical (within float
//! reassociation) to the sequential path.
//!
//! This is the CPU∥SSD half of the three-lane design; the GPU lane composes the
//! same way (feature-gated FFI, validated on an NVIDIA box).

pub mod reconstruct;

use std::os::unix::io::RawFd;

use coli_model::{batch_union, route, Mlp, Routed};
use reconstruct::{mlp_from_blob, MlpDims};

/// Where an expert's weights live.
pub enum ExpertLoc<'a> {
    /// already in RAM — compute immediately on the CPU lane
    Resident(&'a Mlp),
    /// on disk — stream via the io_uring I/O lane, then compute
    Disk(DiskExpert),
}

/// A streamable expert: one coalesced blob at `[offset, offset+len)` in `fd`.
pub struct DiskExpert {
    pub fd: RawFd,
    pub offset: u64,
    pub len: usize,
    pub dims: MlpDims,
}

/// Read every disk expert's blob. On Linux this uses one io_uring
/// `submit_and_wait` for the whole batch; otherwise a `pread` fallback.
fn read_disk_blobs(specs: &[(usize, RawFd, u64, usize, MlpDims)]) -> Vec<(usize, Vec<u8>, MlpDims)> {
    let mut bufs: Vec<Vec<u8>> = specs.iter().map(|&(_, _, _, len, _)| vec![0u8; len]).collect();

    #[cfg(target_os = "linux")]
    let ok = {
        use coli_io::{ReadReq, Reactor};
        match Reactor::new(specs.len().max(1) as u32) {
            Ok(mut reactor) => {
                let mut reqs: Vec<ReadReq> = bufs
                    .iter_mut()
                    .zip(specs)
                    .map(|(b, s)| ReadReq { fd: s.1, offset: s.2, buf: b.as_mut_slice(), tag: s.0 as u64 })
                    .collect();
                reactor.read_many(&mut reqs).is_ok()
            }
            Err(_) => false,
        }
    };
    #[cfg(not(target_os = "linux"))]
    let ok = false;

    if !ok {
        use coli_io::{pread_many, ReadReq};
        let mut reqs: Vec<ReadReq> = bufs
            .iter_mut()
            .zip(specs)
            .map(|(b, s)| ReadReq { fd: s.1, offset: s.2, buf: b.as_mut_slice(), tag: s.0 as u64 })
            .collect();
        pread_many(&mut reqs);
    }

    bufs.into_iter().zip(specs).map(|(b, s)| (s.0, b, s.4)).collect()
}

/// Accumulate one expert's contribution into `out` (gather routed rows → SwiGLU
/// → weighted scatter). Identical math to `coli_model::moe_forward`'s inner loop.
fn contribute(out: &mut [f32], x: &[f32], mlp: &Mlp, r: &Routed, eid: usize, hidden: usize, s_n: usize) {
    let mut rows = Vec::new();
    let mut rw = Vec::new();
    for s in 0..s_n {
        for kk in 0..r.keff[s] as usize {
            if r.idx[s * r.k + kk] as usize == eid {
                rows.push(s);
                rw.push(r.w[s * r.k + kk]);
                break;
            }
        }
    }
    if rows.is_empty() {
        return;
    }
    let nr = rows.len();
    let mut xg = vec![0f32; nr * hidden];
    for (ri, &s) in rows.iter().enumerate() {
        xg[ri * hidden..ri * hidden + hidden].copy_from_slice(&x[s * hidden..s * hidden + hidden]);
    }
    let h = mlp.swiglu(&xg, nr);
    for (ri, (&s, &wgt)) in rows.iter().zip(&rw).enumerate() {
        let dst = &mut out[s * hidden..s * hidden + hidden];
        let src = &h[ri * hidden..ri * hidden + hidden];
        for d in 0..hidden {
            dst[d] += wgt * src[d];
        }
    }
}

/// Concurrent MoE forward: I/O lane streams disk experts while the CPU lane
/// computes resident experts; results merge into one output `[s_n, hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn moe_streamed(
    x: &[f32],
    hidden: usize,
    s_n: usize,
    router_w: &[f32],
    router_bias: &[f32],
    topk: usize,
    norm_topk: bool,
    routed_scale: f32,
    experts: &[ExpertLoc],
    shared: Option<&Mlp>,
) -> Vec<f32> {
    let e_n = experts.len();
    let r = route(x, router_w, router_bias, s_n, hidden, e_n, topk, norm_topk, routed_scale);
    let uniq = batch_union(&r, s_n);

    // partition the batch-union by residency
    let mut resident: Vec<(usize, &Mlp)> = Vec::new();
    let mut disk_specs: Vec<(usize, RawFd, u64, usize, MlpDims)> = Vec::new();
    for &e in &uniq {
        match &experts[e as usize] {
            ExpertLoc::Resident(m) => resident.push((e as usize, m)),
            ExpertLoc::Disk(d) => disk_specs.push((e as usize, d.fd, d.offset, d.len, d.dims)),
        }
    }

    // I/O lane (streaming) ∥ CPU lane (resident compute)
    let (out_from_resident, disk_blobs) = std::thread::scope(|sc| {
        let io = sc.spawn(|| read_disk_blobs(&disk_specs));
        let mut out = vec![0f32; s_n * hidden];
        for &(eid, mlp) in &resident {
            contribute(&mut out, x, mlp, &r, eid, hidden, s_n);
        }
        let blobs = io.join().expect("io lane panicked");
        (out, blobs)
    });

    // compute the streamed experts (now resident) and merge
    let mut out = out_from_resident;
    for (eid, blob, dims) in disk_blobs {
        let mlp = mlp_from_blob(&blob, dims);
        contribute(&mut out, x, &mlp, &r, eid, hidden, s_n);
    }

    if let Some(sh) = shared {
        let hs = sh.swiglu(x, s_n);
        for z in 0..s_n * hidden {
            out[z] += hs[z];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::reconstruct::{blob_len, mlp_to_blob, MlpDims};
    use super::*;
    use coli_core::pack::quant_i4;
    use coli_model::{moe_forward, Mlp, QtWeight};
    use coli_core::QtFmt;
    use std::io::Write;
    use std::os::unix::io::AsRawFd;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    fn qi4(w: &[f32], o: usize, i: usize) -> QtWeight {
        let (q, s) = quant_i4(w, o, i);
        QtWeight::new(QtFmt::Int4, o, i, q, s)
    }

    fn make_mlp(r: &mut Lcg, hidden: usize, inter: usize) -> Mlp {
        let g: Vec<f32> = (0..inter * hidden).map(|_| r.f()).collect();
        let u: Vec<f32> = (0..inter * hidden).map(|_| r.f()).collect();
        let d: Vec<f32> = (0..hidden * inter).map(|_| r.f()).collect();
        Mlp { gate: qi4(&g, inter, hidden), up: qi4(&u, inter, hidden), down: qi4(&d, hidden, inter) }
    }

    #[test]
    fn concurrent_matches_sequential() {
        let (hidden, inter, e_n, k, s_n) = (16usize, 8usize, 6usize, 2usize, 4usize);
        let dims = MlpDims { hidden, inter };
        let mut r = Lcg(0xF00D);

        let x: Vec<f32> = (0..s_n * hidden).map(|_| r.f()).collect();
        let router_w: Vec<f32> = (0..e_n * hidden).map(|_| r.f()).collect();
        let router_bias: Vec<f32> = (0..e_n).map(|_| r.f() * 0.1).collect();
        let experts: Vec<Mlp> = (0..e_n).map(|_| make_mlp(&mut r, hidden, inter)).collect();
        let shared = make_mlp(&mut r, hidden, inter);

        // sequential reference: all experts resident
        let seq = moe_forward(&x, &router_w, &router_bias, &experts, Some(&shared), s_n, hidden, k, true, 2.5);

        // write the odd-indexed experts to a disk blob file; even ones stay resident
        let path = std::env::temp_dir().join(format!("coli_sched_{}", std::process::id()));
        let mut f = std::fs::File::create(&path).unwrap();
        let mut offsets = vec![0u64; e_n];
        let mut cursor = 0u64;
        for (e, expert) in experts.iter().enumerate() {
            if e % 2 == 1 {
                let blob = mlp_to_blob(expert);
                assert_eq!(blob.len(), blob_len(dims));
                offsets[e] = cursor;
                f.write_all(&blob).unwrap();
                cursor += blob.len() as u64;
            }
        }
        f.sync_all().unwrap();
        let rf = std::fs::File::open(&path).unwrap();
        let fd = rf.as_raw_fd();

        let locs: Vec<ExpertLoc> = experts
            .iter()
            .enumerate()
            .map(|(e, m)| {
                if e % 2 == 1 {
                    ExpertLoc::Disk(DiskExpert { fd, offset: offsets[e], len: blob_len(dims), dims })
                } else {
                    ExpertLoc::Resident(m)
                }
            })
            .collect();

        let conc = moe_streamed(&x, hidden, s_n, &router_w, &router_bias, k, true, 2.5, &locs, Some(&shared));

        for z in 0..s_n * hidden {
            let tol = 1e-3 * seq[z].abs().max(1.0);
            assert!((seq[z] - conc[z]).abs() < tol, "z={z} seq={} conc={}", seq[z], conc[z]);
        }
        let _ = std::fs::remove_file(&path);
    }
}
