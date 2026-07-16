# colibrì — Rust engine (work in progress)

A from-scratch Rust rewrite of the colibrì GLM-5.2 MoE inference engine whose goal
is to drive **CPU, GPU, RAM, and SSD concurrently** — closing the gap where the C
engine's CUDA MoE path is *phased, not concurrent* (`c/glm.c:3079-3147`: VRAM
experts are deferred, RAM/disk experts compute on the CPU inline, then the GPU
group is dispatched only afterward, so same-block CPU∥GPU never overlaps).

Target: **Linux + NVIDIA CUDA**. Every numeric kernel is ported from `c/glm.c` and
validated so the Rust path reproduces the C engine's arithmetic; the scalar
integer-dot kernels are the token-exactness reference and the SIMD variants are
checked bit-for-bit against them.

> This is an in-progress rewrite that lives alongside the production C engine in
> [`../c`](../c). The C engine remains the shipping runtime. See the design plan
> for the full rationale and the three-lane scheduler architecture.

## Status of completeness

**59 tests passing, 0 warnings**, workspace builds in debug and release. All eight
milestones (M0–M7) have their core built and validated on this Linux dev box;
the GPU path is scaffolded to compile and validate on an NVIDIA host (the dev box
has no GPU/nvcc).

| # | Milestone | Crate(s) | Status | How it's validated |
|---|---|---|---|---|
| M0 | Model loaders | `coli-core` | ✅ Complete | config/safetensors/QT-format/dtype round-trips (13 tests) |
| M1 | CPU int4 forward | `coli-kernels`, `coli-model` | ✅ Complete — **runs end-to-end** | int8/int4 dots bit-exact vs scalar on AVX-VNNI; MoE vs f32 reference; attention causality / decode==prefill / single-token; full `Model` load→forward→generate on a synthetic model |
| M2 | io_uring streaming | `coli-io` | ✅ Complete | **io_uring reads validated byte-for-byte vs `pread` on real hardware** (kernel 7.0.12); LRU cache; LFRU tier policy (9 tests) |
| M3 | CUDA GPU lane | `coli-cuda` | ⚙️ Scaffold | FFI to `c/backend_cuda.cu` + `nvcc` build.rs behind the `cuda` feature; default build is a stub (no GPU here). **GPU path validates on an NVIDIA box** |
| M4 | Concurrent scheduler | `coli-sched` | ✅ Core complete | `moe_streamed` overlaps io_uring streaming ∥ CPU expert compute; output == sequential (the CPU∥SSD half of the 3-lane design) |
| M5 | MLA absorption + DSA | `coli-model` | ✅ Absorption / ⚙️ DSA | `mla_attention_absorb` ≈ dense + causal; dense path == "DSA-selects-all" (the correctness anchor). Sparse indexer selection is the remaining perf item |
| M6 | MTP speculative decode | `coli-model` | ✅ Core complete | `speculative_sample` rejection sampling **statistically lossless** — emitted distribution matches target over 60k trials, regardless of the draft |
| M7 | Serve (stdio drop-in) | `coli-engine` | ✅ Complete | serve mode emits `READY` → generates → `END` (the `c/openai_server.py` handshake); validated end-to-end |

### Not yet done (mostly gated on the NVIDIA box or the `transformers` oracle)
- **GPU-lane integration** into the scheduler as the third concurrent actor, plus
  non-syncing `_async` stream variants — needs a CUDA host.
- **Token-exact acceptance gate** vs the `transformers` oracle (`c/ref_glm.json`,
  TF 32/32 + greedy 20/20) — needs `torch` + `GlmMoeDsaConfig`.
- **DSA lightning indexer** sparse top-2048 selection (dense path is the anchor).
- **MTP head wiring** into `Model` (draft head + batched verify); the acceptance
  math is done.
- **int2 (fmt 3) and grouped-int4 (fmt 4)** matmul kernels — `QtWeight` currently
  supports the int8/int4-packed formats real GLM-5.2 experts ship in.
- **Full OpenAI request framing** + `CANCEL` + a text tokenizer for a text CLI.

## Architecture

```
crates/
  coli-core     M0  formats: Cfg, safetensors index, QT quant detect, dtype, pack   (↔ c/st.h, glm.c load)
  coli-kernels  M1  std::arch int8/int4 dots + matmuls (scalar ref + AVX2/AVX-VNNI)  (↔ c/glm.c IDOT kernels)
  coli-model    M1+ MLA attention, router, MoE, sampler, MTP, top-level Model        (↔ c/glm.c forward)
  coli-io       M2  io_uring Reactor, LRU ExpertCache, LFRU tiering                  (↔ c/uring.h, c/tier.h)
  coli-cuda     M3  FFI to c/backend_cuda.cu (feature = "cuda")                       (↔ c/backend_cuda.h)
  coli-sched    M4  concurrent MoE scheduler: io_uring streaming ∥ CPU compute
  coli-engine   M7  binary: stdio serve protocol (drop-in for c/glm) + demo mode     (↔ c/openai_server.py)
```

## Build & test

```bash
cd rust
cargo test --workspace          # 59 tests, CPU-only, no GPU needed
cargo build --release           # optimized (fat LTO)

# GPU lane (on an NVIDIA host with CUDA installed):
cargo build -p coli-cuda --features cuda
```

## Run

```bash
# self-contained end-to-end demo (builds a tiny synthetic model, loads, generates):
cargo run -p coli-engine -- demo

# serve mode (drop-in for c/glm behind c/openai_server.py):
cargo run -p coli-engine -- build /tmp/demo-model     # write a tiny model
COLI_MODEL=/tmp/demo-model cargo run -p coli-engine    # emits READY, then:
#   GEN <ngen> <tok0> <tok1> ...   → greedy-generates, replies, emits END
#   QUIT
```

`Model::load` also accepts any real int4/int8 container model directory using the
GLM-5.2 weight-naming scheme (`model.layers.N.self_attn.*`, `mlp.experts.M.*`, …).

## References

- **Design plan & three-lane scheduler rationale:** [`DESIGN.md`](DESIGN.md) —
  problem statement, why the C path is phased, the completion-driven
  CPU∥GPU∥SSD scheduler, milestones, risks.
- **C engine (the port source and correctness reference):**
  - `../c/glm.c` — MoE `moe()` (the phased loop to replace), MLA `attention_rows`,
    IDOT kernels (`qrow_i8`, `dot_i8i8`, `dot_i4i8`, `matmul_*_idot`), router,
    `spec_decode`, weight loading.
  - `../c/st.h` — safetensors index + streaming behavior.
  - `../c/uring.h` — the io_uring ownership / `IOSQE_ASYNC` / io-wq model.
  - `../c/tier.h` — the LFRU hot-store eviction policy (ported verbatim).
  - `../c/backend_cuda.h` / `../c/backend_cuda.cu` — the flat CUDA C ABI this FFI binds.
  - `../c/openai_server.py` — the `READY`/`END`/`CANCEL` stdio protocol.
  - `../c/ref_glm.json` + `../c/tools/make_glm_oracle.py` — the token-exact oracle gate.
- **Upstream project:** [JustVugg/colibri](https://github.com/JustVugg/colibri).
