//! colibrì GLM-5.2 forward pass (M1, in progress).
//!
//! Ported piece by piece from `c/glm.c`, each validated in isolation before the
//! full forward is wired up. Present: the elementary numerics ([`math`]) and the
//! MoE router ([`router`]). Next: MLA attention, the MoE/dense-MLP expert compute
//! on [`coli_kernels`], the full layer/forward loop, and — on a machine with
//! `transformers` — the token-exact gate against `c/ref_glm.json`.

pub mod attention;
pub mod math;
pub mod mlp;
pub mod model;
pub mod mtp;
pub mod router;
pub mod sample;
pub mod testkit;
pub mod weight;

pub use attention::{mla_attention, mla_attention_absorb, AttnWeights, LayerKv};
pub use math::{layernorm, rmsnorm, rope_interleave, sigmoidf, siluf, silu_mul, softmax};
pub use mlp::{moe_forward, Mlp};
pub use model::Model;
pub use mtp::speculative_sample;
pub use router::{batch_union, route, Routed};
pub use sample::{argmax, Sampler};
pub use weight::QtWeight;
