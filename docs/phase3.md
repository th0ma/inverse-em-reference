# Phase 3: frozen boundary-field surrogates

> **Publication context:** This document is a historical phase-time technical record within the final published Phase 0–11 research-code snapshot. Candidate, uncommitted, review-pending, and later-phase-not-implemented statements describe that phase boundary, not the current repository status; they are retained for auditability. Current status is recorded in the [final Phase-11 freeze](phase11.md#final-research-code-freeze-accepted-limitations-and-verification-history) and [repository overview](../README.md). This framing neither changes the contracts or execution limits below nor upgrades historical verification claims.

Phase 3 implements only `[rho/R, cos(theta-phi), sin(theta-phi)] -> [Re(F), Im(F)]`. `E_z` and `H_phi` use independent CPU `float64` MLPs with architecture `3-128-128-128-128-2`, ReLU hidden activations, biases on every linear layer, and 50,306 trainable parameters each. Targets are raw float64 components of pinned complex128 PhysicsTM evaluations; no target normalization is used.

## Historical dataset compatibility

The isolated surrogate protocol uses `numpy.random.default_rng(20260915)` (PCG64), one vectorized squared-radius draw followed by one continuing-generator azimuth draw, 10,000 source IDs, and 72 endpoint-excluded angles. Partitions sort `SHA256("1.0|20260916|<source_id>")` and slice 7,000/1,500/1,500. This does not alter Phase 1's PCG64DXSM convention. Pointwise datasets retain source-level geometry and use `divmod(index,72)`. Each dataset view is bound to the frozen source-array and split hashes, requires explicit source IDs, verifies ID/geometry alignment, and rejects held-out, unknown, duplicate, or cross-partition identities regardless of the caller's partition label.

## Training contract

Scientific training requires PyTorch 2.8.0. It uses Adam (`lr=1e-3`, betas `(0.9,0.999)`, epsilon `1e-8`, zero weight decay), component MSE, batch 512, a dedicated persistent Torch shuffle generator seeded by the model seed, and `drop_last=False`. `ReduceLROnPlateau` is explicit: min mode, factor 0.5, patience 5, relative threshold `1e-4`, cooldown 0, minimum LR 0, epsilon `1e-8`.

BEST is strict reduction in clean-validation component MSE. Early stopping is an independent 20-non-improvement counter. A full 200-epoch run is 197,000 updates, but this implementation phase does not authorize it.

## Differentiability and evidence

Downstream use calls `eval()` and separately freezes parameter gradients; source coordinates remain differentiable and the path is Torch-only. Complex superposition reduces an explicit source axis (default `source_dim=-2`) while preserving the observation-angle axis, including batched and one-source inputs. Central finite-difference tests use step `1e-6`, `rtol=1e-6`, and `atol=1e-8` as engineering test parameters, not scientific constants. Checkpoint and receipt hashes are descriptive evidence only. Historical checkpoints are referenced but neither loaded nor reproduced here.
