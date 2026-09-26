# Phase 8: bounded S3 localization curriculum

> **Publication context:** This document is a historical phase-time technical record within the final published Phase 0–11 research-code snapshot. Candidate, uncommitted, review-pending, and later-phase-not-implemented statements describe that phase boundary, not the current repository status; they are retained for auditability. Current status is recorded in the [final Phase-11 freeze](phase11.md#final-research-code-freeze-accepted-limitations-and-verification-history) and [repository overview](../README.md). This framing neither changes the contracts or execution limits below nor upgrades historical verification claims.

Phase 8 implements bounded reference mechanisms only. Production populations,
production normalizer fitting, production training, historical checkpoint loading,
sealed evaluation, robustness, artifact migration and Phase 9 are not authorized.
No historical full-training reproduction or new scientific result is claimed.

## Scientific descriptors versus execution

`S3Config` freezes the complete scientific contract. Fixture limits are separate:
at most 16 cases/population, two sequential fixture stages, two epochs/stage,
and four cumulative optimization epochs across continuation calls. Fixture clocks
count executed fixture epochs, not skipped production epochs. Fixture stage 2
therefore begins at fixture global epoch 3; production stage 2 begins at 51.
Neither arrangement changes the SHA-key serialization mechanism.

| Stage | Minimum radial separation | Minimum angular separation (degrees) |
|---|---:|---:|
| 1 | 0.10 | 40 |
| 2 | 0.08 | 25 |
| 3 | 0.08 | 20 |
| 4 | 0.07 | 15 |
| 5 | 0.06 | 10 |
| 6 | 0.05 | 8 |
| 7 | 0.05 | 6 |
| 8 | 0.05 | 5 |

Each production stage describes 70,000 cases and 50 epochs. Manual batches of
128 retain 546 full batches plus a 112-case tail: 547 updates/epoch, 27,350/stage,
218,800 overall. Totals are 560,000 training configurations and 400 epochs.
Validation is one fixed clean analytical Stage-8 population of 15,000 cases,
batch size 256, all stages, all three canonical source slots. No early stopping.

## Population and protected execution

The Phase-1 generator is imported locally only after role, count, seed and stage
checks. There is no module-level unrestricted generator or fitter alias.
Only reference-fixture population roles execute; all seven frozen seed literals
and all eight derived production stage seeds are rejected as fixture seeds.
Production plans are immutable mappings of non-callable descriptive values.
Production entrypoints reject unconditionally before scientific execution.

The existing Phase-1 law is reused unchanged: three unit sources, interleaved
(3,2) uniform draws, area-uniform radii on [0.05,0.95), uniform azimuths on
[0,2*pi), complete-triple rejection, all pairs passing independent radial and
wrapped-angular thresholds. Acceptance is `value >= threshold` or `isclose`
with zero relative tolerance and `8*spacing(float64(threshold))` absolute
tolerance. Angular thresholds are converted to radians first. Canonicalization
sorts radius then azimuth; duplicate canonical source sets are rejected.

The analytical fixture seam accepts a Phase-1 forward interface, sums unit-source
complex128 fields, and returns clean channels. Tests inject a tiny deterministic
forward stub; they do not execute production PhysicsTM or load historical weights.

## Seed ledger and exact addressing

| Role | Seed |
|---|---:|
| Training population master | 20262201 |
| Validation | 20262202 |
| Initialization | 20262203 |
| Minibatch order | 20262204 |
| Training augmentation | 20262205 |
| Sealed, reserved | 20262206 |
| Robustness, reserved | 20262207 |

Training population seed derivation reuses Phase 1:
`s3_analytical_noise_final_v1|POPULATION|<master>|STAGE|<stage>`.
SHA-256 first 16 digest bytes are interpreted little-endian for PCG64DXSM.

Configuration IDs are exactly `S3ANF-<train|validation>-ST<stage:02d>-<h[:24]>`,
where h hashes canonical `stack((rho,phi),axis=1).astype('<f8').tobytes()`.
There is no row or seed prefix. Historical U64 serialization is recorded evidence;
fixture identities are immutable tuples of the same strings.

Order and noise keys are UTF-8:
`study|master|stage|global_epoch|exposure|identity|stream`.
Use SHA-256 first 16 bytes, little-endian, PCG64DXSM. Stages and epochs are
one-based; historical exposure is zero. Ordering uses identity ALL and stream ORDER.

Per configuration, noise streams are E_SNR, E_NOISE, H_SNR, H_NOISE in that order.
Each field independently draws Uniform[30,40) and a (30,2) standard-normal array.
E and H do NOT share gamma. The thin adapter calls Phase-2 scaling/addition twice,
retaining the E output from its E-gamma call and H output from its H-gamma call.
It does not duplicate the noise formula. Receipts retain both gammas and raw noisy
channel digests. Clean raw targets remain unchanged for FC. Noise precedes normalization.

## Approved normalization resolution

Closed Phase 2 is unchanged and authoritative for reference accumulation.
All eight clean training populations contribute in stage/stored-row order, four
channels, 30 angles, ddof=0, 16,800,000 scalar values/channel. No validation or
noisy observations contribute. One frozen normalizer, no epsilon, no adaptation.
Phase-2 30-angle block merging is approved. Historical scalar Welford is evidence
only; no bitwise equivalence is claimed and no alternate fitting mode exists.
Bounded tests construct small declared normalizer fixtures; no production fit occurs.

## Model, losses and inference boundary

Reuse Phase 5: circular CNN 4->32->64->96, kernels 5/5/3, dense 2880->128,
four LeakyReLU(0.01) activations, 399,433 localizer parameters, three active slots.
Initialization preserves caller Torch RNG/default dtype using the Phase-6 isolation
convention: float32 layer construction then completed float64 model. This isolation
is a reference convention, not a claim about historical global RNG behavior.
No historical initialization digest is available or invented.

Radius is `0.05+0.90*sigmoid(raw_radius)`; angle is atan2(sin_like,cos_like).
No sorting, clipping, endpoint repair or angular projection.
S3 OUTER uses `(1+rho_true)/1.635` on radial and angular-vector squared errors,
averaged over batch AND source. The circle term is UNWEIGHTED.
`L_loc=L_rho+2*L_phi+0.1*L_circle`; this is not S2's source-sum reduction.

Phase-5 FC uses frozen E/H deployment surrogates, differentiable decoded coordinates,
unit amplitudes, source-axis superposition, and clean analytical raw targets.
Four channel MSEs feed `K=0.5*sum(exp(-s)*L_c+s)` and `L=L_loc+0.03*K`.
The four zero-initialized scalars are unclamped and optimized with the localizer:
399,437 optimized parameters. Effective channel coefficients reported are
`0.015*exp(-s)`. Surrogates are excluded from optimization and scientific inference.

## Stage lifecycle and checkpoint ordering

Every stage recreates Adam (lr .0005, betas .9/.999, eps 1e-8, weight decay 1e-5)
and CosineAnnealingWarmRestarts (T_0=200, T_mult=2, eta_min=1e-6).
The single group includes localizer and Kendall scalars with equal weight decay.
Explicit false flags for foreach/fused/amsgrad/maximize/capturable/differentiable
are reference conventions, not independently persisted historical flags.

Model, Kendall, global clocks, history, validation, normalizer and global BEST
persist. Adam moments and scheduler clock reset. Geometry is not regenerated.
There is no result-dependent transition and no early-stopping state.

Epoch order is training, clean validation, strict global BEST snapshot, history,
`scheduler.step(stage_epoch)`, continuation boundary. Selection uses historical
Torch `sqrt(canonical_errors.square().mean())`, not a substituted NumPy summary.
Additional Phase-5 metrics are descriptive; matching cannot enter selection or loss.
The six-permutation diagnostic matcher retains the first candidate on exact ties.

BEST snapshots are pre-scheduler model/Kendall records. TERMINAL never overwrites
BEST. Continuation is a distinct role, post-scheduler only. It persists model,
Kendall, optimizer, scheduler, stage/local/global clocks, updates, history, global
BEST and its snapshot, bindings, runtime metadata and a versioned boundary.
Completed stages encode transition_pending; the next run recreates stage objects
once before training. No mid-epoch resume, role substitution, identity mismatch,
live-engine rewind or cumulative budget overflow is accepted.
Historical BEST/TERMINAL files do not claim these continuation capabilities.

### Narrow continuation compatibility correction

The initial independent audit found that matching identity bindings alone did
not prevent altered serialized Adam options from being restored. The loader now
prepares a separate optimizer/scheduler pair from the unchanged frozen constructor
and advances only its scheduler to the saved stage-local epoch. It compares the
entire serialized parameter group (including parameter order, runtime flags and
initial/current LR) and scheduler state against that expected evolved state.
Thus valid non-initial LR is accepted, but changed Adam settings, scheduler
settings/clocks or inconsistent optimizer/scheduler LR are rejected.

Adam state must also have the complete parameter mapping, stage-local step counts,
finite correctly shaped/dtyped moments and nonnegative second moments. These
checks and optimizer/scheduler deserialization occur before live model/Kendall
restoration. Rejection regressions compare the destination's full continuation,
terminal record, object identities, gradients, model modes and Torch RNG state.
No optimizer update, initialization, RNG draw or validation is performed by the
compatibility check. Stage reset and scientific configuration remain unchanged.

This correction and its regression tests do not complete the outstanding
independent probes from the first audit or authorize Phase-8 closure. Targeted
independent re-verification is still required.

## Historical evidence and acceptance

The 400-row scalar fixture is projected without numerical alteration from history
SHA-256 `81aa549b1c1655aacd4c8ccc74b27347fc462235c2869b663038a6ac2b2cba3c`.
It contains only global/stage/local epochs, updates, LR, score, strict-BEST flag
and cumulative BEST. Replay constructs no model, optimizer, normalizer or RNG.
It verifies 78 strict BEST events, BEST 394/stage8/local44/update215518 with RMSE
0.01900846562333382, terminal400/update218800 with RMSE0.020064019380502007.
These are historical scalars, not full-training tolerances or reproduced results.

Class A covers exact identities, addressing, counts, transitions and replay.
Class B defaults are rtol=1e-12/atol=1e-14 for appropriate float64 primitives.
There is no Class-C full-training tolerance. No historical checkpoint is loaded.
Provenance receipts are immutable descriptive evidence, not authentication.
Closed Phase 0-7 code is reused unchanged. All candidate files remain uncommitted
pending independent verification and a separate closure authorization.
