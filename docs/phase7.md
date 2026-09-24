# Phase 7: bounded S2 localization

Status: bounded implementation candidate, not a production training runner.
Parent: ffc8da1abb8e723a3bb9887180b328373b058a44, tree
aa44b032d15fbe2c663ce7aebc79652b46440b09.

Authority is the Phase-7 Scientific Contract Resolution v1.0 and Bounded
Implementation Authorization v1.0. The former resolves the early-stop equality
boundary and confirms the Blueprint phase boundary. No Phase 8+ is implemented.

## Frozen configuration

`inverse_em.config.s2.S2Config` uses the established canonical serializer.
Its reference-repository configuration SHA-256 is:

    11893683e0eba9070808ae16e10074eb5033c2c7767aeca4bae523713161fbef

This is NOT the historical final-config file hash. The configuration binds
Phase-5 configuration 8303dc42fce7f495bd882b07573948861bd7a35025a141b2224be6e21dd6c363
and the frozen deployment E/H identities, without loading historical weights.
Every dataclass setting is frozen and validated against its declared value.

| Role | Count | Seed |
|---|---:|---:|
| Training population | 70000 | 20262001 |
| Clean validation population | 15000 | 20262002 |
| Localizer initialization | 1 | 20262003 |
| Minibatch ordering | - | 20262004 |
| Training augmentation | - | 20262005 |
| Sealed, non-executable | 15000 | 20262006 |
| Robustness, non-executable | - | 20262007 |

Only initialization/order/augmentation mechanism roles resolve executable seeds.
Production population roles do not. Tiny populations require unrelated fixture
seeds and explicit reference-fixture generation. Integer seeds are not authority.

## Population and analytical observations

The adapter reuses Phase-1 `generate_s2`, not a second sampler. Proposals contain
two unit sources with area-uniform radius [0.05,0.95) and uniform raw azimuth
[0,2*pi). Reject/redraw the whole pair unless BOTH separations pass. Targets
are sorted radius-first, with raw azimuth breaking exact-radius ties.

Radial threshold is 0.05, with rtol=0 and atol=8*spacing(0.05), equal to
5.551115123125783e-17. Angular threshold is deg2rad(5), equal to
0.08726646259971647, with rtol=0 and atol=1.1102230246251565e-16.
Wrapped separation is min(d,2*pi-d), d=abs(phi0-phi1)%(2*pi).
These eight-ULP comparisons are the approved executable law, not NumPy default
isclose tolerances. Phase-1 generation and its canonicalization are unchanged.

The tiny analytical seam calls the existing forward interface once per active
source, sums complex128 fields, then exposes clean E/H. Thirty endpoint-excluded
angles are 2*pi*arange(30)/30. Channelization is Re(E), Im(E), Re(H), Im(H).
The injected test forward is synthetic; Phase-7 tests forbid actual PhysicsTM
execution. Production inverse observations remain analytical, never surrogate.

Normalization reuses the frozen Phase-2 S2 normalizer. Production science is
clean analytical training-only means/population standard deviations, ddof=0,
over 70000*30 observations per channel. Noise precedes normalization.
Tests inject explicitly synthetic statistics, not production fitted statistics.

Population/normalizer/training plans are immutable descriptive mappings, with
no fitter, callback, partial, token, environment bypass, or execution switch.
Execution-named production entry points reject immediately.

## Localizer, loss, field consistency

Reuse the unchanged Phase-5 CPU float64 common model (399433 parameters):
circular conv 4->32 k5, 32->64 k5, 64->96 k3, flatten2880, dense128,
three radius/six angular outputs and four LeakyReLU(0.01) activations.
The approved radius decoder and atan2 angle decoder are unchanged.
Only slots 0,1 enter public prediction, loss, FC and metrics. Slot2 stays internal.
No post-sorting of predictions is performed.

Phase-5 supervised components/reducer implement:

    mean_batch(sum_slots(radial_squared + 2*angular_vector_squared
                         + 0.1*unit_circle_squared))

There is no mean over source slots, OUTER weighting or permutation minimization.
Diagnostic identity/swap matching is separate and cannot supply canonical loss
targets. Strictly cheaper swap wins; ties retain identity.

Frozen Phase-3 E/H interfaces reconstruct two active unit-source responses with
explicit source-axis superposition. Four channel MSEs average batch/angles
against CLEAN RAW UNNORMALIZED analytical fields. Phase-5 FC implements:

    K = 0.5 * sum(exp(-s_c)*L_c + s_c)
    total = localization + 0.03*K

Four zero-initialized float64 scalars are trainable/unclamped; surrogates stay
frozen/eval and outside Adam. Coordinate gradients remain live. Synthetic
fixture surrogate tensor identities are recorded separately from historical
deployment identities and are never represented as deployment weights.

## Historical keyed noise and order

S2 IDs use canonical little-endian float64 bytes [rho0,phi0,rho1,phi1].
The SHA-256 input is UTF8(str(population_seed)+"|"+str(zero_based_row))+raw.
ID is "S2ANF-"+role+"-"+hexdigest[:32], with role train or validation.
Amplitude and study name are NOT included in that digest. Do not use S1 IDs.

For all addressed streams:

    key = "s2_analytical_noise_final_v1|master|epoch|identity|stream"
    seed = int.from_bytes(sha256(UTF8(key)).digest()[:16], "little")
    rng = Generator(PCG64DXSM(seed))

Epochs are one-based. Augmentation master20262005 uses SNR/E/H streams.
One uniform[30,40) gamma is shared between E/H. Independent E/H arrays have
shape (30,2), columns real/imaginary. Phase-2 scale_directions/add_scaled_noise
compute the physical noise; there is no duplicate S2 noise formula and no
sequential TrainingNoiseStream.

Ordering uses master20262004, identity ALL, stream ORDER. Permute the count,
then manually slice batches128. No DataLoader or loader-generator role exists.
Production arithmetic is descriptive: 70000=546*128+112, 547 updates/epoch,
600 maximum epochs=328200 updates. Bounded order calls accept at most16 cases.
Independent literal ID/key/draw/order fixtures are included.

## Initialization and optimizer

Construct frozen FC/four zero scalars before localizer initialization in normal
assembly. Initialization uses seed20262003, CPU default-float32 layer draws,
then double precision. Phase-5 already has parameter-free LeakyReLU; replacing
ReLU historically consumed no RNG. Caller CPU RNG and default dtype are restored
using the Phase-6 isolation PATTERN, explicitly a reference-repository convention.
No historical initialization digest was supplied or invented.

Adam owns one group: all localizer parameters plus four scalars=399437.
lr=5e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-5. Not AdamW.
Explicit false amsgrad/maximize/foreach/fused/capturable/differentiable flags
are a reference convention, not historically explicit arguments. A two-update
scalar regression compares against the historical default-argument construction.

## Exact epoch-end state machines

S2 order, intentionally different from S1:

1. Training updates.
2. Clean validation.
3. Strict BEST accounting.
4. Material early-stop accounting.
5. Copy new BEST if applicable (pre-scheduler state).
6. Append history (in-memory bounded implementation).
7. scheduler.step(completed_epoch), one-based.
8. Check stale>=200; break if stopping.
9. Represent TERMINAL after loop exit, post-scheduler.

Scheduler is CosineAnnealingWarmRestarts(T_0=200,T_mult=2,eta_min=1e-6).
Recorded LR is the epoch's pre-step LR:

| Epoch | LR |
|---:|---:|
| 1 | 0.0005 |
| 199 | 1.1231131887499658e-06 |
| 200 | 1.0307801958256833e-06 |
| 201 | 0.0005 |
| 392 | 0.00026812143298982614 |

BEST is any score<best, no min_delta; exact ties retain earlier BEST.
The independent material state initializes reference=None, stale=0:

    if es_reference is None or score < es_reference - 1e-5:
        es_reference = score
        stale = 0
    else:
        stale += 1
    stop = stale >= 200

The subtraction expression is literal. Do not algebraically rearrange it.
Equality and the next float64 value above the boundary do not reset; the next
value below does. Nonfinite scores fail. The first finite score sets reference.
BEST may improve without resetting patience.

## Validation and bounded continuation

Clean validation uses model.eval/no_grad, no noise, canonical active slots and
batch256 semantics. All source errors are pooled, never per-batch-RMSE averaged.
Tests use a retained incomplete validation batch inside the 16-case firewall.
Wrapped angular diagnostics come from Phase5; they are not selection criteria.

The bounded trainer accepts <=16 cases per role and <=4 cumulative optimization
epochs, including restoration. Shapes, canonical conditioned targets, roles,
normalizer task, identity overlap and bounds are checked. No file checkpoint
loader is provided. No scientific population is created by the trainer.

Continuation is POST validation/accounting/history/scheduler, distinct from
pre-step BEST. In-memory copied snapshots contain model/Kendall/Adam/scheduler,
completed/next epoch/update, history, separate BEST, material reference/stale,
patience/expression, termination, config/population/normalizer/dependency
identities, actual fixture surrogate tensors, RNG/order versions, seed roles,
runtime versions and descriptive CPU Torch RNG. Stateless addressed draws need
no advancing NumPy state. Loading preserves caller RNG; no stochastic forward
operation exists. Mid-epoch/BEST-as-continuation, altered identities/counters,
inconsistent selection history and out-of-bounds restoration are rejected.

BEST and TERMINAL do not alias live model tensors. Continuation tests compare
all states and history exactly. A test-only short-patience property injection
exercises loop termination within three tiny epochs; it is not a configurable
scientific training parameter. Frozen patience200 is separately tested at its
exact boundary and by the actual historical scalar replay.

## Historical scalar evidence

tests/phase7/history_scalars.jsonl contains a compact scalar extraction of the
392-row historical history, whose original SHA-256 was verified before extraction:
1d4a12e11b09a8444494117a21a53f2a232407e767a247d816f18e528610b842.
Columns: epoch, updates, current canonical RMSE, cumulative BEST RMSE,
BEST epoch, material reference, stale count.
This is a small regression fixture, not model/data migration.

Replay is limited to600 scalar rows, performs no inference or optimization,
and verifies every recorded state transition. Expected: last material reset192,
strict later BESTs194/196/197/199, BEST199/update108853,
first stop392/update214424/stale200. BEST RMSE=0.0022449191022022525;
terminal RMSE=0.0028001476403387822. Mean errors are respectively
0.0018307097348069967 and 0.002371487613891901.

The provenance module stores supplied hashes descriptively. Initialization and
protocol-manifest hashes remain None. Receipts are evidence, not credentials.
No historical checkpoint is opened; no sealed/robustness artifact is traversed.
No full-training Class-C tolerance exists. Mechanism acceptance does not imply
historical model-weight, serialized-byte or full-trajectory reproduction.

## Verification scope

Phase7 tests instrument generation, production normalization, PhysicsTM and
checkpoint-loading boundaries, assert zero protected calls after each test,
and exercise rejected production/reserved requests with sentinels.
Import hygiene tests forbid scientific/network operations and compare RNG and
filesystem state. Prior Phase0-6 tests remain unchanged.

Run focused Phase7, prior-suite excluding Phase7, then the entire repository
suite. Use python -B and a fresh pytest basetemp when Windows shared temp ACLs
prevent access; do not alter scientific code to repair test-directory access.
No production training, final population generation, production fitting,
sealed/robustness execution, historical checkpoint loading or Phase8 work is
part of this implementation.
