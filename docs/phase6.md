# Phase 6: bounded S1 localization infrastructure

Authority: Phase 6 Scientific Contract Resolution v1.0 and Bounded
Implementation Authorization v1.0. This phase encodes the final S1 training
mechanisms. It does not execute or authorize production training, final population
generation, production normalization, historical checkpoint loading/migration,
sealed evaluation, robustness evaluation, or Phase 7+.

## Configuration and execution boundary

`inverse_em.config.s1.S1Config` freezes the final scientific settings and uses the
existing canonical SHA-256 mechanism. The final population roles are 70,000 train
cases (20261811) and 15,000 clean validation cases (20261812). The source law is
one unit-amplitude source, area-uniform radius [0.05,0.95), uniform azimuth
[0,2*pi), and 30 endpoint-excluded observation angles. Initialization, order and
augmentation seeds are 20261813, 20261814 and 20261815. Seeds 20261816 and 20261817
are reserved metadata only; no Phase-6 execution API implements those roles.

`population_plan` returns metadata, not a generated scientific population.
`tiny_population` calls the published Phase-1 generator using a non-production
fixture seed and REFERENCE_FIXTURE role, with at most 16 cases.
`analytical_fixture` is a tiny injectable adapter for the Phase-1 forward API.
`production_normalizer_plan` returns immutable, non-callable descriptive metadata
identifying the existing `fit_clean_training_normalizer(InverseTask.S1)` and its
unchanged clean-training statistics. It contains no executable fitter or bypass.
The former `production_normalizer_entrypoint` explicitly raises PermissionError
before any production work. The production fitter is not imported into the
Phase-6 module. Regression sentinels protect both fitting and population-generation
boundaries. Planning metadata is evidence only, not an execution credential.

The final budget is 200 epochs: 546 full batches of 128 plus a tail of 112,
547 updates per epoch and 109,400 updates total. `budget()` verifies this
arithmetically. The only implemented execution engine, `BoundedS1Trainer`,
accepts at most 16 cases per role and four cumulative fixture epochs. These are
software test bounds, not an alternative scientific training protocol. No CLI
production runner is enabled.

## Reuse and data chain

Population/physics: Phase 1. Noise scaling/addition and frozen normalizer: Phase 2.
Frozen differentiable surrogates: Phase 3. Localizer, active-slot view, loss,
FC/Kendall and canonical metrics: Phase 5.

Training is clean analytical complex128 E/H -> keyed raw-complex augmentation ->
float64 Re(E), Im(E), Re(H), Im(H) channels -> frozen training normalizer -> CPU
float64 model. Validation skips augmentation and uses the same normalizer.
FC receives clean analytical RAW UNNORMALIZED channels, never noisy inputs.
Normalizer statistics use only clean training observations, all 30 angles and
population SD (ddof=0). Production fitting is not part of bounded execution.

## Exact historical RNG addressing

S1 does not use the sequential Phase-2 TrainingNoiseStream. The narrow adapter
creates standardized draws and delegates scientific power/scaling/addition to
Phase 2; it does not duplicate the noise law.

For one-based epoch e and configuration identity id, UTF-8 encode exactly:

    s1_analytical_noise_final_v1|master|e|id|stream

Take the first 16 SHA-256 bytes as a little-endian integer and initialize NumPy
PCG64DXSM. With master 20261815, SNR draws one Uniform[30,40) gamma. Independent
E and H keys draw shape (30,2), column 0 real and column 1 imaginary. Both fields
use the same gamma, but independent powers and Gaussian draws. There is no
clipping, denoising or realized-noise rescaling.

Historical IDs are `S1FINAL-<role>-` plus the first 32 hexadecimal characters of
SHA-256 over UTF-8 `study|population_seed|role|zero_based_row` concatenated with
the little-endian float64 bytes of `[rho,phi,1.0]`. Role strings are exactly
`train` and `validation`. Fixture IDs use explicitly declared fixture seeds.
Historical coordinate arithmetic and Phase-1 coordinate arithmetic need not
produce byte-identical source triples; the ID recipe is exact for the supplied
triple. This distinction is not hidden by a historical-trajectory claim.

Order uses master 20261814, identity ALL and stream ORDER with the same key
derivation, followed by permutation(count). Explicit batch sampling retains the
tail, uses zero workers and no shuffle. A dedicated CPU loader generator uses
20261814 + epoch. The recorded historical epoch-1 70,000-index permutation
digest is covered without generating observations or performing optimization.

## Initialization, model and objective

The approved reference wrapper isolates caller CPU Torch RNG and default dtype,
seeds the owned CPU generator with 20261813, constructs the published localizer
under float32 default layer initialization, and returns the completed float64
model. Phase 5 already converts after construction; the wrapper's final double()
is idempotent. Caller RNG and dtype are restored even on failure. This isolation
is new software hygiene, not a historical-process claim.

Only slot 0 is active. The model has 399,433 parameters; four initially zero
Kendall scalars bring the optimization set to 399,437. Phase-5 composition is:

    L_loc = mean((rho_hat-rho)^2 + 2*((c-cos(phi))^2+(s-sin(phi))^2)
                 + 0.1*(c^2+s^2-1)^2)
    K_FC = 0.5 * sum(exp(-s_c)*L_c + s_c)
    total = L_loc + 0.03*K_FC

No OUTER/inverse-radius weighting, assignment, inactive-slot loss, or Kendall
clamp is added. Frozen eval-mode E/H surrogates are excluded from Adam but retain
coordinate gradients. They do not enter the scientific prediction object.

## Optimizer and epoch ordering

One Adam group owns localizer parameters plus four Kendall scalars, with
lr=5e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-5. Coupled Adam weight decay
also applies to Kendall. Optional flags amsgrad/maximize/foreach/fused/capturable/
differentiable are pinned False as reference software settings; historical code
omitted these flags, with foreach/fused defaulting to None.

CosineAnnealingWarmRestarts uses T_0=200, T_mult=2, eta_min=1e-6. The fixed loop is:

1. train mode, prepare epoch exposures, normalize;
2. explicit batches: zero_grad(set_to_none=True), forward/objective, finite loss,
   backward, finite gradients, optimizer step, sample-weighted float64 sums;
3. eval mode, full clean validation under no_grad, batches of 256;
4. scheduler.step(one_based_epoch), record current/next LR;
5. strict lower Cartesian-RMSE BEST snapshot/callback, then append history.

There is no patience counter or early stop. Exact ties retain the earlier BEST.
Snapshots are deep copies of scientific state, with distinct BEST and TERMINAL
roles. A callback can persist a BEST snapshot; this phase performs no production
checkpoint I/O. Terminal snapshots retain the earlier BEST snapshot.

Reference LR used -> next LR: epoch 1, .0005 -> .0004999692198041743;
epoch 100, .0002544189756692993 -> .0002505;
epoch 199, 1.1231131887499658e-6 -> 1.0307801958256833e-6;
epoch 200, 1.0307801958256833e-6 -> .0005. The restart follows final optimization
and validation. The shared epoch-end seam performs no optimizer step.

## Epoch-boundary continuation and evidence

Continuation binds model, Kendall, Adam, scheduler, completed/next epoch, update
count, BEST score/epoch/update, selected BEST snapshot, explicit post-validation/
post-scheduler boundary, history and history position. It records configuration,
population/normalizer identity, Phase-5 identity, deployment surrogate hashes,
actual fixture surrogate tensor identities, angle identity, RNG convention and
seed roles, order/noise versions, runtime versions and CPU Torch RNG snapshot.

There is no stochastic forward operation. Keyed NumPy streams and isolated
loader generators need no advancing NumPy/global Torch state on resume. The
global Torch snapshot is descriptive; loading does not change caller RNG.
Model eval/train mode is selected explicitly on validation/next epoch.
Mid-epoch snapshots are rejected. Tests compare uninterrupted and resumed
model, Kendall, optimizer, scheduler and history exactly.

Receipts are descriptive evidence, not tokens, credentials or authority. The
fixture surrogate identities do not claim that toy models are the production
deployment pair.

## Historical reference and acceptance

`inverse_em.provenance.s1` stores all authorized historical population,
normalizer, initialization, configuration, manifest, revision and serialized
checkpoint identities, plus all nine historical validation summaries. BEST was
epoch 200/update 109,400 with Cartesian RMSE 0.00022168686968818813. These are
immutable descriptive constants, never production-retraining assertions.

The historical checkpoint hash identifies complete serialized training-state
files, not just model tensors. No historical binary is opened or migrated.
State-key names, serialization, normalization accumulation, angle wrapping and
radius arithmetic differ from historical code. Exact key/draw mechanics do not
guarantee identical full optimization or artifact bytes across implementations.

Acceptance is mechanism-level. Class-A integer/state/digest identities are exact;
Class-B rtol=1e-12/atol=1e-14 apply to appropriate float64 primitive fixtures.
No full-training Class-C tolerance is defined. Full historical metric reproduction
is deferred to separate authorization. Tests forbid production fitting, PhysicsTM
execution and checkpoint loading in Phase 6, and use synthetic fields only.
