# Phase 10: bounded artifact integration

> **Publication context:** This document is a historical phase-time technical record within the final published Phase 0–11 research-code snapshot. Candidate, uncommitted, review-pending, and later-phase-not-implemented statements describe that phase boundary, not the current repository status; they are retained for auditability. Current status is recorded in the [final Phase-11 freeze](phase11.md#final-research-code-freeze-accepted-limitations-and-verification-history) and [repository overview](../README.md). This framing neither changes the contracts or execution limits below nor upgrades historical verification claims.

Status: implementation candidate, not independently verified or closed. No
production artifact acceptance or Phase-11 execution is enabled.

## Authority and scope

`inverse_em.artifact_integration.integration.SyntheticStore` is the only new
acceptance surface. It creates a fresh temporary namespace, registers synthetic
JSON sources and immutable context identities, then validates exact source bytes.
It does not accept checkpoint paths supplied by a caller. `load_production`
unconditionally rejects. No pickle, `torch.load`, closed checkpoint loader,
population generator, fitting function, trainer or evaluator is called.

Closed Phase 0-9 APIs remain unchanged and are not re-exported as alternate
production entry points. Future production codecs and authentication of actual
historical evidence require separate authorization; this candidate is not such
an importer. The firewall prevents accidental unsupported use, not arbitrary
hostile Python code with access to private interpreter state.

## Use and identity contracts

Uses are exactly `INFERENCE`, `CONTINUATION`, `PROVENANCE_ONLY`. Decisions bind
the source identity, context digest, intended use, outcome and evidence. A
provenance-only acceptance never returns a usable replacement snapshot.

The four identity domains are source-file bytes, tensor/state content, scientific
configuration and native artifact. Schema names further qualify every digest.
Mappings explicitly link endpoints, transformation name/version/exact semantics,
and resolved evidence; equal digest text never makes different namespaces interchangeable. Synthetic source identities are explicitly
labelled `synthetic-source-bytes/1` within the source domain, not represented as
historical deployments.

Immutable contexts cover E, H, classifier, S1, S2, S3 and task normalizers. They
bind closed configuration and architecture schemas, CPU/float64 runtime,
physics/RNG contracts, training/validation/provenance identities, expected actual
state, task, role, normalizer, lineage and continuation component identities.
Contexts are revalidated and checked against their independently registered
digest on every integration. Runtime disagreement yields insufficient evidence;
no cross-runtime tolerance or metric-equivalence rule exists.

## Exact validation

Tensor records contain ordered names, exact shapes, CPU/float64 dtype and
little-endian IEEE-754 bytes. Keys, shapes, required empty buffer sets, parameter
ordering, finiteness and active slots are checked before exposure. Signed-zero
bits are preserved by the state-content identity. Tests compare these literal
schemas with synthetic instances of the closed model classes, without invoking
historical initialization helpers.

Normalizers contain explicit tiny statistics only (at most 16 observations).
Exact task, four-channel order, float64 statistics, positive standard deviations,
30-angle counts, ddof=0, S3 stage coverage, provenance and mappings are checked.
The native Phase-2 block-moment implementation remains untouched; no fitting or
historical tolerance is introduced. A separate EvidenceRegistry must establish a
supported tiny clean-training provenance descriptor and observe explicit exact
statistics. Source registration cannot mint this evidence. Resolution checks the
task, Phase-2 configuration, statistics, channel order, counts, ddof and training
identity against immutable stored evidence bytes. There is no public arbitrary
digest/evidence registration method. Unsupported, circular, placeholder or
unresolved provenance cannot authorize acceptance. These producers establish
synthetic fixture evidence only, not authenticity of historical production data.

Localizer lineage contains E/H source-to-state mappings and closed relative-angle
B configuration. Synthetic evidence includes the actual E/H tensor states;
their computed content identities must match the lineage. A copied deployment
hash cannot make altered E/H values pass. No surrogate forward pass is needed.

## Roles and continuation

Localizer BEST is inference-only; CONTINUATION requires separately bound complete
synthetic components; TERMINAL is not automatically continuation-compatible.
Classifier roles explicitly map `best_validation` and `resumable`. S1 preserves
its post-scheduler boundary, S2 its pre-scheduler BEST and completed continuation
boundary, and S3 its distinct pre-scheduler BEST and completed stage-local
continuation boundary. Phase-3/4 snapshots use a schema-bound portable boundary
descriptor; they are not silently interpreted as localizer boundaries.

The bounded continuation interchange is **not** the closed trainers' native
file serialization and cannot restore or run a trainer. Version 2 supports only
explicit eight-case, one-batch synthetic fixtures: at most two classifier epochs,
four other epochs, and two epochs per S3 fixture stage, matching closed bounded
engine limits. Unknown/legacy interchange evidence remains insufficient.

Complete Adam parameter groups and evolved scheduler state are compared to a
detached reference configured from closed task settings on PyTorch 2.8.0. Only
scheduler control state is replayed; no optimizer step, forward pass, fitting or
training occurs. Ordered moment tensors, nonnegative second moments, exact step
clocks, LR, history/counters, actual BEST model/Kendall content, earliest strict selection and
task-specific early stopping are checked. S3 stage/global clocks and transition
state are reconstructed; Adam/scheduler clocks reset at fixture stage boundaries.

E/H requires actual Torch and loader-generator states; classifier requires actual
owned dropout state and a complete hashed PCG64DXSM noise receipt with seed-role
metadata and event counts. S1/S2 requires Torch state plus keyed contract/counters;
S3 requires its stateless keyed contract, next global epoch and exposure. Torch
state bytes and PCG64DXSM state structures are round-trip validated on isolated
objects without drawing samples or initializing a protected seed. Arbitrary
strings do not establish RNG state. Frozen scheduler infinity sentinels have an
explicit exact symbolic encoding, not a numerical tolerance. No missing state is
synthesized, and production continuation equivalence is not claimed.

## Classification and migration

Only successful integration establishes `DIRECTLY_COMPATIBLE` (synthetic scope).
Missing/mismatching acceptance evidence gives `INSUFFICIENT_EVIDENCE`; rejection
exceptions carry a use-bound decision and never usable state. Unregistered or
invalid contexts and mutated sources fail before acceptance.

The separate descriptive decision-rule API requires explicit evidence for
`CONTROLLED_MIGRATION_REQUIRED`, `REGENERATION_REQUIRED` or `RETRAINING_REQUIRED`.
Missing continuation state alone is not a retraining finding. Retraining requires
unavailable/incompatible required trained state and exhausted approved recovery.
Normalizer regeneration is separate. These decisions do not authorize execution.

Resolved mappings support explicit-statistics observation, E/H tensor observation,
validated-container publication and explicit identity/no-op byte copying. Each
operation has fixed endpoint domains/schemas, version 1 semantics and a supported
evidence schema. Evidence must resolve in the independent registry and reproduce
the observed facts, result identity and applicable context/use bindings. No-op is
explicitly represented as exact byte preservation. Descriptive migration plans
do not themselves issue accepted compatibility mappings.

Migration plans allow only container conversion, metadata-key renaming and
one-to-one state-key mapping with identical semantic associations. The synthetic
validation function checks complete bijections and exact values, including
signed zero. It validates caller-supplied transformations rather than executing
them. Scientific transformations, optimizer reset and synthesized state are
rejected. A plan never bypasses ordinary integration validation.

## Transaction and source protection

Sources are exclusively created, hashed and sized. They are checked before
validation and after validation. The destination is staged and independently
revalidated. At commit, Windows mandatory deny-write/deny-delete handles lock both
source and stage; after the exclusive atomic hard link a separate accepted-path
handle is acquired before verifying committed bytes/file identity. Protecting
one hard-link name is not assumed to protect another name against removal.
Pre-commit mutation is rejected; mutation attempts at the synchronized commit
boundary are denied by the OS. An existing destination is never overwritten.
Evidence issuance is transaction-local until publication succeeds. Any failure
before the protected handoff restores the prior evidence registry and removes
only this transaction's new links. Cleanup failure is explicit and retains a
retry owner rather than claiming rollback completed.

There is no live destination mutation API. Successful output is immutable bytes
representing a completely checked synthetic replacement snapshot; model mode,
gradients, optimizer/scheduler, Kendall and caller NumPy/Torch RNG are untouched.
Usable acceptance returns `ProtectedResult`, not a plain `Result`. Its read-only
state is OPEN until explicit caller `close()` or context-manager exit. Source and
accepted-path protection crosses the function-return boundary. Staging writer
flush/close precedes publication; staging-handle release and temporary-name unlink
occur during the protected handoff while both required path handles remain held.
An error here rejects and rolls back; no usable lease is exposed.

Use `with store.integrate(source, context) as result:` for usable outcomes, or
explicitly close the returned lease in `finally`. Completion releases all held
resources; repeated successful close is a no-op. CLOSED disallows new usable-state
access and promises no continuing filesystem immutability. Nonacceptance and
provenance-only outcomes remain plain `Result` objects with no caller-owned lease
and no usable state. Provenance-only protection is released internally.

If release fails, accepted evidence is preserved, `ReleaseFailure.committed`
identifies irrevocable acceptance, and a usable lease enters RELEASE_FAILED.
State access/re-entry is disallowed; retry `lease.close()` (or the exception's
`cleanup.close()` for internal cleanup) to release remaining resources. The error
retains explicit ownership even when an underlying close reports failure after
closing its handle. A pre-commit cleanup error has `committed=False`, exposes no
usable state, and retains the cleanup owner for deterministic retry. Context exit
uses the same semantics, including when the caller body raises.

No background release, timeout, or custom finalizer is used. Correctness requires
explicit completion, never garbage collection or interpreter shutdown. Platforms
without mandatory locking or hard-link support fail closed. The receipt binds the
precise committed document and source snapshot throughout the OPEN lease.

## Verification boundaries

Dedicated tests cover all seven families, all three uses (normalizer continuation
explicitly rejected), five outcomes, adversarial schema/context/role/lineage and
normalizer inputs, source mutation at three points, failed atomic publication,
unchanged caller state, mapping errors and fresh-process import hygiene.
Blocker regressions additionally cover synchronized last-boundary source/stage
mutations, equal-size changes, recomputed metadata, correctly hashed invalid
continuation controls across all six model families, supported/unsupported
provenance and mapping resolution, evolved continuation and stage transitions.
In-memory weakening challenges demonstrate that these regressions detect the
original defects without editing candidate bytes.
Lease regressions additionally challenge writes, equal-size changes, deletion,
rename/replacement and metadata changes after evidence publication and after
return, cleanup/release failures, caller exceptions, and deliberately premature
release of either/both required handles. After-close writes are positive controls.

No historical artifact was needed. Actual historical availability, authoritative
provenance, production decoding and cross-runtime numerical equivalence remain
artifact-dependent and are deliberately not asserted. No closed API change is
required for this bounded candidate.
