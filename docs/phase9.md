# Phase 9: bounded sealed-evaluation and robustness infrastructure

> **Publication context:** This document is a historical phase-time technical record within the final published Phase 0–11 research-code snapshot. Candidate, uncommitted, review-pending, and later-phase-not-implemented statements describe that phase boundary, not the current repository status; they are retained for auditability. Current status is recorded in the [final Phase-11 freeze](phase11.md#final-research-code-freeze-accepted-limitations-and-verification-history) and [repository overview](../README.md). This framing neither changes the contracts or execution limits below nor upgrades historical verification claims.

This candidate implements mechanism-level REFERENCE_FIXTURE execution only.
Production sealed evaluation, robustness, population construction, normalizer
fitting, training, PhysicsTM execution and historical checkpoint loading remain
unavailable. No classifier-to-localizer cascade, artifact migration (Phase 10),
or manuscript-result regression (Phase 11) is implemented.

## Configuration and noise

`EvaluationConfig.sha256` binds the canonical evaluation descriptor and the closed
Phase-2 configuration hash. `production_plan()` is an immutable mapping of
non-callable descriptive values. Production seed literals are metadata only.

The four independent study robustness seeds are classifier 20261907, S1 20261817,
S2 20262007 and S3 20262207. The closed Phase-2 SeedSequence tuple is
`(master_seed, sample_index, realization_index, component)`, with PCG64DXSM and
30 float64 draws/component. Components are E-real, E-imaginary, H-real,
H-imaginary. Rows are zero-based immutable population positions; realizations
are 0..9. Neither SNR nor attempt ID enters the address. The same directions
are scaled at 40, 30, 20, 15, 10 dB, using closed Phase-2 mathematics.

The fixture seam accepts only seeds 0..9999 and row indices 0..15, rejecting
outside this disjoint namespace before any RNG construction. No historical
SHA-keyed adapter or historical robustness-byte reproduction claim exists.

## Primary prediction contract

Bundles contain 1..16 fixture rows, unique non-empty configuration identifiers,
exact integer row coverage, a frozen row-order digest, study/task, evaluation
configuration, BEST checkpoint role/digest, normalizer, population, clean-field,
repository and CPU/float64 environment identities. These are declared fixture
identities, not attestations that historical artifacts were loaded.

Classifier bundles store true counts 1..5, full float64 (N,5) logits and predicted
counts equal to first-maximum argmax+1. Localization bundles contain exactly S
active slots: target rho/phi, physical predicted rho/phi, and original active
cos_like/sin_like. The closed decoder checks predicted phi. Inactive slots are
structurally excluded. Predicted slots are not sorted, angular outputs are not
unit-normalized, and raw pre-sigmoid radii are unnecessary. Numeric arrays are
finite and copied into immutable byte-backed arrays. Identifiers have separate
non-numeric validation. Row indices/labels are int64, coordinates/logits float64.

Robustness conditions bind seed, SNR, realization, Phase-2 hash, addressing,
component order, NumPy version, scaling identity, row-order/clean-field hashes
and environment. Raw noisy fields are not persisted or reopened by reporting.

## Offline metrics

Closed classifier metrics provide accuracy, truth-row confusion matrix,
per-class precision/recall/F1, macro-F1 and additional cross-entropy. Closed
localization metrics provide pooled canonical Cartesian mean/median/RMSE/p95/
p99/max, radial and wrapped angular statistics. Phase 9 adds configuration
d_max summaries. S2/S3 retain optimal-matching Cartesian diagnostics and change
counts/fractions; matching never replaces principal fixed-slot results.

Ten complete, distinct realization records of identical bindings/seed/SNR are
required before aggregation. Scalar and matrix elements receive arithmetic means
and sample SD (ddof=1); all individual records remain available. There is no
pooled-error replacement of realization RMSE/percentiles or pooled-confusion F1
replacement. Historical bins/threshold tables are not mandatory or implemented;
future optional reports must read persisted outputs only.

## Persistence and state

`FixtureEngine` takes a trusted tiny deterministic test producer, not a model,
checkpoint path or population loader. The callback reports completed batch sizes.
It must return a complete typed bundle. This seam is not a security sandbox for
arbitrary Python callbacks and is not a production execution authorization.

Order: validate fixture authorization and predecessor; exclusive study claim;
durable authorization/start records; fixture producer; completeness validation;
immutable primary write; atomic primary commit marker; reopen/hash/identity
verification; metrics on the reopened bundle; immutable report; single SUCCESS
transition commit. Markers use hard-link publication after file flush/fsync,
on one filesystem supporting atomic exclusive hard-link creation. Unsupported
storage fails closed; no overwrite fallback is used. Staging names differ from
payload names even on case-insensitive Windows filesystems.

File hashes and canonical metadata hashes are separately recorded. Readers
ignore unpublished primary staging and treat no report as official without a
SUCCESS marker. Official reads verify report and primary identities. SUCCESS is
the transition commit point; scientific state remains FROZEN on clean failure,
SEALED_CLEAN on robustness failure. The latter advances only after all 50 tiny
fixture condition records and their aggregates are committed. These are fixture
simulations, not production traversals. CLOSED rejects fresh execution.

Attempts, calls, rows, inference-complete traversals, verified bundles and
official evaluations are separate accounting fields. Execution count is marked
uncertain while the producer runs. Completed batches are durably journaled;
partial work is not represented as zero. An I/O failure can prevent an incomplete
receipt too; the durable STARTED record and unreleased claim still prevent replay.
Abrupt process loss may leave unknown work after the last journal entry.

Failures retain claims, records and orphan bundles; no retry/resume is automatic.
An explicit fixture operator action `approve_new_attempt` requires a different
authorization identity/digest and attempt ID, unchanged bindings and predecessor.
It records immutable reconciliation/authorization evidence before releasing the
claim. It does not reuse partial results. Previously completed production work
is outside this bounded interface. Stale durable predecessors and concurrent
duplicate claims are rejected. Receipts enforce discipline, not authentication.

Power-loss durability depends on the underlying filesystem; absence or corruption
of terminal evidence is never inferred to mean success or permission to rerun.
There is no automatic filesystem repair, cleanup, or model traversal on read.

## Acceptance boundary

Tests use tiny synthetic arrays and nonproduction seeds, framework/solver/loader
sentinels, fault injection at nine lifecycle boundaries, actual persistence
failures, concurrency and fresh-process import probes. Class A requires exact
metadata/address/count identity. Existing Class-B primitive tolerances apply.
No Class-C tolerance, historical result equality or scientific reproduction is
claimed. Candidate changes remain unstaged/uncommitted pending independent review.

## Narrow independent-review corrections

Official reads now reconcile SUCCESS against the registered authorization,
STARTED, OWNERSHIP, complete committed primary/report evidence, producer-output
receipts and batch journals. A separately written transition witness binds every
success field, including timestamps and accounting. Before a subsequent attempt
can start, the same complete predecessor verification runs without invoking a
producer or noise RNG. Corrupt/missing evidence is not treated as a valid tip.

A nonzero declared predecessor digest must resolve to exactly one fully verified
SUCCESS receipt for the same study and frozen bindings, with a compatible after
state, and must remain the unique current tip. An empty marker collection or
unrelated success receipts cannot satisfy that requirement. Rejection occurs
before claim acquisition or start-record writes. The existing all-zero fixture
genesis convention is unchanged; it does not waive checks of existing successes.

A persistent per-study lock file carries an OS-held exclusive lock for the entire
attempt (Windows nonblocking byte-range lock; POSIX flock). It is never unlinked.
The durable claim remains after failure, while the OS lock is released on return
or process death. Retry approval must first acquire exclusive OS ownership; an
active/reentrant producer therefore cannot be declared interrupted. Explicit
quiescence evidence is recorded with the new authorization. Final publication
revalidates live ownership, durable ownership/start bindings and predecessor
state immediately before linking SUCCESS. A stale writer cannot commit after
ownership transfer, and valid predecessor state permits at most one successor.

Validated producer output receives a canonical content digest immediately on
return, before post-producer hooks. Staging and reopened verification must match
that receipt. This detects later slot reordering or angular-vector normalization;
it does not assert provenance for arbitrary data fabricated inside the trusted
fixture producer itself. Original active-slot order/magnitudes remain unchanged.

Configuration validation now requires exact frozen Python types recursively as
well as values. The canonical default configuration hash is unchanged. These are
integrity/isolation corrections only; no closed scientific implementation, noise,
metric, decoder, matching or aggregation definition is changed.
