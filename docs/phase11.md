# Phase 11: final regression and reproducibility

> **Current publication status:** This is the final published/frozen bounded research-code snapshot, closed with documented accepted limitations. The [final freeze record](#final-research-code-freeze-accepted-limitations-and-verification-history) governs its current status. Earlier candidate language, closure preconditions, failed checks, and the FAILED independent closure-readiness verdict remain below as historical evidence. The manifest-authority limitation was accepted, not repaired; publication is not production scientific reproduction.

Status: bounded implementation candidate; independent verification and separate
closure/publication authorization are required. No production reproduction claim.

## Baseline and ownership

The sole baseline is commit adf186647476c34f62aeebff0753af747540c067,
tree 4b7cc9bd61556856312c3412f2b0b98fc3f01d22. All tracked baseline files
are pinned by committed-byte SHA-256 in tests/phase11/contract_ledger.json.
No Phase 0-10 file is changed. This phase adds tests, a test-only accounting
plugin, and documentation; no production package, CLI or execution adapter.

## Execution-boundary resolution

The unchanged closed tests may perform exactly their existing historical/reference
operations. This is GRANDFATHERED_CLOSED_PHASE_REGRESSION, not synthetic execution
and not zero execution. It includes the existing historical coordinate generation,
bounded solver calls, seeded initialization and RNG/order reconstruction. Permission
attaches to the unchanged committed test behavior, not an unrestricted callable.
No new Phase-11 test reuses the historical population generator or old test helpers.

Phase-11 fixtures are narrower than closed owner limits: metadata/hash checks,
scalar replay, explicit CPU float64 tensors, explicit normalizer statistics,
schema/role/mapping checks and detached synthetic lease resources. No RNG draws,
model construction, training, fitting, checkpoint deserialization or PhysicsTM
execution are needed. Actual bounded trainer resume and OS publication races
remain exercised by the unchanged owner tests, not duplicated as new authority.

Production population generation/traversal, training, normalizer fitting,
historical checkpoint access/migration, production integration, sealed inference,
robustness inference and operational production RNG remain prohibited.

## Identity and cross-phase contract

The ledger freezes all nine canonical configuration identities. Tests separately
check upstream Phase-1/2/5 references, E/H deployment identities, Phase-10 family
configuration bindings, state schemas, active slots and task-specific boundaries.
Historical file hashes, canonical configuration hashes, tensor-state hashes and
native artifact hashes are different identity domains, even if digest text agrees.
Provenance is descriptive evidence, never authentication or execution authority.

The dependency chain is Phase 0 identity/lifecycle -> Phase 1 physics/populations
-> Phase 2 observations/noise/normalization; Phase 3 supplies frozen relative-angle
E/H surrogates; Phase 4 classifier uses Phase 1/2; Phase 5 common localization
uses Phase 3; Phases 6/7/8 specialize S1/S2/S3; Phase 9 binds evaluation to closed
metrics/configurations; Phase 10 binds synthetic artifacts to closed identities.

Phase-2 four-channel clean-training ddof=0 statistics remain authoritative. S3
uses all eight stages and 30-angle block moments, not historical scalar Welford.
No historical scalar-by-scalar normalizer equivalence is claimed.

Localizers expose first S active slots of three internal slots, with unchanged
0.05+0.90*sigmoid and atan2 decoding. No projection, clipping or post-sorting.
S1 averages one slot; S2 sums slots then averages batch; S3 averages batch/source
with (1+rho)/1.635 on radial/angular components only. Circle remains unweighted.
Canonical physical Cartesian metrics pool sources; matching is diagnostic only.
FC remains four-channel clean raw analytical target loss, coefficient .03, with
four Kendall scalars. Synthetic metrics do not produce scientific results.

S1 BEST follows scheduling; S2/S3 BEST precedes scheduling. Continuation is a
completed post-scheduler boundary, with S3 stage/global clocks and reset semantics.
BEST and TERMINAL are not automatically resumable. Phase-10 interchange is not
the native production trainer serialization. OPEN protected-result leases require
explicit completion; CLOSED promises no continuing filesystem immutability.
RELEASE_FAILED preserves explicit cleanup ownership. Arbitrary unregistered aliases
do not acquire perpetual protection.

## Equality ledger

- Exact: canonical bytes/digests/types, schema domains, dependency identities,
  integer clocks, roles, frozen deterministic arrays and continuation states where
  their owner requires exact equality. Same-environment is not cross-runtime proof.
- Class B: rtol=1e-12, atol=1e-14 for already-authorized primitive comparisons.
- Population boundaries: >= or rel_tol=0 and 8*spacing(float64(threshold)); radians
  before angular tolerance. No replacement by NumPy default isclose.
- Owner-scoped engineering exceptions: Phase-1 area transform (0,2e-16); Phase-3
  features (0,1e-16), periodicity (1e-12,1e-12), gradient cancellation (0,1e-12),
  finite-difference step1e-6 with (1e-6,1e-8); Phase-4 cyclic (2e-14,2e-14).
- Existing normalization tests use atol2e-14/2e-13 with NumPy rtol1e-5. Existing
  pytest.approx defaults are rel1e-6/abs1e-12; the noise-power sanity test uses
  rel.025. These are retained in their tests, not generalized acceptance criteria.
- Historical scalar replay is evidence only. S3 LR replay retains Class B;
  strict BEST events and clocks remain exact. No Class-C training tolerance exists.

## Verification matrix

| Phase | Evidence / check | Equality / boundary |
|---|---|---|
| 0 | canonical, schemas, lifecycle, registry tests | exact identities; no authority from receipts |
| 1 | vendor/reference hashes, grids, laws, roles | exact fixture and eight-ULP acceptance; grandfathered owner execution |
| 2 | channel/noise/normalization config and tests | exact addressing; scoped numerics; no new fit |
| 3 | configuration/source/split/state schemas | exact identities; grandfathered historical fixtures only |
| 4 | classifier model, selection, RNG/checkpoint tests | owner limits 100 cases/two epochs; no new initialization |
| 5 | active slots, decoder, reductions, metrics | synthetic tensors; exact/scoped Class B |
| 6 | S1 bindings, BEST/continuation and reference | owner <=16 cases/four cumulative epochs; historical descriptive |
| 7 | S2 boundaries and 392-row scalar history | exact replay; owner continuation unchanged |
| 8 | S3 config and 400-row scalar history | exact events/clocks, Class-B LR; owner two stages/two epochs each |
| 9 | typed bindings, persistence, predecessor/ownership tests | owner 1..16 rows/fixture seeds; no scientific inference |
| 10 | schemas, mapping, lease and owner race regressions | exact state/bytes and explicit resource lifetime |

Disagreement is failure, insufficient evidence or a material gap, never a reason
to loosen an assertion, edit a closed API, substitute a scientific identity or
repair historical evidence.

## Resource accounting and limitations

Use the test-only plugin phase11_accounting. It validates HEAD/tree and every
closed committed file before testing, requires no tracked changes, and classifies
items by baseline-owned path rather than selected function or seed. The prior
suite is not patched, deselected or substituted. Its Python mechanism entrypoints
are observed using sys.setprofile; the previous profile hook is restored.

Five separate categories are retained:

1. GRANDFATHERED_CLOSED_PHASE_REGRESSION
2. PHASE11_BOUNDED_FIXTURE
3. PROTECTED_ATTEMPT_REJECTED_BEFORE_EXECUTION
4. UNAUTHORIZED_PROTECTED_EXECUTION
5. UNKNOWN_OR_PARTIAL_EXECUTION

Event-backed aggregates explicitly report units: successful pytest items, rejected calls, and
unknown/failed items. Mechanism entries are separately counted by category.
They are NOT optimizer updates, RNG variates, population rows or completed
scientific traversals. Profile observations are in-process; subprocess activity
belongs to the unchanged owning test contract and is not assigned invented
per-call counts. A failed/skipped/interrupted item is unknown, not zero. A nonzero
session exit also records incomplete session evidence. Abrupt process death may
leave no receipt: a missing receipt is UNKNOWN and blocks acceptance.

Phase-11 autouse guards reject production/generator/fitter/loader aliases in
loaded inverse_em modules, RNG constructors, model initialization and solver
entrypoints before execution. They apply only to Phase-11 tests and are restored
before prior tests. This is an accidental-reachability regression firewall, not
a sandbox against hostile code replacing guards or private interpreter state.
Sentinel and counter mutation challenges use detached objects and never execute
the prohibited body. Historical same-function reuse is explicitly rejected.
Profiling is observational, not an exception-raising enforcement boundary.
CPython disables a profile callback when it raises; it is never used to reject.
For protected Python frames its pre-body observation is reconciled with the
monitoring rejection using one attempt identity. An unreconciled return becomes
UNKNOWN, not an assumed zero. Grandfathered entries retain their original units.

### Bounded execution-surface contract v1

Phase 11 provides fail-closed admission for explicitly enumerated mechanisms
available to its owned verification code. It is not a hostile-code sandbox and
does not claim universal interception of arbitrary native implicit dispatch.
The contract is recorded separately in the ledger without changing its 185-file
closed inventory or nine scientific configuration hashes.

Admitted mechanisms are ordinary Python calls, supported native direct calls,
the named `admitted_lengths` callback operation, unchanged grandfathered owner
tests, and the existing explicit metadata/scalar/tensor/detached-resource fixtures.
The callback operation accepts only an exact list/tuple of exact str/bytes and
uses a fixed built-in len callback. It accepts no callable, generic dispatcher,
generator, custom sequence, or opaque iterator. Rejection precedes consumption.
Accounting for this operation comes only from the owned Phase-11 lifecycle.
There is no accepted audit parameter or caller-supplied executable dependency.
Extra positional/keyword arguments are captured solely to record rejection,
without inspecting or invoking their values (including attribute lookup).
This rejection-only envelope is not a callback extension mechanism.
Raw map/filter/reduce/partial, callable-iter, sorted/min/max key dispatch,
list.sort and operator callback adapters from owned Phase-11 code are unsupported;
the accounting snapshot's fixed internal string-key sort is separately admitted.
the one named operation owns its fixed map invocation. No general-purpose
callable executor or production authority is exposed. Pytest-owned fixture hooks
and fixed fresh-process import/inert probes are verification lifecycle mechanisms,
not a source of scientific authorization.

Persistent `sys.monitoring` CALL/PY_START guards use pre-sentinel callable/code
identities, covering local, imported and rebound aliases without relying on a
module-global sweep. Installation occurs only during a Phase-11 fixture lifecycle;
import is passive. Missing monitoring support or slots fails closed. Prior tests
do not receive these guards. Profile and monitoring lifecycle loss is UNKNOWN.

RNG coverage captures public callable numpy.random identities (excluding the
read-only get_state and library test entry), including RandomState, default_rng,
Generator, SeedSequence, PCG64/PCG64DXSM and other exposed bit generators/draws.
Closed RNG-addressing/noise routines and Torch initialization/seed/draw boundaries
are separately guarded. No protected production streams are drawn by probes.

A preconstructed map(len, ...) can execute during FOR_ITER without a monitored
target CALL. An isolated harmless regression records that interpreter limitation.
Such an object is outside the admitted execution surface and is rejected by
the callback admission gate without consumption. This is not a universal native
iterator firewall. Deliberately executing opaque objects outside the admitted
surface is not a supported Phase-11 capability or scientific authority.

Accounting requires exact bool completion; invalid status leaves the token
pending and records UNKNOWN. Duplicate/unknown completion also records UNKNOWN.
Private event history and transitions, not mutable public counters, determine
receipts. Public aggregate/pending views and snapshots are detached copies.
An attempt observed by multiple layers counts once; distinct attempts count
separately. Contradictory evidence records UNKNOWN and retains UNAUTHORIZED
crossings rather than relabeling them as rejection. No hostile-memory guarantee.

Import checks use fresh processes, RNG snapshots and write/network/model/RNG
sentinels. No Phase-11 production package exists; direct test-support/test-module
imports are checked, while closed package imports remain the owner tests' scope.

## Gate sequence and reproduction claims

Closure-state admission checks the exact ten candidate paths, not a blanket
empty `git diff HEAD`. All closed files must match their frozen hashes in HEAD,
the index and the worktree. Candidate files must match explicit reviewed SHA-256
identities in the worktree and, when staged, the index. Exact unstaged, staged
and mixed additions are supported; unexpected paths, missing files, altered
bytes and non-addition baseline deltas are rejected.

Every gate also requires `--phase11-candidate-manifest` (an external JSON mapping
of the ten exact paths to SHA-256 hashes) and `--phase11-candidate-sha256` (the
reviewed digest of that manifest's bytes). Both are mandatory; no runtime
self-freezing or fallback is permitted. The external input avoids hashing a
manifest containing its own identity. Review/closure must independently pin this
manifest and digest; supplying them is not scientific execution authority.
The manifest, receipts and external harnesses must not enter the repository.

Set PYTHONPATH to src plus tests/phase11, PYTHONDONTWRITEBYTECODE=1, and run Python
with -B. Every pytest command uses -p no:cacheprovider -p phase11_accounting,
an external --basetemp and unique external --phase11-accounting JSON path.

1. test_phase11_contract_firewall.py
2. complete tests/phase11
3. tests --ignore=tests/phase11 (unchanged prior suite)
4. tests (complete suite)
5. hashes, inventory, tracked diff/index, resource receipts and hygiene checks

Any implementation debugging attempts are reported separately, not hidden by a
later successful run. Acceptance requires all final gates passing with no skips,
unauthorized execution zero, unknown/partial zero, and unchanged grandfathered
scope. Independent verification must challenge literal expected identities,
callsite ownership, accounting and guard coverage, numerical reductions, scalar
replay, predecessor evidence and premature protection release. Existing owner
adversarial tests remain mandatory and unchanged.

Passing establishes bounded repository regression, identity coherence, fixture
behavior, scalar replay and the owner suites' bounded continuation/compatibility.
It does not reproduce historical production training, checkpoint compatibility,
sealed/robustness metrics or cross-runtime bitwise science. No scientific output
is written; receipts are external engineering artifacts. No staging, commit,
push, tag or later phase is authorized. Closure requires a separately reviewed
exact candidate/hash freeze with no unresolved blocker or material test gap.

## Final research-code freeze: accepted limitations and verification history

The final user authorization supersedes the earlier closure prerequisites above:
publish the reviewed implementation with documented limitations, without another
correction cycle. Only this documentary closure record is added to the reviewed
candidate. No implementation, scientific configuration or test logic is repaired.
This is closure of the research-code snapshot, not an unrestricted security or
independent scientific-reproduction certification.

### History retained at freeze

- Phases 0-10 were already closed and published at commit
  `adf186647476c34f62aeebff0753af747540c067`, tree
  `4b7cc9bd61556856312c3412f2b0b98fc3f01d22` (185 frozen files).
- Phase-11 development included failed independent checks and narrowly corrected
  enforcement/accounting gaps. It was not an uninterrupted sequence of passes.
  Persistent rejection, RNG coverage, exact-Boolean completion, event-backed
  accounting and attempt deduplication were subsequently checked independently.
- M1: the former `admitted_lengths(values, audit=None)` accepted an unchecked
  accounting dependency. A foreign `rejected` callback could run without an
  authoritative rejection event. The narrow data-only/owned-accounting correction
  was independently re-verified: zero foreign/body entries, retained rejection
  evidence, and detection of the former behavior restored in memory.
  The resulting complete repository gate passed 1,131 tests.
- The first authoritative staged-candidate closure attempt failed during
  `pytest_sessionstart`, before any tests executed, with
  `RuntimeError: Closed tracked files changed`. The blanket `git diff HEAD`
  guard misclassified authorized staged additions. Earlier unstaged passing gates
  are not that failed staged closure gate.
- The subsequent closure-harness correction checks closed HEAD/index/worktree
  bytes and exact candidate inventory/hashes. Independent staged, unstaged and
  mixed-candidate controls passed; closed-file and unexpected-candidate mutations
  were rejected. The latest independent sequence passed 15 closure-focused,
  88 direct, 145 Phase-11, 1,001 prior and 1,146 complete tests, with zero failures
  or skips (the focused selection deselected unrelated tests).
- That independent review nevertheless FAILED its closure-readiness verdict:
  changing both candidate bytes and the caller-supplied manifest/digest admitted
  a replacement candidate in an inert temporary repository. This remains
  unresolved and is explicitly accepted for this publication. Passing tests do
  not erase that finding or convert the failed independent verdict into a pass.

### Manifest authority and test-harness lifecycle limitations

The manifest guard verifies consistency against a supplied digest, not an
independent immutable authorization anchor. An unchanged externally reviewed
digest rejects changed manifests; replacing both the manifest and supplied digest
can authorize changed candidate bytes at session start. Operators must retain
the external review evidence. The CLI digest is not a self-attestation defense.
The published snapshot does not claim to resolve this limitation.

The external manifest and audit receipts are not repository artifacts. The
pre-publication accounting harness also pins HEAD/tree to the Phase-10 baseline
and expects the ten Phase-11 files as additions. Once the closure commit changes
HEAD, that same session-start guard rejects it with `Baseline HEAD changed`.
Therefore pre-publication staged-candidate passing results are not a claim that
the unchanged harness passes at the published HEAD. This repository-state
limitation is accepted without repair; it concerns verification infrastructure,
not demonstrated scientific behavior. No final test failure may be reported as
a pass. The publication report records the actual final invocation outcomes.

### Scope of guarantees

- Enforcement is bounded, not a hostile-code sandbox or universal CPython/native
  interception mechanism. Pre-existing callback-bearing native objects can
  dispatch implicitly outside monitored CALL events; the admitted API rejects
  opaque objects, but arbitrary implicit native dispatch is not universally caught.
- Event-backed accounting protects against the documented API/view failures,
  not arbitrary hostile memory mutation. Grandfathered mechanism entries are
  regression observations, not scientific rows or completed production operations.
- Phase 10 establishes bounded synthetic artifact compatibility and validation,
  not universal historical production-checkpoint or artifact compatibility.
- Protected-result guarantees depend on explicit caller completion and cover
  documented registered paths/bytes during the OPEN lease. They do not establish
  perpetual filesystem immutability.
- Regression passes do not establish historical training-trajectory reproduction,
  production checkpoint equivalence, cross-runtime bitwise scientific equivalence,
  sealed/robustness result reproduction, production migration equivalence or new
  performance measurements. Historical metrics remain descriptive evidence unless
  reproduced under a separately authorized scientific protocol.

The appropriate claim is: this is the source-code and regression baseline
accompanying the research work, containing implemented scientific methods, frozen
configurations, provenance contracts, bounded evaluation infrastructure,
artifact-integration checks and regression/reproducibility tests developed through
Phases 0-11. It supports the research work; it does not by itself prove every
reported scientific result. Known limitations are accepted in the final freeze.
