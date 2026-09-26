# Project governance and reference-document register

## Public repository context

The governing specifications and authorization documents referenced below governed development where stated; they are not necessarily distributed with this public repository. This register does not reconstruct them or imply that a public reader has independently verified unavailable documents. Readers can inspect the committed implementation and [public phase records](scientific_method.md).

Current publication status is represented by the committed repository and the [final Phase-11 research-code freeze](phase11.md#final-research-code-freeze-accepted-limitations-and-verification-history), subject to its documented accepted limitations. Descriptive receipts, hashes, seeds, and provenance evidence are not authentication or execution authority. Publication does not authorize scientific execution or establish production reproduction.

## Historical development authority and document register

The authority order and register below are retained as the development record. Phase-0 inclusion statements and the prospective public-registry sentence describe that historical context, not a promise of further work or a claim that those documents are now bundled.

Authority order:

1. **Scientific & Software Specification v1.0**, as clarified by the Blueprint Resolution Addendum v1.0.
2. Frozen and persisted scientific evidence.
3. Verified final implementation and configuration.
4. Repository audit and historical documentation.
5. Development and historical code.

Design inputs:

| Document | Role | Inclusion |
|---|---|---|
| Scientific & Software Specification v1.0 | Frozen scientific contract/original decision record | Referenced; not copied into this repository during Phase 0 |
| Repository Blueprint & Implementation Plan v1.0 | Approved software blueprint | Referenced; not copied |
| Blueprint Resolution Addendum v1.0 | Project-owner resolution of design questions | Referenced; not copied |
| Blueprint Resolution Report v1.0 | Verified implementation clarifications and Phase-0 gate | Referenced; not copied |

Source-document hashes are intentionally not asserted here because not every governing document exists as a stable standalone source file. A future authorized clean specification edition may add a complete public document registry.

## Release version

`src/inverse_em/_version.py` is the authoritative software-version source. Runtime and package metadata derive from it. `CITATION.cff` necessarily repeats release metadata for citation tooling; the Phase-0 consistency test must pass before a release or baseline update.
