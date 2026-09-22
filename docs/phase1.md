# Phase 1: PhysicsTM and deterministic populations

Phase 1 implements only the pinned analytical PhysicsTM boundary and deterministic source-population construction. It does not implement measurement noise, normalization, surrogates, classifiers, localizers, losses, training, inference, sealed evaluation, or robustness evaluation.

## Physics identity

The immutable vendored core is the exact `src/inverse_source_em/physics/physics_tm.py` file from `th0ma/inverse-source-em` commit `a079c3899bd33f636ab9c8b1571d8e919c839335`, SHA-256 `b0d8200ef30d05720f0b697822dc0e037935243beea9084edf410c9725c32005`, distributed under the MIT License, Copyright 2026 Thomas D. Papadopoulos. Runtime verification rejects modified vendor bytes. The local API identity is separate from this upstream identity.

The normalized constants are `R=1`, `OMEGA=4`, `eps0=mu0=mu1=I0=1`, `eps1=1.3225`, `N=20`, modes `-20..20`, and `complex128`. Observation grids use `2*pi*arange(n)/n`; frozen counts are 72 and 30.

## Population identity

All Phase-1 population generation uses an owned `numpy.random.Generator(numpy.random.PCG64DXSM(seed))`; imports and physics calls initialize no RNG. Radius sampling is uniform in squared radius on a half-open annulus and azimuth is uniform on `[0,2*pi)`. Sources have unit amplitude.

The surrogate population law is 10,000 sources on `[0.01,0.99)`, with deterministic source-level 7,000/1,500/1,500 hash partitions. Historical final surrogate source generation used `np.random.default_rng(seed)`, and therefore NumPy PCG64. The frozen new reference implementation intentionally uses PCG64DXSM and does not claim byte-identical reproduction of the historical surrogate source sequence.

Classifier classes 1–5 are balanced and retain proposal order without separation constraints. S1 uses `[0.05,0.95)`. S2 redraws whole pairs, requires radial 0.05 and wrapped angular 5-degree separation, then orders by exact `(rho, phi)`. S3 redraws whole triples; every pair must pass the selected eight-stage curriculum constraint before exact `(rho, phi)` ordering.

Frozen typed identities bind each final role to its authoritative seed, count, source count, and stage where applicable. A centralized protected-identity registry rejects every final sealed seed under any non-sealed role and requires explicit authorization under the sealed role before RNG initialization. Classifier `proposal_ordinal` is one-based across the shared continuous whole-configuration proposal stream; it is new reference-repository provenance, not a claimed historically persisted identity.

Boundary acceptance is `value >= threshold or isclose(value, threshold, rel_tol=0, abs_tol=8*spacing(float64(threshold)))`. Each S3 stage derives its own radial and angular tolerance. Proposal ordinals identify accepted configurations without retaining full rejection histories.

Sealed seed values are metadata. Public generators refuse sealed roles unless a separately authorized caller explicitly opts in; Phase-1 tests use dedicated fixture seeds only.

## Numerical limitation

Class-C floating tolerances remain unresolved. Deterministic same-environment fixtures use exact bytes and digests. Phase 1 does not promote a cross-platform numerical tolerance or claim cross-platform bitwise identity.
