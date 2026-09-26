# Reproducibility

## Published snapshot and evidence boundary

This is the reproducibility entry point for the final published Phase 0–11 bounded research-code snapshot. The canonical serialization material below is retained unchanged as the identity/provenance foundation; its opening tolerance-status statement records the Phase-0 boundary, not a new cross-runtime guarantee.

- [Phase 1](phase1.md) and [Phase 2](phase2.md) define population identities, physics, noise, and clean-training normalization contracts.
- [Phase 3](phase3.md), [Phase 4](phase4.md), and [Phase 5](phase5.md) document model and common loss mechanisms; [S1](phase6.md), [S2](phase7.md), and [S3](phase8.md) specify bounded training, selection, and continuation.
- Historical scalar evidence and replay are described in the task records, including S2 and S3. Replay is not retraining or production checkpoint reproduction.
- [Phase 9](phase9.md) covers bounded evaluation infrastructure; [Phase 10](phase10.md) covers synthetic compatibility and explicit protected-result lifetimes.
- [Phase 11](phase11.md) records the final freeze, historical verification outcomes, accepted manifest-authority limitation, and baseline-pinned closure-harness lifecycle. Its pre-publication evidence is not a post-publication closure-suite pass.

Repository regression and bounded reproducibility do not establish full historical production-training reproduction, cross-runtime bitwise scientific reproduction, sealed/robustness result reproduction, or production migration equivalence. Descriptive identities and receipts are not authentication or execution authority. See [governance](governance.md) and [limitations](limitations.md).

## Historical Phase-0 identity foundation

Phase 0 establishes deterministic canonical JSON hashing, typed provenance schemas, and lifecycle receipts. Class A uses exact identity. Class B defaults to `rtol=1e-12`, `atol=1e-14`. Class C PhysicsTM and full-model tolerances remain unresolved pending evidence-based characterization; no tolerance is invented here.

## canonical-json-v1

Canonical JSON is UTF-8 with NFC-normalized strings and keys, lexicographically sorted keys, and no insignificant whitespace. Mapping keys must be strings and collisions after NFC normalization are rejected. `null`, booleans, integers, and floats remain distinct typed values: integer `1`, float `1.0`, and boolean `true` have different identities.

Only finite binary64 floats are accepted. Floats use 17 significant decimal digits (`.17g`), a lowercase unpadded decimal exponent, and an explicit `.0` when a non-exponential float would otherwise resemble an integer. Positive and negative zero both encode as `0.0`. Fixed byte vectors test this contract on supported Python versions (Python 3.11 and later).

Logical paths taking part in portable configuration identity must use nonempty repository-relative POSIX syntax. Absolute paths, backslashes, empty segments, `.` segments, and `..` segments are rejected. Environment-specific paths belong only in non-portable provenance fields.

Every string value in `ScientificConfig` is recursively checked, independent of its field name. Values syntactically identifiable as Windows drive-absolute, UNC, or POSIX absolute filesystem paths are rejected before hashing. Ordinary labels, semantic identifiers, DOI/URN strings, and URL-like values remain representable. Run-manifest environment provenance may record local execution paths because it is explicitly distinct from portable configuration identity.

Supported Phase-0 schemas are exactly `scientific-config/1.0`, `scientific-run-manifest/1.0`, `artifact-registry/1.0`, `authorization-receipt/1.0`, and `canonical-json-v1`. Unknown versions fail closed.
