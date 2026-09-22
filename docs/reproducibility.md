# Reproducibility

Phase 0 establishes deterministic canonical JSON hashing, typed provenance schemas, and lifecycle receipts. Class A uses exact identity. Class B defaults to `rtol=1e-12`, `atol=1e-14`. Class C PhysicsTM and full-model tolerances remain unresolved pending evidence-based characterization; no tolerance is invented here.

## canonical-json-v1

Canonical JSON is UTF-8 with NFC-normalized strings and keys, lexicographically sorted keys, and no insignificant whitespace. Mapping keys must be strings and collisions after NFC normalization are rejected. `null`, booleans, integers, and floats remain distinct typed values: integer `1`, float `1.0`, and boolean `true` have different identities.

Only finite binary64 floats are accepted. Floats use 17 significant decimal digits (`.17g`), a lowercase unpadded decimal exponent, and an explicit `.0` when a non-exponential float would otherwise resemble an integer. Positive and negative zero both encode as `0.0`. Fixed byte vectors test this contract on supported Python versions (Python 3.11 and later).

Logical paths taking part in portable configuration identity must use nonempty repository-relative POSIX syntax. Absolute paths, backslashes, empty segments, `.` segments, and `..` segments are rejected. Environment-specific paths belong only in non-portable provenance fields.

Every string value in `ScientificConfig` is recursively checked, independent of its field name. Values syntactically identifiable as Windows drive-absolute, UNC, or POSIX absolute filesystem paths are rejected before hashing. Ordinary labels, semantic identifiers, DOI/URN strings, and URL-like values remain representable. Run-manifest environment provenance may record local execution paths because it is explicitly distinct from portable configuration identity.

Supported Phase-0 schemas are exactly `scientific-config/1.0`, `scientific-run-manifest/1.0`, `artifact-registry/1.0`, `authorization-receipt/1.0`, and `canonical-json-v1`. Unknown versions fail closed.
