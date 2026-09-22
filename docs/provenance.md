# Provenance

Scientific manifests use exact supported schema versions and typed immutable subrecords. They distinguish `PRESENT`, `MISSING`, `NOT_APPLICABLE`, and `UNRESOLVED`; every non-present value carries a reason. BEST and TERMINAL checkpoint metadata are separate qualified fields.

Canonical identities use the versioned `canonical-json-v1` contract and SHA-256. Artifact identity is independent of hosting location. Multiple logical IDs may alias identical bytes only when byte size, format, and applicable artifact-schema identity agree; contradictory metadata for one digest is invalid.

Portable scientific identities use logical, repository-relative POSIX paths. Machine-specific paths may be recorded as environment provenance, but they are not portable artifact identities.

Authorization receipts for `TRAINING → FROZEN`, `FROZEN → SEALED_CLEAN`, and `SEALED_CLEAN → ROBUSTNESS` require `PRESENT` BEST-checkpoint, population, and normalizer identities. `MISSING`, `UNRESOLVED`, and `NOT_APPLICABLE` cannot substitute for those bindings.
