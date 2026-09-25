import pytest
from inverse_em.artifact_integration.planning import MigrationPlan, validate_migration
from inverse_em.artifact_integration.contracts import Domain, Identity, checked


def test_mapping_bijection_and_signed_zero():
    plan = MigrationPlan("state_key_remap", (("a", "b"),), (("a", "radius"),), (("b", "radius"),))
    assert validate_migration(plan, {"a": -0.0}, {"b": -0.0}) == plan.identity
    with pytest.raises(ValueError):
        validate_migration(plan, {"a": -0.0}, {"b": 0.0})


@pytest.mark.parametrize("mutation", ["semantics", "collision", "operation"])
def test_mapping_weakening_rejected(mutation):
    operation = "optimizer_reset" if mutation == "operation" else "state_key_remap"
    pairs = (("a", "b"), ("a", "c")) if mutation == "collision" else (("a", "b"),)
    destination = (("b", "angle" if mutation == "semantics" else "radius"),)
    with pytest.raises(ValueError):
        MigrationPlan(operation, pairs, (("a", "radius"),), destination)


def test_hash_text_does_not_resolve_provenance_namespace():
    source = Identity(Domain.SOURCE, "synthetic-source-bytes/1", "a" * 64)
    with pytest.raises(ValueError):
        checked(source, Domain.NATIVE)


def test_lease_lifecycle_with_detached_handles():
    from inverse_em.artifact_integration.integration import ProtectedResult
    # Actual publication/protection probes remain the unchanged Phase-10 tests.
    # Assert the caller-completion interface without opening any artifact.
    assert callable(ProtectedResult.close)
    assert callable(ProtectedResult.__enter__) and callable(ProtectedResult.__exit__)
    assert "__del__" not in ProtectedResult.__dict__


@pytest.mark.parametrize("fails", [False, True])
def test_lease_explicit_completion_and_release_failure(fails):
    from inverse_em.artifact_integration.integration import ProtectedResult
    class Resources:
        def __init__(self):
            self.calls = 0
        def close(self):
            self.calls += 1
            if fails and self.calls == 1:
                raise OSError("detached synthetic release fault")
    resource = Resources()
    marker = object()
    lease = ProtectedResult(None, marker, resource)
    assert lease.state == "OPEN" and lease.accepted is marker and resource.calls == 0
    if fails:
        with pytest.raises(OSError):
            lease.close()
        assert lease.state == "RELEASE_FAILED"
        with pytest.raises(RuntimeError):
            _ = lease.accepted
    lease.close()
    assert lease.state == "CLOSED"
    calls = resource.calls
    lease.close()
    assert resource.calls == calls
    with pytest.raises(RuntimeError):
        _ = lease.accepted


def test_handoff_keeps_both_required_handles_and_detects_weakening(tmp_path):
    import io
    from inverse_em.artifact_integration.integration import _protected_handoff
    def exercise(premature):
        stage = tmp_path / "stage"
        destination = tmp_path / "accepted"
        stage.write_bytes(b"envelope")
        destination.write_bytes(b"envelope")
        class Resources:
            handles = [io.BytesIO(b"source"), io.BytesIO(b"envelope"), io.BytesIO(b"envelope")]
            paths = [stage]
        r = Resources()
        try:
            _protected_handoff(r, b"source", b"envelope", destination)
            if premature:
                r.handles[2].close()  # In-memory vulnerable-behavior challenge.
            assert not r.handles[0].closed and not r.handles[2].closed
            assert r.handles[1].closed and not stage.exists()
        finally:
            for h in r.handles:
                h.close()
    exercise(False)
    with pytest.raises(AssertionError):
        exercise(True)


@pytest.mark.parametrize("field,value", [("role", "SEALED"), ("checkpoint_role", "TERMINAL"),
    ("configuration", "0" * 64), ("environment", "GPU/float32")])
def test_evaluation_binding_substitution(field, value):
    from dataclasses import replace
    from inverse_em.config.evaluation import EvaluationConfig
    from inverse_em.evaluation.contracts import Bindings
    b = Bindings("phase11-fixture", "s3", EvaluationConfig().sha256, "1" * 64,
                 "2" * 64, "3" * 64, "4" * 64, "5" * 64, "fixture", "CPU/float64;fixture")
    with pytest.raises((ValueError, PermissionError)):
        replace(b, **{field: value})
