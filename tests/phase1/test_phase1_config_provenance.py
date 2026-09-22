import hashlib,subprocess,sys
import pytest
import numpy as np
from inverse_em.config import PHASE1_SEEDS,Phase1ScientificConfig
from inverse_em.provenance.phase1 import PhysicsPin,ReferenceFixtureProvenance
def test_config_identity_and_sealed_metadata_only():
    c=Phase1ScientificConfig();assert len(c.sha256)==64 and c.surrogate_grid.count==72 and c.inverse_grid.count==30
    assert PHASE1_SEEDS["s3_sealed"].seed==20262206
def test_fixture_provenance_complete():
    p=ReferenceFixtureProvenance.create("unit","a"*64,"b"*64,"c"*64,"d"*64);assert len(p.sha256)==64 and p.physics_pin==PhysicsPin() and p.dtype=="complex128"
@pytest.mark.parametrize("bad",["a"*63,"a"*65,"A"*64,"g"*64,1,None])
def test_fixture_provenance_rejects_invalid_sha256(bad):
    with pytest.raises(ValueError):ReferenceFixtureProvenance.create("unit",bad,"b"*64,"c"*64,"d"*64)
def test_phase1_seed_registry_is_immutable():
    with pytest.raises(TypeError):PHASE1_SEEDS["s1_sealed"]=None
def test_import_rng_safety_subprocess():
    code="import numpy as n;n.random.seed(123);a=n.random.get_state();import inverse_em.physics,inverse_em.populations;b=n.random.get_state();assert n.array_equal(a[1],b[1]) and a[2:]==b[2:]"
    subprocess.run([sys.executable,"-c",code],check=True)
