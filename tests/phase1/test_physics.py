import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from inverse_em.physics import AngleGridConfig,PhysicsTMConfig,PhysicsTMForward,Source,observation_angles,vendored_source_sha256,verify_vendored_source
from inverse_em.physics.vendor import UPSTREAM_COMMIT,UPSTREAM_SOURCE_SHA256

def digest(*arrays):return hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)).hexdigest()
def test_pin():
    assert UPSTREAM_COMMIT=="a079c3899bd33f636ab9c8b1571d8e919c839335"
    assert UPSTREAM_SOURCE_SHA256=="b0d8200ef30d05720f0b697822dc0e037935243beea9084edf410c9725c32005"
    assert vendored_source_sha256()==UPSTREAM_SOURCE_SHA256 and verify_vendored_source()
def test_license():
    text=(Path(__file__).parents[2]/"src/inverse_em/physics/vendor/LICENSE").read_text()
    assert "MIT License" in text and "Copyright (c) 2026 Thomas D. Papadopoulos" in text
def test_constants_and_modes():
    c=PhysicsTMConfig();assert (c.radius,c.omega,c.eps0,c.mu0,c.eps1,c.mu1,c.amplitude,c.modal_order,c.dtype)==(1,4,1,1,1.3225,1,1,20,"complex128")
    assert np.array_equal(c.modal_indices,np.arange(-20,21,dtype=np.int64))
@pytest.mark.parametrize("count",[30,72])
def test_endpoint_excluded_grid(count):
    x=observation_angles(count);assert x.dtype==np.float64 and x[0]==0 and x[-1]<2*np.pi and np.array_equal(x,2*np.pi*np.arange(count,dtype=np.float64)/count)
def test_bad_endpoint_rejected():
    with pytest.raises(Exception):AngleGridConfig(30,endpoint=True)
def test_source_cartesian_and_validation():
    s=Source(.5,np.pi/2);assert s.cartesian[0]==pytest.approx(0,abs=1e-16) and s.cartesian[1]==pytest.approx(.5)
    with pytest.raises(Exception):Source(.2,.1,.5)
def test_determinism_dtype_and_fixture():
    f=PhysicsTMForward();a=f.evaluate(Source(.25,.75),observation_angles(30));b=f.evaluate(Source(.25,.75),observation_angles(30))
    assert a.electric.dtype==a.magnetic.dtype==np.complex128 and np.array_equal(a.electric,b.electric) and np.array_equal(a.magnetic,b.magnetic)
    assert digest(a.electric,a.magnetic)=="fc31e917eb2eb55654bf7ebb911c3a7c26ff515f9fb05de96c33a4f4fbc367d1"
    manifest=json.loads((Path(__file__).parents[1]/"reference/physics_tm_reference_v1.json").read_text())
    assert manifest["fixture_sha256"]==digest(a.electric,a.magnetic) and manifest["dtype"]=="complex128"
    for value in (a.electric,a.magnetic):
        with pytest.raises(ValueError):value.setflags(write=True)
    changed=a.electric.copy();changed[0]=0;assert not np.array_equal(changed,a.electric)
def test_physics_does_not_mutate_global_rng():
    np.random.seed(123);before=np.random.get_state();PhysicsTMForward().evaluate(Source(.2,.3),observation_angles(30));after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
def test_no_phase2_imports_in_physics():
    text=(Path(__file__).parents[2]/"src/inverse_em/physics/api.py").read_text()
    assert all(word not in text.lower() for word in ("torch","noise","normalization","surrogate"))
