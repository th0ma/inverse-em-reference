import numpy as np
import pytest
from inverse_em.noise import *
from inverse_em.observations import ComplexFields,complex_fields_to_channels

def fields():
    k=np.arange(30,dtype=np.float64);return ComplexFields((1+.1*k)+1j*(2-.03*k),(.2+.02*k)+1j*(-.4+.01*k))
def test_zero_direction_limit():
    z=np.zeros(30);d=StandardizedNoiseDirections(z,z,z,z,1,0,0);f=fields();n=scale_directions(f,d,20)
    assert np.array_equal(add_scaled_noise(f,n).electric,f.electric) and np.array_equal(n.magnetic,z.astype(np.complex128))
def test_fixed_identity_determinism_and_realization_difference():
    a=draw_standardized_directions(740100,2,3);b=draw_standardized_directions(740100,2,3);c=draw_standardized_directions(740100,2,4)
    for name in ("electric_real","electric_imag","magnetic_real","magnetic_imag"):assert np.array_equal(getattr(a,name),getattr(b,name))
    assert not np.array_equal(a.electric_real,c.electric_real)
def test_four_streams_are_distinct():
    d=draw_standardized_directions(740101,0,0);v=(d.electric_real,d.electric_imag,d.magnetic_real,d.magnetic_imag)
    assert all(not np.array_equal(v[i],v[j]) for i in range(4) for j in range(i+1,4))
def test_channel_order_and_dtype():
    f=fields();x=complex_fields_to_channels(f);assert x.shape==(4,30) and x.dtype==np.float64
    assert np.array_equal(x[0],f.electric.real) and np.array_equal(x[1],f.electric.imag) and np.array_equal(x[2],f.magnetic.real) and np.array_equal(x[3],f.magnetic.imag)
def test_power_and_factor_two_exact():
    f=fields();d=draw_standardized_directions(740102,0,0);n=scale_directions(f,d,30);pe=np.mean(np.abs(f.electric)**2);ph=np.mean(np.abs(f.magnetic)**2)
    assert n.electric_power==pe and n.magnetic_power==ph and pe!=ph
    assert n.electric_sigma**2==pytest.approx(pe*1e-3/2) and n.magnetic_sigma**2==pytest.approx(ph*1e-3/2)
def test_empirical_complex_noise_power_convention():
    f=ComplexFields(np.full(30,2+1j),np.full(30,.5-.25j));e=[];h=[]
    for i in range(1200):
        n=scale_directions(f,draw_standardized_directions(740103,i,0),20);e.append(np.abs(n.electric)**2);h.append(np.abs(n.magnetic)**2)
    assert np.mean(e)==pytest.approx(field_power(f.electric)*1e-2,rel=.025) and np.mean(h)==pytest.approx(field_power(f.magnetic)*1e-2,rel=.025)
def test_paired_snr_scaling_only_changes_amplitude():
    f=fields();d=draw_standardized_directions(740104,8,2);a=scale_directions(f,d,40);b=scale_directions(f,d,20);ratio=10**((40-20)/20)
    assert np.allclose(b.electric,a.electric*ratio) and np.allclose(b.magnetic,a.magnetic*ratio)
    paired=scale_paired_robustness(f,d);assert tuple(x.snr_db for x in paired)==(40.,30.,20.,15.,10.)
def test_training_snr_continuous_half_open_and_deterministic():
    a=TrainingNoiseStream(740105);b=TrainingNoiseStream(740105);x=np.array([a.draw_snr_db() for _ in range(1000)]);y=np.array([b.draw_snr_db() for _ in range(1000)])
    assert np.array_equal(x,y) and np.all((x>=30)&(x<40)) and np.any(x!=np.floor(x)) and len(np.unique(x))>990
def test_training_stream_directions_distinct_and_deterministic():
    a=TrainingNoiseStream(740106);b=TrainingNoiseStream(740106);x=a.draw_directions(0,0);y=b.draw_directions(0,0)
    assert np.array_equal(x.electric_real,y.electric_real) and not np.array_equal(x.electric_real,x.magnetic_real)
def test_training_event_freezes_snr_then_four_block_draw_order():
    event_stream=TrainingNoiseStream(740109);manual_stream=TrainingNoiseStream(740109)
    snr,directions=event_stream.draw_event(4,2)
    expected_snr=manual_stream.draw_snr_db();expected=manual_stream.draw_directions(4,2)
    assert snr==expected_snr
    for name in ("electric_real","electric_imag","magnetic_real","magnetic_imag"):
        assert np.array_equal(getattr(directions,name),getattr(expected,name))
def test_no_global_numpy_rng_mutation():
    np.random.seed(71);before=np.random.get_state();draw_standardized_directions(740107,1,1);TrainingNoiseStream(3).draw_snr_db();after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
def test_noise_arrays_immutable():
    d=draw_standardized_directions(740108,0,0);f=fields();n=scale_directions(f,d,30)
    for x in (d.electric_real,d.electric_imag,d.magnetic_real,d.magnetic_imag,n.electric,n.magnetic,complex_fields_to_channels(f)):
        with pytest.raises(ValueError):x.setflags(write=True)
