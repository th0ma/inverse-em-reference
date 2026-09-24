from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from inverse_em.config.evaluation import EvaluationConfig, production_plan
from inverse_em.evaluation.contracts import Bindings, PredictionBundle, row_identity
from inverse_em.evaluation.execution import FixtureEngine, fixture_noise, protected_evaluation, load_historical_checkpoint
from inverse_em.evaluation.reporting import bundle_metrics, aggregate_realizations
from inverse_em.provenance.evaluation import FixtureAuthorization
from inverse_em.scientific.state import ScientificState as State
from inverse_em.observations import ComplexFields


def bindings(task="classifier"):
    return Bindings("fixture-nine", task, EvaluationConfig().sha256, "1"*64, "2"*64, "3"*64,
                    row_identity(("a","b")), "4"*64, "fixture-revision", "CPU/float64;fixture")


def bundle(b=None, attempt="a", condition=None):
    b=b or bindings()
    if b.task=="classifier":
        a=dict(truth=np.array([1,2],dtype=np.int64), predicted=np.array([1,1],dtype=np.int64),logits=np.zeros((2,5),dtype=np.float64)); s=0
    else:
        s=int(b.task[1]); rho=np.tile(np.linspace(.2,.6,s),(2,1))
        a=dict(target_rho=rho,target_phi=np.zeros_like(rho),rho=rho[:,::-1].copy(),phi=np.zeros_like(rho),cos_like=np.ones_like(rho)*2,sin_like=np.zeros_like(rho))
    return PredictionBundle(b,attempt,("a","b"),np.arange(2,dtype=np.int64),tuple(a.items()),s,condition)


def auth(b,before=State.FROZEN,after=State.SEALED_CLEAN,predecessor="0"*64,identity="owner"):
    return FixtureAuthorization(identity,b,before,after,predecessor,"fixture-owner")


def producer(b,attempt="a"):
    def run(condition,completed):
        completed(2)
        return bundle(b,attempt,condition)
    return run


@pytest.fixture(autouse=True)
def sentinels(monkeypatch):
    from inverse_em.populations import generation
    from inverse_em.normalization import api
    from inverse_em.physics import PhysicsTMForward
    def stop(*a,**k): pytest.fail("Protected dependency called")
    monkeypatch.setattr(torch,"load",stop)
    monkeypatch.setattr(PhysicsTMForward,"__init__",stop)
    monkeypatch.setattr(PhysicsTMForward,"__call__",stop,raising=False)
    monkeypatch.setattr(api,"fit_clean_training_normalizer",stop)
    for name in dir(generation):
        if name.startswith("generate") and callable(getattr(generation,name)): monkeypatch.setattr(generation,name,stop)


def test_config():
    assert len(EvaluationConfig().sha256)==64
    assert production_plan()["robustness_seeds"]==(("classifier",20261907),("s1",20261817),("s2",20262007),("s3",20262207))
    assert EvaluationConfig().realizations==tuple(range(10))
    with pytest.raises(TypeError): production_plan()["execution_available"]=True
    with pytest.raises(ValueError): EvaluationConfig(snrs=(10,))
    def check(x):
        assert not callable(x)
        if isinstance(x,tuple):
            for y in x: check(y)
    for x in production_plan().values(): check(x)


@pytest.mark.parametrize("seed",[20261907,20261817,20262007,20262207,20262201,-1,True])
def test_firewall(seed,monkeypatch):
    monkeypatch.setattr(np.random,"SeedSequence",lambda *a:pytest.fail("Protected RNG"))
    with pytest.raises(PermissionError): fixture_noise(None,seed,0,0)


@pytest.mark.parametrize("fn",[protected_evaluation,load_historical_checkpoint])
def test_production(fn):
    with pytest.raises(PermissionError): fn(role="SEALED",seed=20262207)


@pytest.mark.parametrize("row,r",[(0,0),(1,0),(15,9)])
def test_noise(row,r):
    fields=ComplexFields(np.ones(30,dtype=np.complex128),np.ones(30,dtype=np.complex128)*2)
    d, noisy=fixture_noise(fields,123,row,r)
    blocks=[d.electric_real,d.electric_imag,d.magnetic_real,d.magnetic_imag]
    for c,v in enumerate(blocks):
        expected=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence((123,row,r,c)))).standard_normal(30,dtype=np.float64)
        np.testing.assert_array_equal(v,expected)
    assert not np.array_equal(blocks[0],blocks[1])
    np.testing.assert_allclose((noisy[1].electric-fields.electric)/(noisy[0].electric-fields.electric),np.sqrt(10),rtol=1e-12,atol=1e-14)
    again,_=fixture_noise(fields,123,row,r)
    np.testing.assert_array_equal(d.electric_real,again.electric_real)
    other,_=fixture_noise(fields,123,row,(r+1)%10)
    assert not np.array_equal(d.electric_real,other.electric_real)


@pytest.mark.parametrize("task",["classifier","s1","s2","s3"])
def test_bundle_metrics(task):
    x=bundle(bindings(task)); y=PredictionBundle.from_mapping(json.loads(json.dumps(x.to_mapping())))
    assert bundle_metrics(x)==bundle_metrics(y)
    for _,v in x.arrays:
        with pytest.raises(ValueError): v.flat[0]=0
    m=bundle_metrics(y)
    if task=="classifier":
        assert m["accuracy"]==.5 and m["confusion_matrix"][1][0]==1 and m["per_class_f1"][4]==0
    else:
        assert m["source_count"]==2*int(task[1]) and "d_max" in m
        if task!="s1":
            assert m["cartesian"]["rmse"]>0 and m["diagnostic_matching"]["cartesian"]["rmse"]==0


@pytest.mark.parametrize("case",["rows","ids","empty","dtype","nan","shape","argmax","truth","identity","inactive","phi"])
def test_bad_bundle(case):
    x=bundle(bindings("s2" if case in ("inactive","phi") else "classifier")); a=dict(x.arrays); changes={}
    if case=="rows": changes["rows"]=np.array([0,0],dtype=np.int64)
    if case=="ids": changes["ids"]=("a","a")
    if case=="empty": changes["ids"]=("","b")
    if case=="identity": changes["bindings"]=replace(x.bindings,row_order="5"*64)
    if case=="dtype": a["logits"]=a["logits"].astype(np.float32)
    if case=="nan": a["logits"]=np.full((2,5),np.nan)
    if case=="shape": a["logits"]=np.zeros((2,4))
    if case=="argmax": a["predicted"]=np.array([2,2],dtype=np.int64)
    if case=="truth": a["truth"]=np.array([0,2],dtype=np.int64)
    if case=="inactive": a["rho"]=np.zeros((2,3))
    if case=="phi": a["phi"]=np.ones((2,2))
    changes["arrays"]=tuple(a.items())
    with pytest.raises((ValueError,PermissionError)): replace(x,**changes)


def test_wrap_tie():
    x=bundle(bindings("s2")); a=dict(x.arrays)
    a["target_rho"]=a["rho"]=np.full((2,2),.4); a["target_phi"]=np.full((2,2),2*np.pi-.01)
    m=bundle_metrics(replace(x,arrays=tuple(a.items())))
    assert m["angular_degrees"]["rmse"]==pytest.approx(np.rad2deg(.01))
    assert m["diagnostic_matching"]["assignment_change_count"]==0


def records(n=10):
    return [dict(complete=True,bindings="b",seed=123,snr=40,realization=i,metrics=dict(rmse=float(i),matrix=[[i,0],[0,i]])) for i in range(n)]


def test_aggregation():
    r=aggregate_realizations(records()); assert r["summary"]["rmse"]["mean"]==4.5
    assert r["summary"]["rmse"]["sample_sd"]==pytest.approx(np.std(np.arange(10),ddof=1))
    assert r["summary"]["matrix"]["mean"][0][0]==4.5 and len(r["realizations"])==10
    assert r["summary"]["rmse"]["mean"]!=np.sqrt(np.mean(np.arange(10)**2))


@pytest.mark.parametrize("n",[0,1,9,11])
def test_incomplete_aggregate(n):
    with pytest.raises(ValueError): aggregate_realizations(records(n))


@pytest.mark.parametrize("phase",["before_inference","during_inference","after_inference","during_primary_staging","after_primary_persistence","during_reopen_verification","during_metrics","during_report_persistence","before_final_transition_commit"])
@pytest.mark.parametrize("robust",[False,True])
def test_faults(tmp_path,phase,robust):
    b=bindings(); before,after=(State.SEALED_CLEAN,State.ROBUSTNESS) if robust else (State.FROZEN,State.SEALED_CLEAN)
    e=FixtureEngine(tmp_path,b,before); permission=auth(b,before,after)
    def fault(point):
        if point==phase: raise RuntimeError("injected")
    def produce(condition,completed):
        if phase=="during_inference":
            completed(1)
            raise RuntimeError("partial")
        completed(2)
        return bundle(b,condition=condition)
    with pytest.raises(RuntimeError): e.run(permission,"a",produce,fault=fault)
    assert e.state is before and e.store.official("a") is None
    failure=json.loads((tmp_path/"a"/"INCOMPLETE.json").read_text())
    assert failure["phase"]==phase and failure["accounting"]["officially_committed_evaluations"]==0
    if phase=="during_inference": assert failure["accounting"]["rows_processed"]==1 and not failure["accounting"]["exact_execution_count_known"]
    with pytest.raises(FileExistsError): e.run(permission,"b",producer(b,"b"))


def test_success(tmp_path,monkeypatch):
    import inverse_em.evaluation.execution as execution
    b=bindings(); live=bundle(b); original=execution.bundle_metrics
    def metrics(x):
        assert x is not live and (tmp_path/"a"/"clean.PRIMARY_COMMIT.json").exists()
        return original(x)
    monkeypatch.setattr(execution,"bundle_metrics",metrics)
    e=FixtureEngine(tmp_path,b)
    def produce(c,completed): completed(2); return live
    result=e.run(auth(b),"a",produce)
    assert e.state is State.SEALED_CLEAN and result["accounting"]["officially_committed_evaluations"]==1
    stale=FixtureEngine(tmp_path,b)
    with pytest.raises(PermissionError): stale.run(auth(b,identity="other"),"b",producer(b,"b"))


def test_robust_success(tmp_path):
    b=bindings("s1"); e=FixtureEngine(tmp_path,b,State.SEALED_CLEAN)
    r=e.run(auth(b,State.SEALED_CLEAN,State.ROBUSTNESS),"a",producer(b))
    assert r["accounting"]["verified_primary_bundles"]==50 and e.state is State.ROBUSTNESS


def test_closed(tmp_path):
    b=bindings(); e=FixtureEngine(tmp_path,b,State.CLOSED)
    with pytest.raises(PermissionError): e.run(auth(b),"a",lambda *a:pytest.fail("inference"))
    assert list(tmp_path.iterdir())==[]


def test_tamper(tmp_path):
    b=bindings(); e=FixtureEngine(tmp_path,b)
    def fault(point):
        if point=="after_primary_persistence": (tmp_path/"a"/"clean.primary.json").write_text("{}")
    with pytest.raises(ValueError): e.run(auth(b),"a",producer(b),fault=fault)
    assert e.state is State.FROZEN


@pytest.mark.parametrize("case",["duplicate","snr","seed","incomplete","nan"])
def test_bad_aggregation_identity(case):
    rows=records()
    if case=="duplicate": rows[0]["realization"]=1
    if case=="snr": rows[0]["snr"]=30
    if case=="seed": rows[0]["seed"]=999
    if case=="incomplete": rows[0]["complete"]=False
    if case=="nan": rows[0]["metrics"]["rmse"]=float("nan")
    with pytest.raises(ValueError): aggregate_realizations(rows)


@pytest.mark.parametrize("step",["stage_primary","publish_primary","reopen","publish_report","publish_success"])
def test_actual_io_failure(tmp_path,monkeypatch,step):
    b=bindings(); e=FixtureEngine(tmp_path,b)
    def fail(*a,**k): raise OSError("I/O injected")
    if step in ("stage_primary","publish_primary","reopen"):
        monkeypatch.setattr(e.store,step,fail)
    else:
        original=e.store.publish
        def publish(directory,name,value):
            if name==("REPORT" if step=="publish_report" else "SUCCESS"): raise OSError("I/O injected")
            return original(directory,name,value)
        monkeypatch.setattr(e.store,"publish",publish)
    with pytest.raises(OSError): e.run(auth(b),"a",producer(b))
    assert e.state is State.FROZEN and e.store.official("a") is None
    assert (tmp_path/"a"/"STARTED.json").exists() and (tmp_path/"a"/"INCOMPLETE.json").exists()


def test_concurrent_duplicate_attempt(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    b=bindings()
    def run(attempt):
        try:
            FixtureEngine(tmp_path,b).run(auth(b),attempt,producer(b,attempt))
            return "success"
        except (FileExistsError,PermissionError): return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,["a","b"]))
    assert sorted(results)==["rejected","success"]


def test_missing_terminal_marker(tmp_path):
    b=bindings(); e=FixtureEngine(tmp_path,b)
    e.store.start("a",auth(b),{"test":"durable-start"})
    assert e.store.official("a") is None
    try:
        with pytest.raises((FileExistsError,PermissionError)): FixtureEngine(tmp_path,b).run(auth(b,identity="new"),"b",producer(b,"b"))
    finally:
        e.store.relinquish()


def test_condition_immutability_and_bad_rows():
    from inverse_em.evaluation.contracts import Condition
    from inverse_em.config.phase2 import Phase2ScientificConfig
    b=bindings()
    c=Condition(40,0,123,b.row_order,b.clean_fields,b.environment,Phase2ScientificConfig().sha256,np.__version__)
    x=bundle(b,condition=c)
    assert replace(x,attempt="b").condition==x.condition
    with pytest.raises(ValueError): replace(c,realization=10)
    with pytest.raises(ValueError): replace(x,condition=replace(c,row_order="9"*64))
    data=bundle().to_mapping(); data["rows"]=[0.1,1.0]
    with pytest.raises(ValueError): PredictionBundle.from_mapping(data)


def test_explicit_new_authorization(tmp_path):
    b=bindings(); e=FixtureEngine(tmp_path,b)
    def failed(c,done): done(1); raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError): e.run(auth(b),"a",failed)
    with pytest.raises(PermissionError): e.store.approve_new_attempt("a","b",auth(b))
    fresh=auth(b,identity="explicit-second")
    e.store.approve_new_attempt("a","b",fresh)
    result=e.run(fresh,"b",producer(b,"b"))
    assert result["accounting"]["officially_committed_evaluations"]==1
    assert (tmp_path/"a"/"INCOMPLETE.json").exists()
    assert e.store.official("a") is None


def test_frozen_dependency_change_rejected(tmp_path):
    b=bindings(); e=FixtureEngine(tmp_path,b)
    e.run(auth(b),"a",producer(b))
    altered=replace(b,checkpoint="9"*64)
    with pytest.raises(PermissionError):
        FixtureEngine(tmp_path,altered).run(auth(altered,identity="other"),"b",producer(altered,"b"))


@pytest.mark.parametrize("field",["authorization_sha256","predecessor","before","after","accounting","study","attempt","bindings","report_sha256","completed"])
def test_correction_success_mutations(tmp_path,field):
    b=bindings(); e=FixtureEngine(tmp_path,b)
    e.run(auth(b),"a",producer(b))
    path=tmp_path/"a"/"SUCCESS.json"
    value=json.loads(path.read_bytes())
    value[field]={"officially_committed_evaluations":999} if field=="accounting" else "f"*64
    path.write_text(json.dumps(value))
    before=path.read_bytes()
    with pytest.raises(ValueError): e.store.official("a")
    assert path.read_bytes()==before and e.state is State.SEALED_CLEAN


@pytest.mark.parametrize("artifact",["REPORT.json","clean.primary.json","clean.PRIMARY_COMMIT.json","STARTED.json","OWNERSHIP.json","TRANSITION_WITNESS.json","clean.PRODUCER.json"])
def test_correction_predecessor_corruption(tmp_path,artifact):
    b=bindings(); e=FixtureEngine(tmp_path,b)
    e.run(auth(b),"a",producer(b))
    path=tmp_path/"a"/artifact
    path.write_text("{}")
    snapshot={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    calls=[]
    def never(c,done): calls.append(1); raise AssertionError("producer must not run")
    with pytest.raises((ValueError,KeyError,PermissionError)):
        e.run(auth(b,State.SEALED_CLEAN,State.ROBUSTNESS,e.predecessor,"next"),"b",never)
    assert calls==[] and e.state is State.SEALED_CLEAN
    assert snapshot=={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_correction_live_retry_isolation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    b=bindings(); e=FixtureEngine(tmp_path,b); entered=Event(); finish=Event()
    def blocked(c,done):
        entered.set()
        if not finish.wait(10): raise AssertionError("test synchronization timeout")
        done(2)
        return bundle(b)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running=pool.submit(e.run,auth(b),"a",blocked)
        assert entered.wait(10)
        other=FixtureEngine(tmp_path,b)
        snapshot={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file() and p.suffix!='.lock'}
        try:
            with pytest.raises(PermissionError): other.store.approve_new_attempt("a","b",auth(b,identity="new"))
            with pytest.raises(PermissionError): e.store.approve_new_attempt("a","b",auth(b,identity="new"))
            assert snapshot=={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file() and p.suffix!='.lock'}
            with pytest.raises(PermissionError): other.run(auth(b,identity="new"),"b",producer(b,"b"))
        finally:
            finish.set()
        running.result(timeout=10)
    assert len(list(tmp_path.glob('*/SUCCESS.json')))==1


def test_correction_stale_writer(tmp_path):
    b=bindings(); e=FixtureEngine(tmp_path,b)
    def relinquish(point):
        if point=="before_final_transition_commit": e.store.relinquish()
    with pytest.raises(PermissionError): e.run(auth(b),"a",producer(b),fault=relinquish)
    assert e.state is State.FROZEN and e.store.official("a") is None
    assert not (tmp_path/"a"/"SUCCESS.json").exists()


@pytest.mark.parametrize("mode",["untouched","reordered","normalized"])
@pytest.mark.parametrize("task",["s1","s2","s3"])
def test_correction_producer_binding(tmp_path,mode,task):
    if mode=="reordered" and task=="s1":
        # Row order substitution for the one-slot task.
        pass
    b=bindings(task); original=bundle(b); a=dict(original.arrays)
    # Asymmetric rows ensure row reordering in S1 changes content.
    a["rho"]=a["rho"].copy(); a["rho"][1]+=0.01
    original=replace(original,arrays=tuple(a.items())); e=FixtureEngine(tmp_path,b)
    def produce(c,done): done(2); return original
    def substitute(point):
        if point!="after_inference" or mode=="untouched": return
        values=dict(original.arrays)
        if mode=="reordered":
            for key in ("rho","phi","cos_like","sin_like"):
                values[key]=values[key][::-1,:].copy() if task=="s1" else values[key][:,::-1].copy()
        else:
            values["cos_like"]=values["cos_like"]/2
            values["sin_like"]=values["sin_like"]/2
        replacement=replace(original,arrays=tuple(values.items()))
        object.__setattr__(original,"arrays",replacement.arrays)
    if mode=="untouched":
        assert e.run(auth(b),"a",produce,fault=substitute)["after"]=="SEALED_CLEAN"
    else:
        with pytest.raises(ValueError): e.run(auth(b),"a",produce,fault=substitute)
        assert e.state is State.FROZEN and e.store.official("a") is None


@pytest.mark.parametrize("changes",[{"aggregation_ddof":True},{"aggregation_ddof":1.0},{"snrs":(40.,30.,20.,15.,10.)},{"realizations":(False,1,2,3,4,5,6,7,8,9)}])
def test_correction_config_types(changes):
    with pytest.raises(ValueError): EvaluationConfig(**changes)
    assert EvaluationConfig().sha256=="3c4f1e60d40aec813ead60c6ea5d9310471b2ae73f4d4bb4774cef570043f2d2"


@pytest.mark.parametrize("case", ["missing", "empty", "other_attempt", "other_study",
                                  "stale", "unrelated", "incomplete", "corrupt"])
def test_positive_predecessor_resolution(tmp_path, monkeypatch, case):
    b = bindings()
    e = FixtureEngine(tmp_path, b)
    e.run(auth(b), "a", producer(b))
    declared = e.predecessor
    if case in ("missing", "empty", "other_attempt", "other_study", "unrelated"):
        # Only disposable fixture evidence is removed, never repository artifacts.
        (tmp_path / "a" / "SUCCESS.json").unlink()
    if case == "empty":
        assert list(tmp_path.glob("*/SUCCESS.json")) == []
    if case == "other_attempt":
        other = FixtureEngine(tmp_path, b)
        other.run(auth(b, identity="other"), "other", producer(b, "other"))
    if case in ("other_study", "unrelated"):
        for index in range(2 if case == "unrelated" else 1):
            other_b = replace(b, study=f"unrelated-{index}")
            other = FixtureEngine(tmp_path, other_b)
            name = f"other-{index}"
            other.run(auth(other_b, identity=name), name, producer(other_b, name))
    if case == "stale":
        declared = "f" * 64
        e.predecessor = declared
    if case == "incomplete":
        (tmp_path / "a" / "clean.PRIMARY_COMMIT.json").unlink()
    if case == "corrupt":
        (tmp_path / "a" / "SUCCESS.json").write_text("{}")
    permission = auth(b, State.SEALED_CLEAN, State.ROBUSTNESS, declared, "next")
    def snapshot():
        return {str(p.relative_to(tmp_path)): p.read_bytes()
                for p in tmp_path.rglob("*") if p.is_file()}
    before = snapshot()
    paths = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    numpy_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
    calls = []
    def never(*args, **kwargs):
        calls.append(1)
        raise AssertionError("producer/protected boundary reached")
    monkeypatch.setattr(np.random, "SeedSequence", never)
    monkeypatch.setattr(torch.nn.Module, "_call_impl", never)
    monkeypatch.setattr(torch, "load", never)
    with pytest.raises((PermissionError, ValueError, KeyError, FileNotFoundError)):
        e.run(permission, "next", never)
    assert calls == []
    assert e.state is State.SEALED_CLEAN and e.predecessor == declared
    assert e.store._lease is None and e.store._owner is None
    assert not (tmp_path / "next").exists()
    assert snapshot() == before
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == paths
    assert e.store.official("next") is None
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0]
    np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    assert torch.equal(torch_state, torch.get_rng_state())


def test_positive_predecessor_valid_chain(tmp_path):
    b = bindings()
    e = FixtureEngine(tmp_path, b)
    e.run(auth(b), "a", producer(b))
    permission = auth(b, State.SEALED_CLEAN, State.ROBUSTNESS, e.predecessor, "next")
    e.run(permission, "next", producer(b, "next"))
    assert e.state is State.ROBUSTNESS
    assert e.store.official("next") is not None


def test_correction_stale_writer_after_transfer(tmp_path):
    b=bindings(); first=FixtureEngine(tmp_path,b); saved={}
    def transfer(point):
        if point!="before_final_transition_commit": return
        first.store.relinquish()
        second=FixtureEngine(tmp_path,b)
        permission=auth(b,identity="authorized-replacement")
        second.store.approve_new_attempt("a","b",permission)
        second.run(permission,"b",producer(b,"b"))
        saved["success"]=(tmp_path/"b"/"SUCCESS.json").read_bytes()
        saved["incomplete"]=(tmp_path/"a"/"INCOMPLETE.json").read_bytes()
    with pytest.raises(PermissionError): first.run(auth(b),"a",producer(b),fault=transfer)
    assert first.state is State.FROZEN and first.store.official("a") is None
    assert (tmp_path/"b"/"SUCCESS.json").read_bytes()==saved["success"]
    assert (tmp_path/"a"/"INCOMPLETE.json").read_bytes()==saved["incomplete"]
    assert len(list(tmp_path.glob('*/SUCCESS.json')))==1


@pytest.mark.parametrize("field",["authorization_sha256","predecessor","before","after","accounting"])
def test_correction_receipt_reconciliation_not_hash_only(tmp_path,field):
    from inverse_em.provenance.canonical import canonical_sha256
    b=bindings(); e=FixtureEngine(tmp_path,b); e.run(auth(b),"a",producer(b))
    path=tmp_path/"a"/"SUCCESS.json"; value=json.loads(path.read_bytes())
    value[field]={"officially_committed_evaluations":999} if field=="accounting" else "f"*64
    path.write_text(json.dumps(value))
    # Even a self-consistent replacement digest is not authorization/evidence.
    (tmp_path/"a"/"TRANSITION_WITNESS.json").write_text(json.dumps({"success_sha256":canonical_sha256(value)}))
    with pytest.raises(ValueError): e.store.official("a")
