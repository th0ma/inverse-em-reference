import importlib
import hashlib
import json
import math
import sys
import subprocess
from pathlib import Path

import pytest
from phase11_accounting import (Accounting, GRANDFATHERED, BOUNDED, REJECTED,
                                UNAUTHORIZED, UNKNOWN, classify_path, ROOT)
from phase11_accounting import DirectFirewall, admitted_lengths
from phase11_accounting import ACCOUNTING
from phase11_accounting import CANDIDATE_PATHS, verify_baseline, read_candidate_manifest

# Captured during collection, before the Phase-11 autouse guards.
import numpy as np
RANDOM_STATE_ALIAS = np.random.RandomState
DEFAULT_RNG_ALIAS = np.random.default_rng
NATIVE_ALIAS = math.gcd


def test_plugin_is_mandatory(request):
    assert request.config.pluginmanager.hasplugin("phase11_accounting")


def closure_fixture(root, state="staged"):
    root.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)
    git("init", "-q")
    git("config", "user.name", "Inert fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "core.autocrlf", "false")
    (root / "closed.txt").write_bytes(b"frozen\n")
    git("add", "closed.txt")
    git("-c", "core.hooksPath=", "commit", "-qm", "inert baseline")
    ledger = {"baseline": git("rev-parse", "HEAD").decode().strip(),
              "tree": git("rev-parse", "HEAD^{tree}").decode().strip(),
              "closed_files": {"closed.txt": hashlib.sha256(b"frozen\n").hexdigest()}}
    candidate = {}
    for path in CANDIDATE_PATHS:
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"inert candidate\n")
        candidate[path] = hashlib.sha256(file.read_bytes()).hexdigest()
        if state == "staged" or state == "mixed" and path == "docs/phase11.md":
            git("add", "--", path)
    return git, ledger, candidate


@pytest.mark.parametrize("state", ["unstaged", "staged", "mixed"])
def test_closure_exact_candidate(tmp_path, state):
    root = tmp_path / "repository"
    git, ledger, candidate = closure_fixture(root, state)
    verify_baseline(root, ledger, candidate)


@pytest.mark.parametrize("mutation", ["closed_modify", "closed_delete", "closed_rename",
    "closed_index_only", "extra_staged", "extra_untracked", "extra_phase11",
    "candidate_altered", "candidate_index_only", "candidate_missing"])
def test_closure_reject_mutation(tmp_path, mutation):
    root = tmp_path / "repository"
    git, ledger, candidate = closure_fixture(root)
    if mutation == "closed_modify":
        (root / "closed.txt").write_bytes(b"changed\n")
    elif mutation == "closed_delete":
        (root / "closed.txt").unlink()
    elif mutation == "closed_rename":
        git("mv", "closed.txt", "renamed.txt")
    elif mutation == "closed_index_only":
        (root / "closed.txt").write_bytes(b"changed\n")
        git("add", "closed.txt")
        (root / "closed.txt").write_bytes(b"frozen\n")
    elif mutation.startswith("extra"):
        name = "tests/phase11/extra.py" if mutation == "extra_phase11" else "extra.txt"
        (root / name).write_bytes(b"unexpected\n")
        if mutation == "extra_staged": git("add", name)
    elif mutation == "candidate_missing":
        (root / "docs/phase11.md").unlink()
    else:
        file = root / "docs/phase11.md"
        file.write_bytes(b"altered\n")
        if mutation == "candidate_index_only":
            git("add", "docs/phase11.md")
            file.write_bytes(b"inert candidate\n")
    with pytest.raises((RuntimeError, FileNotFoundError, subprocess.CalledProcessError)):
        verify_baseline(root, ledger, candidate)


def test_closure_manifest_identity(tmp_path):
    path = tmp_path / "candidate.json"
    data = json.dumps({p: "a" * 64 for p in CANDIDATE_PATHS}).encode()
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    assert set(read_candidate_manifest(path, digest)) == CANDIDATE_PATHS
    for value, identity in ((None, None), (path, "0" * 64)):
        with pytest.raises(RuntimeError): read_candidate_manifest(value, identity)
    path.write_bytes(b"{}")
    with pytest.raises(RuntimeError): read_candidate_manifest(path, hashlib.sha256(b"{}").hexdigest())


def test_closure_weakening_challenges(tmp_path, monkeypatch):
    module = sys.modules[__name__]
    original = verify_baseline
    def old_guard(root, ledger, candidate):
        if subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--name-only"]).strip():
            raise RuntimeError("Closed tracked files changed")
        return original(root, ledger, candidate)
    # Actual new regressions run unchanged, with in-memory guard replacement.
    monkeypatch.setattr(module, "verify_baseline", old_guard)
    with pytest.raises(RuntimeError, match="Closed tracked"):
        test_closure_exact_candidate(tmp_path, "staged")
    other = tmp_path / "other"
    other.mkdir()
    def accepts_arbitrary_additions(root, ledger, candidate):
        for path, digest in ledger["closed_files"].items():
            assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    monkeypatch.setattr(module, "verify_baseline", accepts_arbitrary_additions)
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        test_closure_reject_mutation(other, "extra_staged")


@pytest.mark.parametrize("category", [GRANDFATHERED, BOUNDED])
def test_partial_is_not_zero(category):
    audit = Accounting()
    token = audit.begin(category, "detached-adversarial-fixture")
    assert audit.snapshot()["categories"][UNKNOWN] == 1
    audit.finish(token, False)
    assert audit.snapshot()["categories"][UNKNOWN] == 1
    assert audit.snapshot()["categories"][category] == 0


def test_rejection_and_success_have_different_units():
    audit = Accounting()
    audit.rejected()
    token = audit.begin(BOUNDED, "pure-fixture")
    audit.finish(token, True)
    counts = audit.snapshot()["categories"]
    assert counts[REJECTED] == counts[BOUNDED] == 1
    assert counts[UNKNOWN] == counts[UNAUTHORIZED] == 0
    with pytest.raises(KeyError):
        audit.finish(token, True)


@pytest.mark.parametrize("category", [REJECTED, UNAUTHORIZED, UNKNOWN, "trusted"])
def test_cannot_self_attest_execution(category):
    with pytest.raises(ValueError):
        Accounting().begin(category, "execution")


def test_callsite_not_function_or_seed_confers_grandfathering():
    ledger = json.loads((Path(__file__).with_name("contract_ledger.json")).read_text())
    assert classify_path("tests/phase3/test_data.py", ledger["closed_files"]) == GRANDFATHERED
    assert classify_path("tests/phase11/test_phase11_contract_firewall.py", ledger["closed_files"]) == BOUNDED
    with pytest.raises(PermissionError):
        classify_path("tests/phase3/new_generator_test.py", ledger["closed_files"])


@pytest.mark.parametrize("module,name,args", [
    ("inverse_em.surrogates.data", "generate_historical_sources", ()),
    ("inverse_em.populations", "generate_s3", (1, 20262202, 8)),
    ("inverse_em.populations.generation", "generate_s1", (1, 740001)),
    ("inverse_em.normalization.api", "fit_clean_training_normalizer", ("s1",)),
    ("inverse_em.training.s1", "initialize_localizer", ()),
    ("inverse_em.training.s2", "initialize_localizer", ()),
    ("inverse_em.training.s3", "initialize_localizer", ()),
    ("inverse_em.localization.s1", "noise_event", ("fixture", 1)),
    ("numpy.random", "default_rng", (20260915,)),
    ("numpy.random", "PCG64DXSM", (20262202,)),
    ("torch", "load", ("never-open-this-checkpoint",)),
    ("torch", "manual_seed", (20261813,)),
])
def test_new_callsite_cannot_borrow_old_authority(module, name, args):
    with pytest.raises(PermissionError, match="Phase-11"):
        getattr(importlib.import_module(module), name)(*args)


def test_physics_rejected_before_body():
    from inverse_em.physics.api import PhysicsTMForward
    # No solver construction or source creation: guard must reject even this.
    with pytest.raises(PermissionError, match="Phase-11"):
        PhysicsTMForward.evaluate(None, None, None)


def test_ledger_has_no_production_execution_handle():
    ledger = json.loads((ROOT / "tests/phase11/contract_ledger.json").read_text())
    assert ledger["equality"]["class_c"] is None
    assert ledger["equality"]["class_b"] == {"rtol": 1e-12, "atol": 1e-14}
    assert all(isinstance(v, str) and len(v) == 64 for v in ledger["configs"].values())


@pytest.mark.parametrize("value", ["FAILED", "PASSED", 1, 0, [], {}, object(), None])
def test_exact_boolean_completion(value):
    a = Accounting()
    token = a.begin(BOUNDED, "inert")
    with pytest.raises(TypeError):
        a.finish(token, value)
    assert token in a.pending
    assert a.snapshot()["categories"][BOUNDED] == 0
    assert a.snapshot()["categories"][UNKNOWN] == 2
    a.finish(token, True)
    assert a.snapshot()["categories"][BOUNDED] == 1
    assert a.snapshot()["categories"][UNKNOWN] == 1


@pytest.mark.parametrize("moment", ["pending", "finished"])
@pytest.mark.parametrize("mutation", ["zero", "delete", "wrong", "inconsistent"])
def test_public_views_cannot_rewrite_evidence(moment, mutation):
    a = Accounting()
    token = a.begin(BOUNDED, "inert")
    a.rejected()
    if moment == "finished":
        a.finish(token, True)
    public = a.items
    if mutation == "zero":
        public.update({key: -public[key] for key in public})
    elif mutation == "delete":
        del public[REJECTED]
    else:
        public[REJECTED] = "bad" if mutation == "wrong" else 99
    if moment == "pending":
        a.finish(token, True)
    snap = a.snapshot()
    assert snap["categories"][BOUNDED] == snap["categories"][REJECTED] == 1
    snap["categories"][REJECTED] = 0
    assert a.snapshot()["categories"][REJECTED] == 1


def test_invalid_token_evidence_and_no_aliasing():
    a = Accounting()
    t = a.begin(BOUNDED, "inert")
    a.pending.clear()
    a.finish(t, True)
    for bad in (t, 987654, True):
        with pytest.raises(KeyError):
            a.finish(bad, True)
    assert a.snapshot()["categories"][BOUNDED] == 1
    assert a.snapshot()["categories"][UNKNOWN] == 3


def test_attempt_dedup_and_conflicts():
    a = Accounting()
    token = a.attempt()
    for layer in ("admission", "monitor", "profile"):
        a.rejected(token, layer)
    assert a.snapshot()["categories"][REJECTED] == 1
    a.rejected()
    assert a.snapshot()["categories"][REJECTED] == 2
    a.observe_attempt(token, UNAUTHORIZED, "late-evidence")
    assert a.snapshot()["categories"][UNAUTHORIZED] == 1
    assert a.snapshot()["categories"][UNKNOWN] == 1


@pytest.mark.parametrize("kind", ["python", "native"])
def test_captured_alias_persistent_rejection(kind):
    seen = []
    def inert(*args):
        seen.append(1)
    alias = inert if kind == "python" else NATIVE_ALIAS
    imported = {"rebound": alias}
    a = Accounting()
    before = sys.getprofile()
    with DirectFirewall(a, (alias,)):
        for _ in range(3):
            with pytest.raises(PermissionError):
                imported["rebound"](4, 2)
        assert sys.getprofile() is before
    assert seen == []
    assert a.snapshot()["categories"][REJECTED] == 3
    assert a.snapshot()["categories"][UNAUTHORIZED] == a.snapshot()["categories"][UNKNOWN] == 0


@pytest.mark.parametrize("alias", [RANDOM_STATE_ALIAS, DEFAULT_RNG_ALIAS])
def test_rng_captured_before_install_is_rejected(alias):
    import torch
    before = np.random.get_state()
    state = torch.get_rng_state().clone()
    for _ in range(3):
        with pytest.raises(PermissionError):
            alias(20262202)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("module,name", [("torch.inert", "uniform_"), ("torch.inert", "set_state"), ("numpy.random.inert", "normal")])
def test_inert_native_rng_method_classification(module, name):
    seen = []
    def inert(self, *args):
        seen.append(1)
    inert.__name__ = name
    cls = type("InertRNG", (), {"__module__": module, name: inert})
    alias = getattr(cls(), name)
    for _ in range(3):
        with pytest.raises(PermissionError):
            alias(20262202)
    assert seen == []


def test_callback_boundary_and_positive_controls():
    before = ACCOUNTING.snapshot()["categories"][REJECTED]
    assert admitted_lengths(["a", b"bc"]) == (1, 2)
    assert admitted_lengths([]) == ()
    # Construction outside the owned surface is solely a harmless limitation
    # fixture. No scientific callback is ever placed inside this object.
    opaque = eval("map(len, [(), (1,)])")
    for _ in range(3):
        with pytest.raises(PermissionError):
            admitted_lengths(opaque)
    assert next(opaque) == 0  # Admission did not consume even one item.
    assert ACCOUNTING.snapshot()["categories"][REJECTED] == before + 3
    with pytest.raises(PermissionError):
        map(len, [()])
    with pytest.raises(PermissionError):
        filter(bool, [1])


@pytest.mark.parametrize("kind", ["foreign", "duck", "subclass", "callable", "proxy", "lookup"])
@pytest.mark.parametrize("form", ["positional", "keyword"])
def test_m1_foreign_dependency_zero_entry(kind, form):
    callbacks, bodies = [], []
    def protected_inert():
        bodies.append(1)
    def hook(*args, **kwargs):
        callbacks.append(1)
        protected_inert()
    class Foreign:
        rejected = hook
    class Duck:
        rejected = hook
    class Subclass(Accounting):
        rejected = hook
    class Callable:
        __call__ = hook
        rejected = hook
    class Proxy:
        def __getattr__(self, name):
            callbacks.append(1)
            return hook
    class Lookup:
        def __getattribute__(self, name):
            callbacks.append(1)
            return hook
    dependency = {"foreign": Foreign, "duck": Duck, "subclass": Subclass,
                  "callable": Callable, "proxy": Proxy, "lookup": Lookup}[kind]()
    before = ACCOUNTING.snapshot()["categories"][REJECTED]
    event_start = len(ACCOUNTING._events)
    with pytest.raises(PermissionError):
        if form == "positional":
            admitted_lengths(object(), dependency)
        else:
            admitted_lengths(object(), audit=dependency)
    assert callbacks == [] and bodies == []
    assert ACCOUNTING.snapshot()["categories"][REJECTED] == before + 1
    events = ACCOUNTING._events[event_start:]
    assert len(events) == 1
    assert events[0][0] == "attempt" and events[0][2:] == (REJECTED, "admission")


def test_m1_data_only_positive_and_owned_rejection():
    assert admitted_lengths(["a", b"bc"]) == (1, 2)
    assert admitted_lengths(("", b"abc")) == (0, 3)
    assert admitted_lengths(()) == ()
    before = ACCOUNTING.snapshot()["categories"][REJECTED]
    for args, kwargs in ((([], None), {}), (([],), {"audit": None}), ((object(),), {})):
        with pytest.raises(PermissionError):
            admitted_lengths(*args, **kwargs)
    assert ACCOUNTING.snapshot()["categories"][REJECTED] == before + 3


def test_m1_in_memory_weakening_detected():
    import subprocess
    script = r'''
import test_phase11_contract_firewall as regression
from phase11_accounting import ACCOUNTING
regression.test_m1_foreign_dependency_zero_entry("foreign", "positional")
def vulnerable(values, audit=None):
    audit = ACCOUNTING if audit is None else audit
    if type(values) not in (tuple, list) or any(type(v) not in (str, bytes) for v in values):
        audit.rejected(layer="admission")
        raise PermissionError()
    return tuple(map(len, values))
regression.admitted_lengths = vulnerable
try:
    regression.test_m1_foreign_dependency_zero_entry("foreign", "positional")
except AssertionError:
    print("M1 unchanged regression detected restored vulnerability")
else:
    raise AssertionError("M1 weakening survived")
'''
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "detected restored vulnerability" in result.stdout


def test_interpreter_limitation_in_isolated_inert_process():
    import subprocess
    script = '''
import sys
it = map(len, [(), (1,)])
seen = []
def call(c, o, fn, arg):
    if fn is len: seen.append(1)
m = sys.monitoring
m.use_tool_id(4, "inert-limit")
m.register_callback(4, m.events.CALL, call)
m.set_events(4, m.events.CALL)
out = [v for v in it]
m.set_events(4, 0)
m.free_tool_id(4)
assert out == [0, 1] and seen == []
'''
    r = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("form", ["sorted", "min", "max", "sort", "iter", "partial", "reduce", "methodcaller"])
def test_unadmitted_callback_forms(form):
    import functools
    import operator
    seen = []
    def inert(value=None):
        seen.append(1)
        return 0
    with pytest.raises(PermissionError):
        if form == "sorted":
            sorted([1], key=inert)
        elif form == "min":
            min([1], key=inert)
        elif form == "max":
            max([1], key=inert)
        elif form == "sort":
            [1].sort(key=inert)
        elif form == "iter":
            iter(inert, 1)
        elif form == "partial":
            functools.partial(inert)
        elif form == "reduce":
            functools.reduce(inert, [1, 2])
        else:
            operator.methodcaller("inert")
    assert seen == []


def test_in_memory_weakening_challenges():
    """Identical assertions must fail under six independently weakened models."""
    import subprocess
    script = r'''
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from phase11_accounting import *
from conftest import _rng_targets
def must_fail(check):
    try: check()
    except (AssertionError, TypeError, KeyError): return
    raise AssertionError("weakening survived")
def repeated(factory):
    body=[]
    def inert(*a): body.append(1)
    alias=inert
    with factory(alias):
        for _ in range(3):
            try: alias(20262202)
            except PermissionError: pass
    assert body == []
@contextmanager
def old_profile(fn):
    previous=sys.getprofile()
    def observe(f,e,a):
        if e == 'call' and f.f_code is fn.__code__: raise PermissionError()
    sys.setprofile(observe)
    try: yield
    finally: sys.setprofile(previous)
repeated(lambda fn: DirectFirewall(Accounting(), (fn,)))
must_fail(lambda: repeated(old_profile))
def opaque_check(gate):
    it=map(len,[()])
    rejected=False
    try: gate(it)
    except PermissionError: rejected=True
    assert rejected and next(it) == 0
opaque_check(lambda it: admitted_lengths(it))
must_fail(lambda: opaque_check(tuple))
repeated(lambda fn: DirectFirewall(Accounting(), _rng_targets(SimpleNamespace(RandomState=fn))))
must_fail(lambda: repeated(lambda fn: DirectFirewall(Accounting(), ())))
def status_check(cls):
    a=cls(); t=a.begin(BOUNDED,'inert')
    try: a.finish(t,'FAILED')
    except TypeError: pass
    assert t in a.pending and a.snapshot()['categories'][BOUNDED] == 0
class Truthy(Accounting):
    def finish(self,t,s): return super().finish(t,bool(s))
status_check(Accounting)
must_fail(lambda: status_check(Truthy))
def reset_check(cls):
    a=cls(); t=a.begin(BOUNDED,'inert'); a.finish(t,True); a.rejected()
    a.items.clear()
    assert a.snapshot()['categories'][REJECTED] == 1
class Mutable(Accounting):
    def __init__(self):
        super().__init__(); self.public=None
    @property
    def items(self):
        if self.public is None: self.public=super().snapshot()['categories']
        return self.public
    def snapshot(self):
        out=super().snapshot()
        if self.public is not None: out['categories']=self.public
        return out
reset_check(Accounting)
must_fail(lambda: reset_check(Mutable))
def dedup_check(cls):
    a=cls(); t=a.attempt()
    for layer in ('admission','monitor','profile'): a.rejected(t,layer)
    assert a.snapshot()['categories'][REJECTED] == 1
class Inflated(Accounting):
    def rejected(self,t=None,layer='sentinel'): return super().rejected(None,layer)
dedup_check(Accounting)
must_fail(lambda: dedup_check(Inflated))
print('six corrected controls passed; six in-memory weakenings detected')
'''
    r = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "six in-memory weakenings detected" in r.stdout
