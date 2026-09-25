"""Test-only observational ledger. Not a production authorization interface.

Units are pytest items and observed Python mechanism entries, NOT scientific rows,
RNG draws, optimizer updates or completed traversals. Child-process work is covered
by the unchanged test contract, not attributed a fabricated per-call count.
"""
from collections import Counter
import builtins
import functools
import operator
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = Path(__file__).with_name("contract_ledger.json")
GRANDFATHERED = "GRANDFATHERED_CLOSED_PHASE_REGRESSION"
BOUNDED = "PHASE11_BOUNDED_FIXTURE"
REJECTED = "PROTECTED_ATTEMPT_REJECTED_BEFORE_EXECUTION"
UNAUTHORIZED = "UNAUTHORIZED_PROTECTED_EXECUTION"
UNKNOWN = "UNKNOWN_OR_PARTIAL_EXECUTION"
CATEGORIES = (GRANDFATHERED, BOUNDED, REJECTED, UNAUTHORIZED, UNKNOWN)


class Accounting:
    def __init__(self):
        self._events = []
        self._pending = {}
        self._serial = 0
        self._attempts = {}
        self._frames = {}

    @property
    def items(self):
        # Public aggregates are disposable views, never the evidence authority.
        return Counter(self.snapshot()["categories"])

    @property
    def mechanisms(self):
        return Counter(self.snapshot()["mechanism_entries"])

    @property
    def pending(self):
        return dict(self._pending)

    def record_entry(self, category, label):
        self._events.append(("entry", category, label))

    def invalid(self, reason):
        self._events.append(("invalid", reason))

    def begin(self, category, mechanism):
        if category not in (GRANDFATHERED, BOUNDED):
            raise ValueError("Execution cannot self-classify as rejected/unknown/authorized")
        self._serial += 1
        self._pending[self._serial] = (category, mechanism)
        self._events.append(("begin", self._serial, category, mechanism))
        return self._serial

    def finish(self, token, success):
        if type(success) is not bool:
            self.invalid("non-Boolean finish")
            raise TypeError("finish requires an exact bool")
        if type(token) is not int or token not in self._pending:
            self.invalid("unknown or duplicate finish")
            raise KeyError(token)
        category, mechanism = self._pending.pop(token)
        self._events.append(("finish", token, category if success else UNKNOWN))
        return category, mechanism

    def attempt(self):
        self._serial += 1
        token = self._serial
        self._attempts[token] = None
        return token

    def observe_attempt(self, token, outcome, layer):
        if token not in self._attempts or outcome not in (REJECTED, UNAUTHORIZED, UNKNOWN):
            self.invalid("invalid attempt observation")
            raise ValueError("Unregistered attempt/outcome")
        previous = self._attempts[token]
        if previous is not None and previous != outcome:
            self.invalid("conflicting attempt evidence")
            # Never erase an observed unauthorized crossing.
            outcome = UNAUTHORIZED if UNAUTHORIZED in (previous, outcome) else UNKNOWN
        self._attempts[token] = outcome
        self._events.append(("attempt", token, outcome, layer))

    def rejected(self, token=None, layer="sentinel"):
        token = self.attempt() if token is None else token
        self.observe_attempt(token, REJECTED, layer)
        return token

    def snapshot(self):
        counts = dict.fromkeys(CATEGORIES, 0)
        entries = Counter()
        for event in self._events:
            if event[0] == "finish":
                counts[event[2]] += 1
            elif event[0] == "invalid":
                counts[UNKNOWN] += 1
            elif event[0] == "entry":
                entries[event[1] + ":" + event[2]] += 1
        for outcome in self._attempts.values():
            counts[UNKNOWN if outcome is None else outcome] += 1
        counts[UNKNOWN] += len(self._pending)
        return {"categories": counts, "mechanism_entries": dict(sorted(entries.items())),
                "event_count": len(self._events),
                "units": "completed pytest items; rejected calls; unknown items; observed Python entries",
                "coverage": "closed admitted Phase-11 surface, not a sandbox; opaque callback objects excluded; child-process scope attested by owning tests"}


ACCOUNTING = Accounting()

# Closed callback vocabulary: no callable argument and no generic executor.
def admitted_lengths(values, *foreign_dependencies, **foreign_keywords):
    # Extras are rejection-only capture, never an accepted dependency protocol.
    # Do not inspect, format, look up attributes on, or invoke their values.
    if foreign_dependencies or foreign_keywords:
        ACCOUNTING.rejected(layer="admission")
        raise PermissionError("Phase-11 admits data only, not accounting dependencies")
    if type(values) not in (tuple, list) or any(type(v) not in (str, bytes) for v in values):
        ACCOUNTING.rejected(layer="admission")
        raise PermissionError("Phase-11 rejects opaque callback objects before consumption")
    return tuple(map(len, values))


class DirectFirewall:
    """Explicit lifecycle; CPython monitoring, not a universal native sandbox.

    Targets are identities captured BEFORE sentinel replacement. Python code
    identities also protect calls made through aliases and native callbacks.
    Opaque iterators are excluded at admitted_lengths, not introspected/consumed.
    """
    def __init__(self, audit, targets=()):
        self.audit = audit
        self.targets = tuple(targets)
        self.codes = {fn.__code__ for fn in targets if hasattr(fn, "__code__")}
        self.tool = None

    def deny(self, layer):
        self.audit.rejected(layer=layer)
        raise PermissionError("Phase-11 protected or unsupported execution route")

    def start(self, code, offset):
        frame = sys._getframe(1)
        label = mechanism(frame.f_globals.get("__name__", ""), code.co_name)
        if code in self.codes or (label and label not in ("synthetic_integration", "fixture_evaluation_entry")):
            self.audit.rejected(self.audit._frames.get(id(frame)), "python-start")
            raise PermissionError("Phase-11 protected Python entry")

    def call(self, code, offset, fn, first):
        if any(fn is target for target in self.targets):
            self.deny("direct-call")
        owner = getattr(fn, "__self__", None)
        owner_class = type(owner) if owner is not None else getattr(fn, "__objclass__", None)
        owner_module = getattr(owner_class, "__module__", "")
        if owner_module.startswith("numpy.random") and getattr(fn, "__name__", "") != "get_state":
            self.deny("native-rng-method")
        if getattr(fn, "__name__", "") in (
                "manual_seed", "seed", "set_state", "uniform_", "normal_", "random_",
                "bernoulli_", "exponential_", "cauchy_", "log_normal_", "geometric_"):
            if owner_module.startswith("torch"):
                self.deny("native-seed-method")
        # These dispatchers are not exposed as arbitrary callback APIs. The
        # sole admitted native callback is map(len, concrete strings) above.
        owned = code.co_filename.replace("\\", "/")
        if "/tests/phase11/" in owned and code is not admitted_lengths.__code__:
            dispatchers = (builtins.map, builtins.filter, functools.reduce, functools.partial,
                           builtins.iter, builtins.sorted, builtins.min, builtins.max,
                           operator.methodcaller, operator.attrgetter, operator.itemgetter)
            internal_sort = fn is builtins.sorted and code is Accounting.snapshot.__code__
            if (any(fn is d for d in dispatchers) and not internal_sort
                    or fn is list.sort
                    or isinstance(owner, list) and getattr(fn, "__name__", "") == "sort"):
                self.deny("unsupported-dispatch")

    def __enter__(self):
        if not hasattr(sys, "monitoring"):
            self.audit.invalid("required monitoring unavailable")
            raise RuntimeError("Phase-11 requires CPython sys.monitoring")
        m = sys.monitoring
        for tool in (4, 3, 2, 1, 0, 5):
            if m.get_tool(tool) is None:
                self.tool = tool
                break
        if self.tool is None:
            self.audit.invalid("no monitoring slot")
            raise RuntimeError("No Phase-11 monitoring slot available")
        m.use_tool_id(self.tool, "phase11-bounded")
        m.register_callback(self.tool, m.events.PY_START, self.start)
        m.register_callback(self.tool, m.events.CALL, self.call)
        m.set_events(self.tool, m.events.PY_START | m.events.CALL)
        return self

    def __exit__(self, *exc):
        m = sys.monitoring
        if m.get_events(self.tool) != m.events.PY_START | m.events.CALL:
            self.audit.invalid("monitoring lifecycle lost")
        m.set_events(self.tool, 0)
        m.register_callback(self.tool, m.events.PY_START, None)
        m.register_callback(self.tool, m.events.CALL, None)
        m.free_tool_id(self.tool)


def classify_path(relative, closed_files):
    # A new path cannot acquire grandfathering by selecting a seed or calling
    # an old function. Only a baseline-owned test path is eligible.
    if relative.startswith("tests/phase11/"):
        return BOUNDED
    if relative.startswith("tests/") and relative in closed_files:
        return GRANDFATHERED
    raise PermissionError("Unregistered regression call site")


CANDIDATE_PATHS = frozenset((
    "docs/phase11.md", "tests/phase11/conftest.py",
    "tests/phase11/contract_ledger.json", "tests/phase11/phase11_accounting.py",
    "tests/phase11/test_phase11_contract_firewall.py",
    "tests/phase11/test_phase11_identities.py",
    "tests/phase11/test_phase11_import_hygiene.py",
    "tests/phase11/test_phase11_integration.py",
    "tests/phase11/test_phase11_numerics.py",
    "tests/phase11/test_phase11_reference_replay.py",
))


def read_candidate_manifest(path, digest):
    # External reviewed input avoids a self-referential candidate-file hash.
    if not path or not digest:
        raise RuntimeError("Explicit frozen candidate manifest and SHA-256 required")
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("Candidate manifest identity changed")
    candidate = json.loads(data)
    if type(candidate) is not dict or set(candidate) != CANDIDATE_PATHS:
        raise RuntimeError("Candidate manifest inventory changed")
    if any(type(v) is not str or len(v) != 64 or
           any(c not in "0123456789abcdef" for c in v) for v in candidate.values()):
        raise RuntimeError("Invalid candidate SHA-256")
    return candidate


def verify_baseline(root, ledger, candidate):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    if git("rev-parse", "HEAD").decode().strip() != ledger["baseline"]:
        raise RuntimeError("Baseline HEAD changed")
    if git("rev-parse", "HEAD^{tree}").decode().strip() != ledger["tree"]:
        raise RuntimeError("Baseline tree changed")
    if set(candidate) != CANDIDATE_PATHS:
        raise RuntimeError("Candidate inventory changed")
    tracked = set(git("ls-tree", "-r", "--name-only", "HEAD").decode().splitlines())
    if tracked != set(ledger["closed_files"]):
        raise RuntimeError("Closed inventory changed")
    for path, digest in ledger["closed_files"].items():
        for data in (git("show", "HEAD:" + path), git("show", ":" + path),
                     (root / path).read_bytes()):
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError("Closed bytes changed: " + path)
    # Candidate additions may be all unstaged, all staged, or mixed. A staged
    # candidate blob must equal its reviewed identity, not merely its worktree.
    indexed = set(git("ls-files", "-z").decode().split("\0")) - {""}
    untracked = set(git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")) - {""}
    if indexed - tracked != CANDIDATE_PATHS - untracked or untracked - CANDIDATE_PATHS:
        raise RuntimeError("Unexpected or missing candidate paths")
    for path, digest in candidate.items():
        data = (root / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("Candidate bytes changed: " + path)
        if path in indexed and hashlib.sha256(git("show", ":" + path)).hexdigest() != digest:
            raise RuntimeError("Staged candidate bytes changed: " + path)
    # Reject mode changes, renames, deletions, and every non-candidate delta.
    for row in git("diff", "HEAD", "--raw", "--no-renames", "--no-abbrev").decode().splitlines():
        metadata, path = row.split("\t", 1)
        oldmode, newmode, oldoid, newoid, status = metadata.split()
        if path not in CANDIDATE_PATHS or status != "A" or newmode != "100644":
            raise RuntimeError("Unauthorized baseline delta: " + path)


def mechanism(module, name):
    if not module.startswith("inverse_em."):
        return None
    if module == "inverse_em.populations.generation" and name.startswith("generate"):
        return "population_generation"
    if module == "inverse_em.surrogates.data" and name == "generate_historical_sources":
        return "historical_coordinate_generation"
    if module == "inverse_em.physics.api" and name in ("evaluate", "superpose"):
        return "PhysicsTM_forward"
    if module.startswith("inverse_em.normalization") and name.startswith(("fit", "_fit")):
        return "normalizer_fitting_entry"
    if "checkpoint" in module and name.startswith(("load", "save")):
        return "checkpoint_io_entry"
    if module.startswith(("inverse_em.training", "inverse_em.classifier.training", "inverse_em.surrogates.training")) and name in ("run", "train", "train_epoch", "train_model", "fit"):
        return "training_entry"
    if module.startswith("inverse_em.artifact_integration") and name == "integrate":
        return "synthetic_integration"
    if module == "inverse_em.evaluation.execution" and name in ("run", "execute", "run_clean", "run_robustness"):
        return "fixture_evaluation_entry"
    if module.startswith(("inverse_em.noise", "inverse_em.localization")) and name in ("noise_event", "epoch_minibatches", "draw_standardized_directions", "draw_training_snr", "keyed_rng"):
        return "RNG_address_or_draw_entry"
    if name in ("initialize_localizer", "make_initialized_classifier"):
        return "historical_seeded_initialization"
    return None


def pytest_addoption(parser):
    parser.addoption("--phase11-accounting", default=None, help="External JSON receipt path")
    parser.addoption("--phase11-candidate-manifest", default=None)
    parser.addoption("--phase11-candidate-sha256", default=None)


def pytest_sessionstart(session):
    session.config._phase11_ledger = json.loads(LEDGER_PATH.read_text())
    candidate = read_candidate_manifest(
        session.config.getoption("--phase11-candidate-manifest"),
        session.config.getoption("--phase11-candidate-sha256"))
    verify_baseline(ROOT, session.config._phase11_ledger, candidate)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    relative = Path(item.path).resolve().relative_to(ROOT).as_posix()
    category = classify_path(relative, item.config._phase11_ledger["closed_files"])
    token = ACCOUNTING.begin(category, item.nodeid)
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "call":
            label = mechanism(frame.f_globals.get("__name__", ""), frame.f_code.co_name)
            if label:
                if category == BOUNDED and label not in ("synthetic_integration", "fixture_evaluation_entry"):
                    # Backstop for an alias captured before the autouse guards.
                    # A profile 'call' event occurs before the function body.
                    # A profile event precedes PY_START; it is NOT evidence of
                    # body execution. Persistent enforcement belongs to monitoring.
                    ACCOUNTING._frames[id(frame)] = ACCOUNTING.attempt()
                    return
                ACCOUNTING.record_entry(category, label)
        elif event == "return" and id(frame) in ACCOUNTING._frames:
            attempt = ACCOUNTING._frames.pop(id(frame))
            if ACCOUNTING._attempts[attempt] is None:
                ACCOUNTING.observe_attempt(attempt, UNKNOWN, "unreconciled-profile-entry")
        if previous:
            previous(frame, event, arg)

    sys.setprofile(observe)
    try:
        yield
    finally:
        if sys.getprofile() is not observe:
            ACCOUNTING.invalid("profile observation lifecycle lost")
        sys.setprofile(previous)
        reports = getattr(item, "_phase11_reports", [])
        ACCOUNTING.finish(token, len(reports) == 3 and all(r.passed for r in reports))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if not hasattr(item, "_phase11_reports"):
        item._phase11_reports = []
    item._phase11_reports.append(outcome.get_result())


def pytest_sessionfinish(session, exitstatus):
    report = ACCOUNTING.snapshot()
    report["pytest_exitstatus"] = int(exitstatus)
    if exitstatus:
        report["categories"][UNKNOWN] += 1
    destination = session.config.getoption("--phase11-accounting")
    if destination:
        path = Path(destination).resolve()
        if path.is_relative_to(ROOT):
            raise RuntimeError("Accounting output must stay outside repository")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line("PHASE11 ACCOUNTING " + json.dumps(ACCOUNTING.snapshot()["categories"], sort_keys=True))
