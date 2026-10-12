"""
experiments.py -- Fresh measurements of DA-ADKG v2 for the revised Sec. V-VI.

Every number this module produces comes from executing the v2 code
(network.run_dkg_async, real cryptography) or from the validated honest-case
estimator (network.estimate_performance).  Nothing is calibrated to, or
checked against, a value from the submitted manuscript.

WHAT IT MEASURES
    1. Executed scenarios (Sec. VI-D).  Each scenario runs the protocol with
       real cryptography over many seeds.  Reported per (scenario, t):
       success with a Wilson interval, completion time, gas by category,
       bytes, and ACCOUNTABILITY INVARIANTS checked on every run (no honest
       party penalised; every commission-faulty dealer excluded and slashed
       exactly once; omission-faulty dealers excluded but not slashed; stake
       conservation; reconstructed secret matches and g^s is the key).
       Scenarios inside the model's assumptions (<= t faults, network
       synchronous within the planned windows) are kept apart from scenarios
       that deliberately break an assumption; only the latter may fail.
    2. Large-n performance (Table II/III).  Honest-case estimator at the n
       the published anchors use (n = 3t+1 exactly where possible; 128 -> 127
       and 32 -> 31 are flagged), per chain profile: completion, bytes sent
       per node (bytes, MB and MiB), P2P total, ledger writes and reads, gas.
       The estimator is re-checked against executed runs at small n.
    3. Scaling (Table III).  Log-log slopes with standard errors (no model
       selection by R^2 between nested polynomials).
    4. Dependability (Sec. V).  A phase-type CTMC whose phase durations are
       MEASURED from executed runs, whose failure rates come from FAULT
       INJECTION (conditional failure probability per phase), and whose
       predictions are VALIDATED on held-out runs with Poisson fault arrivals.
       The exogenous fault rate is a sweep parameter, not a claimed value.
       Fault event: t+1 random nodes cut off for the longest planned window
       (sensitivity at 0.5x and 2x).  Reported: success per attempt and the
       expected time to a usable key; steady-state availability is not
       reported (undefined for a one-shot DKG without a re-keying workload).
    5. Recovery (Sec. V-D).  Measured: omission recovery through the ledger
       (complaint -> answer), and failure detection + re-run time.

PROVENANCE
    Every results file records: SHA-256 of each source file, library
    versions, the unit costs (values and source), chain and network
    parameters, seeds, and the byte unit.  Results without provenance must
    not be reported.

Run:
    python3 experiments.py                       # self-tests (a few minutes)
    python3 experiments.py --run quick --out r.json
    python3 experiments.py --run paper --out r.json --workers 2   # hours
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy import stats
from scipy.linalg import expm

import baselines as bl
import network as nw
from protocol import (
    WINDOW_ORDER,
    Strategy,
    SystemParams,
    WindowSchedule,
    gas_anchor,
    gas_answer,
    gas_complaint,
    gas_dispute,
    gas_pok_fraud,
    gas_register,
    gas_submit_key,
)

SCHEMA = "dadkg-experiments/2"
PHASES = tuple(w.name for w in WINDOW_ORDER)      # REGISTRATION .. KEY_DERIVATION
# Paper-level states (protocol.Phase): S0 = REGISTRATION, S1 = DEALING,
# S2 = COMPLAINT + ANSWER + DISPUTE, S3 = QUALIFICATION (instantaneous: fixed
# at the block that closes DISPUTE), S4 = KEY_DERIVATION.
PAPER_STATE_OF_PHASE = {"REGISTRATION": "S0", "DEALING": "S1", "COMPLAINT": "S2",
                        "ANSWER": "S2", "DISPUTE": "S2", "KEY_DERIVATION": "S4"}


# ==========================================================================
# 0. Provenance and the measurement context.
# ==========================================================================

_SOURCES = ("crypto.py", "protocol.py", "network.py", "baselines.py", "experiments.py")


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def provenance(ctx: "Context") -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    hashes = {}
    for f in _SOURCES:
        with open(os.path.join(here, f), "rb") as fh:
            hashes[f] = hashlib.sha256(fh.read()).hexdigest()
    return {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha256": hashes,
        "python": sys.version.split()[0], "numpy": np.__version__,
        "scipy": scipy.__version__, "machine": platform.machine(),
        "system": platform.system(), "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "unit_costs": asdict(ctx.unit),
        "cost_model_fit_ts": list(ctx.model.fit_ts),
        "net": asdict(ctx.net),
        "latency_bound_s": ctx.net.latency_bound(),
        "byte_unit": "bytes; MB = 1e6 bytes; MiB = 2^20 bytes",
        "time_unit": "simulated seconds",
    }


@dataclass(frozen=True)
class Context:
    """Everything a run depends on besides (params, scenario, seed).  The
    same Context must be used for every number that is compared."""
    unit: nw.UnitCosts
    model: nw.CostModel
    net: nw.NetTiming = nw.DEFAULT_NET


def make_context(unit: Optional[nw.UnitCosts] = None,
                 net: nw.NetTiming = nw.DEFAULT_NET, calib_reps: int = 60) -> Context:
    """Unit costs are MEASURED once (pure-Python reference on this machine,
    decision D7e-2) and shared by every run that is compared."""
    unit = unit or nw.calibrate_unit_costs(reps=calib_reps)
    return Context(unit=unit, model=nw.fit_cost_model(unit=unit), net=net)


def plan_windows(p: SystemParams, chain: nw.ChainProfile, ctx: Context,
                 scale: float = 1.0) -> WindowSchedule:
    """The window plan run_dkg_async would use (recommended_windows), made
    explicit so that injected faults can be placed relative to it.  `scale`
    < 1 deliberately undersizes the plan (an out-of-assumption scenario)."""
    w = nw.recommended_windows(p, chain, ctx.net, ctx.model.phase_compute(p, ctx.unit))
    if scale == 1.0:
        return w
    return WindowSchedule(*(max(1, math.floor(getattr(w, f.lower()) * scale))
                            for f in PHASES))


def window_bounds(w: WindowSchedule, chain: nw.ChainProfile) -> Dict[str, Tuple[float, float]]:
    out, h = {}, 0
    for f in PHASES:
        b = getattr(w, f.lower())
        out[f] = (h * chain.block_time, (h + b) * chain.block_time)
        h += b
    return out


def stable_seed(*parts: object) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


# ==========================================================================
# 1. Scenarios.
# ==========================================================================

COMMISSION = {Strategy.CORRUPT_ALL, Strategy.SELECTIVE, Strategy.BAD_ANSWER,
              Strategy.BAD_POK}
OMISSION = {Strategy.SILENT, Strategy.WITHHOLD}


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    within_assumptions: bool


SCENARIOS: Dict[str, ScenarioSpec] = {s.name: s for s in (
    ScenarioSpec("honest", "No faults.", True),
    ScenarioSpec("crash", "k nodes register, then stop (SILENT).", True),
    ScenarioSpec("commission_all", "k dealers send signed invalid shares to "
                 "everyone (CORRUPT_ALL).", True),
    ScenarioSpec("commission_selective", "k dealers send signed invalid shares "
                 "to t honest targets, valid ones to the rest (SELECTIVE).", True),
    ScenarioSpec("omission", "k dealers send nothing to t honest targets and "
                 "never answer (WITHHOLD).", True),
    ScenarioSpec("omission_cured", "k dealers withhold from t targets, then "
                 "answer on-chain (WITHHOLD_THEN_ANSWER).", True),
    ScenarioSpec("bad_answer", "k dealers withhold, then answer with an "
                 "invalid share (BAD_ANSWER).", True),
    ScenarioSpec("bad_pok", "k dealers anchor an invalid PoK (BAD_POK).", True),
    ScenarioSpec("false_accuser", "k nodes dispute one honest dealer's VALID "
                 "share (FALSE_ACCUSER).", True),
    ScenarioSpec("false_complainer", "k nodes claim non-delivery against t "
                 "honest dealers (FALSE_COMPLAINER).", True),
    ScenarioSpec("mixed", "k faulty nodes cycling CORRUPT_ALL, WITHHOLD, "
                 "FALSE_ACCUSER, SELECTIVE, FALSE_COMPLAINER.", True),
    ScenarioSpec("absent", "k nodes never register (crash before the run).", True),
    # ---- deliberately outside the model's assumptions -------------------
    ScenarioSpec("slow_links", "k honest nodes with 20x outbound latency, "
                 "beyond the bound Delta used to plan the windows.", False),
    ScenarioSpec("long_partition", "t+1 honest nodes cut off from everyone, "
                 "chain included, from DEALING start to COMPLAINT end.", False),
    ScenarioSpec("crash_beyond_t", "t+1 honest nodes crash when "
                 "KEY_DERIVATION opens.", False),
    ScenarioSpec("undersized_windows", "Window plan scaled by 0.25.", False),
)}

_MIXED = (Strategy.CORRUPT_ALL, Strategy.WITHHOLD, Strategy.FALSE_ACCUSER,
          Strategy.SELECTIVE, Strategy.FALSE_COMPLAINER)
_SIMPLE = {"crash": Strategy.SILENT, "commission_all": Strategy.CORRUPT_ALL,
           "commission_selective": Strategy.SELECTIVE, "omission": Strategy.WITHHOLD,
           "omission_cured": Strategy.WITHHOLD_THEN_ANSWER,
           "bad_answer": Strategy.BAD_ANSWER, "bad_pok": Strategy.BAD_POK,
           "false_accuser": Strategy.FALSE_ACCUSER,
           "false_complainer": Strategy.FALSE_COMPLAINER}


@dataclass
class Setup:
    adversaries: Dict[int, nw.AdversaryProfile]
    partitions: List[nw.Partition]
    windows: WindowSchedule
    faulty: Dict[int, str]          # node -> label, for every node outside "honest"


def build_setup(name: str, p: SystemParams, k: int, chain: nw.ChainProfile,
                ctx: Context, rng: random.Random) -> Setup:
    """Faulty nodes and targets are drawn from `rng` (recorded per run)."""
    if name not in SCENARIOS:
        raise KeyError(name)
    n, t = p.n, p.t
    if not 0 <= k <= t:
        raise ValueError("k must lie in [0, t]")
    nodes = list(range(1, n + 1))
    windows = plan_windows(p, chain, ctx, scale=0.25 if name == "undersized_windows" else 1.0)
    wb = window_bounds(windows, chain)
    adv: Dict[int, nw.AdversaryProfile] = {}
    parts: List[nw.Partition] = []
    faulty: Dict[int, str] = {}

    if name in _SIMPLE or name == "mixed":
        bad = sorted(rng.sample(nodes, k))
        honest = [x for x in nodes if x not in bad]
        for idx, b in enumerate(bad):
            s = _MIXED[idx % len(_MIXED)] if name == "mixed" else _SIMPLE[name]
            if s is Strategy.FALSE_ACCUSER:
                targets = (rng.choice(honest),)
            elif s in (Strategy.SELECTIVE, Strategy.WITHHOLD,
                       Strategy.WITHHOLD_THEN_ANSWER, Strategy.BAD_ANSWER,
                       Strategy.FALSE_COMPLAINER):
                targets = tuple(sorted(rng.sample(honest, min(t, len(honest)))))
            else:
                targets = ()
            adv[b] = nw.AdversaryProfile(s, targets=targets)
            faulty[b] = s.name
    elif name == "slow_links":
        for b in sorted(rng.sample(nodes, k)):
            adv[b] = nw.AdversaryProfile(Strategy.HONEST, delay_factor=20.0)
            faulty[b] = "SLOW(x20)"
    elif name == "absent":
        for b in sorted(rng.sample(nodes, k)):
            adv[b] = nw.AdversaryProfile(Strategy.HONEST, crash_time=0.0)
            faulty[b] = "ABSENT"
    elif name == "long_partition":
        group = frozenset(rng.sample(nodes, t + 1))
        parts.append(nw.Partition(wb["DEALING"][0], wb["COMPLAINT"][1], group))
        faulty.update({g: "PARTITIONED" for g in group})
    elif name == "crash_beyond_t":
        for b in sorted(rng.sample(nodes, t + 1)):
            adv[b] = nw.AdversaryProfile(Strategy.HONEST,
                                         crash_time=wb["KEY_DERIVATION"][0])
            faulty[b] = "CRASH@KEY"
    return Setup(adv, parts, windows, faulty)


# ==========================================================================
# 2. Executed runs and accountability invariants.
# ==========================================================================


@dataclass
class RunRecord:
    scenario: str
    t: int
    n: int
    k: int
    chain: str
    seed: int
    faulty: Dict[str, str]
    within_assumptions: bool
    success: bool
    reason: str
    completion_time: Optional[float]
    qual: List[int]
    honest_excluded: List[int]
    commission_in_qual: List[int]
    penalties: Dict[str, float]
    verdicts: Dict[str, int]
    complaints: int
    invariants: Dict[str, bool]
    gas_total: int
    gas_by_category: Dict[str, int]
    sent_bytes_mean: float
    sent_bytes_max: int
    p2p_total_bytes: int
    ledger_write_bytes: int
    tx_rejected: Dict[str, int]
    honest_tx_rejected: int
    window_times: Dict[str, List[float]]
    recovery_latencies: List[float]       # complaint inclusion -> answer inclusion (s)
    compute_seconds_max: float
    rewards_to_faulty: float              # reporter rewards earned by faulty parties


def _invariants(p: SystemParams, r: nw.AsyncResult, faulty_strat: Dict[int, Strategy],
                honest: List[int], penalties: Dict[int, float],
                upheld: Dict[int, float]) -> Dict[str, bool]:
    commission = [i for i, s in faulty_strat.items() if s in COMMISSION]
    omission = [i for i, s in faulty_strat.items() if s in OMISSION]
    accusers = [i for i, s in faulty_strat.items() if s is Strategy.FALSE_ACCUSER]
    slashed_total = sum(d["slashed"] for d in r.disputes)
    rewards_total = sum(d["reward"] for d in r.disputes)
    # Replay the stake ledger in order: every slash must be the configured
    # fraction of the party's stake AT THAT TIME (a faulty dealer may first
    # have earned a reporter reward by disputing another faulty dealer).
    stake = {i: p.s_min for i in range(1, p.n + 1)}
    fractions_ok = True
    slashes_of: Counter = Counter()
    for d in r.disputes:
        if d["slashed"] <= 0:
            continue
        if d["verdict"] == "upheld":
            who, frac = d["dealer"], p.alpha
            slashes_of[who] += 1
        else:
            who, frac = d["accuser"], p.gamma
        fractions_ok &= abs(d["slashed"] - frac * stake[who]) < 1e-9
        stake[who] -= d["slashed"]
        if d["verdict"] == "upheld":
            stake[d["accuser"]] += d["reward"]
            fractions_ok &= abs(d["reward"] - p.reporter_share * d["slashed"]) < 1e-9
    inv = {
        "no_honest_party_penalised": all(penalties.get(h, 0.0) == 0.0 for h in honest),
        "honest_dealers_in_qual": all(h in r.qual for h in honest),
        "commission_excluded": all(i not in r.qual for i in commission),
        "commission_slashed_exactly_once": all(slashes_of[i] == 1 and upheld.get(i, 0) > 0
                                               for i in commission),
        "slash_and_reward_fractions": fractions_ok,
        "omission_excluded_not_slashed": all(i not in r.qual and penalties.get(i, 0.0) == 0.0
                                             for i in omission),
        "false_accuser_penalised": all(penalties.get(i, 0.0) > 0.0 for i in accusers),
        "stake_conserved": abs(r.treasury - (slashed_total - rewards_total)) < 1e-9,
        "audit_secret_and_key": bool(r.audit) and all(r.audit.values()),
    }
    return inv


def execute(scenario: str, t: int, k: int, rep: int, chain: nw.ChainProfile,
            ctx: Context, audit: bool = True) -> RunRecord:
    p = SystemParams(t=t)
    seed = stable_seed(scenario, t, k, chain.name, rep)
    rng = random.Random(seed)
    su = build_setup(scenario, p, k, chain, ctx, rng)
    r = nw.run_dkg_async(p, adversaries=su.adversaries, partitions=su.partitions,
                         chain=chain, net=ctx.net, unit_costs=ctx.unit,
                         windows=su.windows, cost_model=ctx.model, seed=seed,
                         audit_reconstruct=audit)
    faulty_strat = {i: pr.strategy for i, pr in su.adversaries.items()
                    if pr.strategy is not Strategy.HONEST}
    honest = [i for i in range(1, p.n + 1) if i not in su.faulty]
    penalties: Dict[int, float] = Counter()
    upheld: Dict[int, float] = Counter()
    verdicts: Counter = Counter()
    for d in r.disputes:
        verdicts[f"{d['kind']}:{d['verdict']}"] += 1
        if d["verdict"] == "upheld":
            penalties[d["dealer"]] += d["slashed"]
            upheld[d["dealer"]] += d["slashed"]
        elif d["slashed"] > 0:
            penalties[d["accuser"]] += d["slashed"]
    inv = _invariants(p, r, faulty_strat, honest, penalties, upheld)
    if not audit:
        inv.pop("audit_secret_and_key")
    bt = chain.block_time
    comp = {(e["accuser"], e["dealer"]): e["height"] for e in r.events
            if e["type"] == "Complaint"}
    rec_lat = [(e["height"] - comp[(e["recipient"], e["dealer"])]) * bt
               for e in r.events if e["type"] == "Answered"
               and (e["recipient"], e["dealer"]) in comp]
    honest_rej = sum(1 for e in r.events if e["type"] == "Rejected" and e["who"] in honest)
    sent = list(r.comm.sent.values()) or [0]
    return RunRecord(
        scenario=scenario, t=t, n=p.n, k=k, chain=chain.name, seed=seed,
        faulty={str(i): v for i, v in sorted(su.faulty.items())},
        within_assumptions=SCENARIOS[scenario].within_assumptions,
        success=r.success, reason=r.reason, completion_time=r.completion_time,
        qual=list(r.qual), honest_excluded=[h for h in honest if h not in r.qual],
        commission_in_qual=[i for i, s in faulty_strat.items()
                            if s in COMMISSION and i in r.qual],
        penalties={str(i): v for i, v in sorted(penalties.items()) if v},
        verdicts=dict(verdicts), complaints=len(r.complaints), invariants=inv,
        gas_total=r.gas_total, gas_by_category=dict(r.gas_by_category),
        sent_bytes_mean=float(statistics.mean(sent)), sent_bytes_max=int(max(sent)),
        p2p_total_bytes=r.comm.p2p_total, ledger_write_bytes=r.comm.ledger_writes,
        tx_rejected=dict(r.tx_rejected), honest_tx_rejected=honest_rej,
        window_times={k_: list(v) for k_, v in r.window_times.items()},
        recovery_latencies=rec_lat,
        compute_seconds_max=max(r.compute_seconds.values() or [0.0]),
        rewards_to_faulty=float(sum(d["reward"] for d in r.disputes
                                    if d["verdict"] == "upheld"
                                    and d["accuser"] in su.faulty)))


def _execute_job(args) -> RunRecord:
    return execute(*args)


# --------------------------------------------------------------------------
# Resumable execution.  A long run (--run paper, hours) can be interrupted;
# with a cache directory every finished job is appended to jobs.jsonl and the
# measured Context is stored once in context.json, so a restart re-uses both
# and only runs what is missing.  Jobs are deterministic given their
# arguments, the Context and the source code; the cache refuses to resume
# if any source file changed.
# --------------------------------------------------------------------------


def source_digest() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for f in _SOURCES:
        with open(os.path.join(here, f), "rb") as fh:
            h.update(f.encode() + b"\0" + fh.read())
    return h.hexdigest()


class JobCache:
    def __init__(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.dir = directory
        self.path = os.path.join(directory, "jobs.jsonl")
        self.done: Dict[str, object] = {}
        if os.path.exists(self.path) and os.path.getsize(self.path):
            with open(self.path, "rb+") as fh:          # isolate a torn last line
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    fh.write(b"\n")
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:                       # a torn last line is ignored
                        try:
                            k, v = json.loads(line)
                        except ValueError:
                            continue
                        self.done[k] = v

    @staticmethod
    def key(fn_name: str, job: tuple) -> str:
        def enc(x):
            if isinstance(x, Context):
                return "ctx"
            if isinstance(x, nw.ChainProfile):
                return x.name
            if isinstance(x, FaultEvent):
                return asdict(x)
            return x
        return hashlib.sha256(json.dumps([fn_name, [enc(x) for x in job]],
                                         sort_keys=True).encode()).hexdigest()

    def put(self, k: str, value: object) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps([k, value], default=_jsonable) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.done[k] = value


_CACHE: Optional[JobCache] = None          # set by run_all(cache=...)


def save_context(ctx: Context, directory: str) -> None:
    with open(os.path.join(directory, "context.json"), "w", encoding="utf-8") as fh:
        json.dump({"source_digest": source_digest(), "unit": asdict(ctx.unit),
                   "net": asdict(ctx.net),
                   "model": {"coeffs": ctx.model.coeffs, "fit_ts": list(ctx.model.fit_ts)}},
                  fh, indent=1)


def load_context(directory: str) -> Optional[Context]:
    path = os.path.join(directory, "context.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if d["source_digest"] != source_digest():
        raise RuntimeError(f"{directory}: source code changed since this cache was "
                           "started; use a new cache directory")
    model = nw.CostModel(coeffs={a: {k: tuple(v) for k, v in ks.items()}
                                 for a, ks in d["model"]["coeffs"].items()},
                         fit_ts=tuple(d["model"]["fit_ts"]))
    return Context(unit=nw.UnitCosts(**d["unit"]), model=model,
                   net=nw.NetTiming(**d["net"]))


def _encode(fn, value):
    return asdict(value) if isinstance(value, RunRecord) else value


def _decode(fn, value):
    if fn is _execute_job:
        return RunRecord(**value)
    if fn is _injection_job:
        return tuple(value)
    return value


def _map(fn, jobs: Sequence[tuple], workers: int, log=None) -> list:
    """Deterministic regardless of `workers`: each job carries its own seed
    and the shared Context (unit costs and cost model are passed, never
    re-measured inside a worker); results keep the job order.  With an
    active cache, finished jobs are skipped and new results are persisted."""
    cache = _CACHE
    keys = [JobCache.key(fn.__name__, j) for j in jobs] if cache else [None] * len(jobs)
    todo = [i for i, k in enumerate(keys) if not (cache and k in cache.done)]
    out: List[object] = [None] * len(jobs)
    for i, k in enumerate(keys):
        if cache and k in cache.done:
            out[i] = _decode(fn, cache.done[k])
    t0 = time.time()

    def store(i, v, n_done):
        out[i] = v
        if cache:
            cache.put(keys[i], _encode(fn, v))
        if log and (n_done % 50 == 0 or n_done == len(todo)):
            el = time.time() - t0
            log(f"    {fn.__name__}: {n_done}/{len(todo)} new jobs, "
                f"{el:.0f} s elapsed, ~{el / n_done * (len(todo) - n_done):.0f} s left")

    if workers <= 1:
        for c, i in enumerate(todo, 1):
            store(i, fn(jobs[i]), c)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for c, (i, v) in enumerate(zip(todo, ex.map(fn, [jobs[i] for i in todo],
                                                          chunksize=1)), 1):
                store(i, v, c)
    return out


def run_jobs(jobs: Sequence[tuple], workers: int = 1, log=None) -> List[RunRecord]:
    return _map(_execute_job, jobs, workers, log)


# ==========================================================================
# 3. Statistics.
# ==========================================================================


def wilson(successes: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(0.5 + conf / 2)
    ph = successes / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def zero_failure_upper_bound(n: int, conf: float = 0.95) -> float:
    """Exact (Clopper-Pearson) upper bound on the failure probability after
    n runs with no failure: 1 - (1-conf)^(1/n)  (~3/n at 95%)."""
    return 1.0 - (1.0 - conf) ** (1.0 / n)


def runs_for_failure_bound(p_max: float, conf: float = 0.95) -> int:
    """Failure-free runs needed so that the bound above is <= p_max."""
    return math.ceil(math.log(1.0 - conf) / math.log(1.0 - p_max))


def describe(xs: Sequence[float]) -> dict:
    a = np.asarray([x for x in xs if x is not None], float)
    if a.size == 0:
        return {"count": 0}
    return {"count": int(a.size), "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "median": float(np.median(a)),
            "p95": float(np.quantile(a, 0.95)), "max": float(a.max())}


@dataclass
class Comparison:
    label: str
    mean_a: float
    mean_b: float
    diff: float
    diff_ci95: Tuple[float, float]
    welch_t: Optional[float]
    welch_p: Optional[float]
    cohens_d: Optional[float]
    note: str = ""


def compare_samples(label: str, a: Sequence[float], b: Sequence[float],
                    seed: int = 0, n_boot: int = 4000) -> Comparison:
    """Mean difference with a bootstrap 95% CI; Welch t and Cohen's d only
    when both samples vary (the window-driven completion time is often
    constant across seeds, where a t-test is undefined)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = (rng.choice(a, (n_boot, a.size)).mean(1)
             - rng.choice(b, (n_boot, b.size)).mean(1))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if a.size > 1 and b.size > 1 and va > 0 and vb > 0:
        t_, p_ = stats.ttest_ind(a, b, equal_var=False)
        pooled = math.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2))
        d = float((a.mean() - b.mean()) / pooled)
        welch_t, welch_p, note = float(t_), float(p_), ""
    else:
        welch_t = welch_p = d = None
        note = "zero variance in a sample: t-test undefined; difference reported"
    return Comparison(label, float(a.mean()), float(b.mean()), float(a.mean() - b.mean()),
                      (float(lo), float(hi)), welch_t, welch_p, d, note)


def holm(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (uniformly more powerful than
    Bonferroni, same family-wise guarantee)."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adj[idx] = running
    return adj


def fisher_success(s_a: int, n_a: int, s_b: int, n_b: int) -> float:
    """Two-sided Fisher exact p-value for two success proportions (valid at
    the small counts and 0/100% rates seen here, unlike chi-square)."""
    return float(stats.fisher_exact([[s_a, n_a - s_a], [s_b, n_b - s_b]])[1])


# ==========================================================================
# 4. Summaries of executed runs.
# ==========================================================================


def summarise(records: Sequence[RunRecord]) -> List[dict]:
    groups: Dict[Tuple[str, int, int, str], List[RunRecord]] = {}
    for r in records:
        groups.setdefault((r.scenario, r.t, r.k, r.chain), []).append(r)
    out = []
    for (sc, t, k, ch), rs in sorted(groups.items()):
        s = sum(r.success for r in rs)
        lo, hi = wilson(s, len(rs))
        inv_names = sorted(set().union(*(r.invariants for r in rs)))
        inv = {name: sum(r.invariants.get(name, False) for r in rs) for name in inv_names}
        out.append({
            "scenario": sc, "t": t, "n": 3 * t + 1, "k": k, "chain": ch,
            "within_assumptions": SCENARIOS[sc].within_assumptions,
            "runs": len(rs), "successes": s, "success_rate": s / len(rs),
            "success_wilson95": [lo, hi],
            "failure_upper95_if_none": (zero_failure_upper_bound(len(rs))
                                        if s == len(rs) else None),
            "failure_reasons": dict(Counter(r.reason for r in rs if not r.success)),
            "invariants_held": inv,
            "completion_s": describe([r.completion_time for r in rs]),
            "gas_total": describe([r.gas_total for r in rs]),
            "gas_by_category_mean": {c: float(np.mean([r.gas_by_category.get(c, 0)
                                                       for r in rs]))
                                     for c in sorted(set().union(*(r.gas_by_category
                                                                   for r in rs)))},
            "sent_bytes_per_node_mean": describe([r.sent_bytes_mean for r in rs]),
            "ledger_write_bytes": describe([r.ledger_write_bytes for r in rs]),
            "honest_tx_rejected": sum(r.honest_tx_rejected for r in rs),
            "runs_with_rewards_to_faulty": sum(r.rewards_to_faulty > 0 for r in rs),
            "rewards_to_faulty_mean": float(np.mean([r.rewards_to_faulty for r in rs])),
            "honest_excluded_runs": sum(bool(r.honest_excluded) for r in rs),
            "commission_in_qual_runs": sum(bool(r.commission_in_qual) for r in rs),
            "recovery_latency_s": describe([x for r in rs for x in r.recovery_latencies]),
        })
    return out


def attack_overheads(records: Sequence[RunRecord]) -> List[dict]:
    """Per scenario vs `honest` at the same (t, chain), paired by repetition
    index: completion-time difference (bootstrap CI) and extra gas/bytes.
    Within assumptions the completion time is set by the window plan, so the
    cost of an attack shows up in gas and ledger bytes, not in latency."""
    by = {}
    for r in records:
        by.setdefault((r.scenario, r.t, r.chain), []).append(r)
    rows = []
    for (sc, t, ch), rs in sorted(by.items()):
        if sc == "honest" or ("honest", t, ch) not in by:
            continue
        base = by[("honest", t, ch)]
        a = [r.completion_time for r in rs if r.success]
        b = [r.completion_time for r in base if r.success]
        if len(a) < 2 or len(b) < 2:
            continue
        c = compare_samples(f"{sc} - honest (t={t}, {ch})", a, b,
                            seed=stable_seed(sc, t, ch))
        rows.append({"scenario": sc, "t": t, "chain": ch,
                     "completion": asdict(c),
                     "extra_gas_mean": float(np.mean([r.gas_total for r in rs])
                                             - np.mean([r.gas_total for r in base])),
                     "extra_ledger_bytes_mean": float(
                         np.mean([r.ledger_write_bytes for r in rs])
                         - np.mean([r.ledger_write_bytes for r in base]))})
    ps = [r["completion"]["welch_p"] for r in rows if r["completion"]["welch_p"] is not None]
    adj = iter(holm(ps)) if ps else iter(())
    for r in rows:
        r["completion"]["welch_p_holm"] = (next(adj) if r["completion"]["welch_p"]
                                           is not None else None)
    return rows


# ==========================================================================
# 5. Large-n performance and Table II assembly.
# ==========================================================================

# n = 3t+1 values matching the published anchors (4, 13, 25, 49: Zhang'23;
# 16, 64: Das'23; 256, 1024: EthDKG/Sober'23, H2CT) plus 127 (closest to 128).
PAPER_SIZES = (4, 13, 16, 25, 31, 49, 64, 127, 256, 1024)


def representable(n: int) -> Tuple[int, bool]:
    """(n' = 3t+1 closest to n from below, n' == n)."""
    t = bl.matching_t(n)
    if t is not None:
        return n, True
    t = max(1, (n - 1) // 3)
    return 3 * t + 1, False


def performance_row(n: int, chain: nw.ChainProfile, ctx: Context) -> dict:
    t = bl.matching_t(n)
    if t is None:
        raise ValueError(f"n={n} is not of the form 3t+1")
    p = SystemParams(t=t)
    e = nw.estimate_performance(p, chain=chain, net=ctx.net, unit_costs=ctx.unit,
                                cost_model=ctx.model)
    w = e.windows
    pc = e.compute_per_node
    return {
        "n": n, "t": t, "chain": chain.name, "block_time_s": chain.block_time,
        "finality_blocks": chain.finality_blocks,
        "completion_s": e.completion_time,
        "window_blocks": asdict(w),
        "compute_per_node_s": asdict(pc),
        "sent_bytes_per_node": e.per_node_sent_bytes,
        "sent_MB_per_node": e.per_node_sent_bytes / 1e6,
        "sent_MiB_per_node": e.per_node_sent_bytes / 2 ** 20,
        "p2p_total_bytes": e.p2p_total_bytes,
        "ledger_write_bytes": e.ledger_writes_bytes,
        "ledger_read_bytes_per_node": e.ledger_writes_bytes,
        "gas_total": e.gas_total, "gas_per_participant": e.gas_total / n,
        "unit_source": e.unit_source,
    }


def gas_per_transaction(t: int) -> dict:
    """DA-ADKG gas per transaction type at n = 3t+1 (protocol.py schedule,
    EIP-2929 prices).  For comparison with EthDKG/Sober'23 per-transaction
    anchors; note EthDKG's n = 2t+1 and commitment length t+1 differ."""
    p = SystemParams(t=t)
    return {"n": p.n, "t": t, "commitment_points": p.degree + 1,
            "register": gas_register(), "anchor": gas_anchor(),
            "complaint": gas_complaint(), "answer": gas_answer(p),
            "dispute": gas_dispute(p), "pok_fraud": gas_pok_fraud(p),
            "submit_key": gas_submit_key(False),
            "submit_key_finalizing": gas_submit_key(True)}


def table_ii(perf: Sequence[dict]) -> List[dict]:
    """Pairs every published anchor (runtime, bytes per node) with the fresh
    DA-ADKG row at the same n, or at the closest representable n with a flag.
    No test statistic: different testbeds and different meanings of time."""
    by_n = {(r["n"], r["chain"]): r for r in perf}
    chains = sorted({r["chain"] for r in perf})
    rows = []
    for a in bl.anchors():
        if a.metric not in ("runtime_s", "sent_per_node_MB") or a.n is None:
            continue
        n2, exact = representable(a.n)
        for ch in chains:
            d = by_n.get((n2, ch))
            if d is None:
                continue
            dval = d["completion_s"] if a.metric == "runtime_s" else d["sent_MB_per_node"]
            rows.append({
                "anchor_id": a.id, "system": a.system, "reported_in": bl.SOURCES[a.source].short,
                "metric": a.metric, "n": a.n, "anchor_value": a.value,
                "anchor_shares": a.shares, "anchor_group": a.group,
                "dadkg_n": n2, "n_differs": not exact, "chain": ch,
                "dadkg_value": dval,
                "dadkg_sent_MiB": d["sent_MiB_per_node"],
                "dadkg_ledger_write_bytes": d["ledger_write_bytes"],
                "caveat": ("runtime: published wall-clock on the source testbed vs "
                           "DA-ADKG simulated completion under the stated chain, "
                           "latency model and measured unit costs"
                           if a.metric == "runtime_s" else
                           "bytes: MB undefined in the source; DA-ADKG P2P bytes "
                           "exclude ledger traffic (reported separately)"),
            })
    return rows


def estimator_agreement(records: Sequence[RunRecord], ctx: Context) -> List[dict]:
    """Honest executed runs vs the estimator at the same n and chain."""
    out = []
    for r in records:
        if r.scenario != "honest" or not r.success:
            continue
        chain = {c.name: c for c in (nw.BFT_CHAIN, nw.ETHEREUM_L1)}[r.chain]
        e = performance_row(r.n, chain, ctx)
        out.append({"n": r.n, "seed": r.seed, "chain": r.chain,
                    "completion_sim": r.completion_time, "completion_est": e["completion_s"],
                    "sent_mean_sim": r.sent_bytes_mean,
                    "sent_est": e["sent_bytes_per_node"],
                    "gas_sim": r.gas_total, "gas_est": e["gas_total"],
                    "ledger_sim": r.ledger_write_bytes, "ledger_est": e["ledger_write_bytes"]})
    return out


# ==========================================================================
# 6. Scaling.
# ==========================================================================


def loglog_slope(ns: Sequence[float], ys: Sequence[float]) -> dict:
    """Least-squares slope of log y on log n with its standard error: the
    empirical growth exponent over the range given."""
    x, y = np.log(np.asarray(ns, float)), np.log(np.asarray(ys, float))
    res = stats.linregress(x, y)
    return {"exponent": float(res.slope), "stderr": float(res.stderr),
            "r2": float(res.rvalue ** 2), "n_range": [float(min(ns)), float(max(ns))]}


def scaling(perf: Sequence[dict], n_min: int = 64) -> dict:
    out = {}
    for ch in sorted({r["chain"] for r in perf}):
        rows = sorted((r for r in perf if r["chain"] == ch and r["n"] >= n_min),
                      key=lambda r: r["n"])
        if len(rows) < 3:
            continue
        ns = [r["n"] for r in rows]
        out[ch] = {
            "sent_bytes_per_node": loglog_slope(ns, [r["sent_bytes_per_node"] for r in rows]),
            "p2p_total_bytes": loglog_slope(ns, [r["p2p_total_bytes"] for r in rows]),
            "ledger_write_bytes": loglog_slope(ns, [r["ledger_write_bytes"] for r in rows]),
            "gas_total": loglog_slope(ns, [r["gas_total"] for r in rows]),
            "completion_s": loglog_slope(ns, [r["completion_s"] for r in rows]),
            "completion_breakdown_s": [
                {"n": r["n"],
                 "windows_before_key": sum(r["window_blocks"][f.lower()]
                                           for f in PHASES[:-1]) * r["block_time_s"],
                 "key_phase": r["completion_s"] - sum(
                     r["window_blocks"][f.lower()] for f in PHASES[:-1]) * r["block_time_s"],
                 "verify_compute": r["compute_per_node_s"]["verify"],
                 "key_verify_compute": r["compute_per_node_s"]["key_verify"]}
                for r in rows],
        }
    return out


# ==========================================================================
# 7. Dependability: phase-type CTMC, fault injection, held-out validation.
# ==========================================================================


def measure_phase_durations(records: Sequence[RunRecord]) -> Dict[str, List[float]]:
    """Durations of the six phases in successful runs: the first five are the
    window lengths actually used; KEY_DERIVATION ends when the key is final."""
    out: Dict[str, List[float]] = {f: [] for f in PHASES}
    for r in records:
        if not r.success:
            continue
        for f in PHASES[:-1]:
            b, e = r.window_times[f]
            out[f].append(e - b)
        out["KEY_DERIVATION"].append(r.completion_time - r.window_times["KEY_DERIVATION"][0])
    return out


@dataclass(frozen=True)
class PhaseTypeCTMC:
    """Sequential phases j = 0..5, each an Erlang chain of `stages` stages
    with rate stages/mean_j, plus absorbing SF (success) and Sfail.  The
    failure rate out of every stage of phase j is mu[j] (per second).
    stages = 1 is the exponential-phase CTMC of the manuscript; large
    `stages` approaches the deterministic window durations of v2."""
    means: Tuple[float, ...]
    mu: Tuple[float, ...]
    stages: int = 1

    @property
    def size(self) -> int:
        return len(self.means) * self.stages + 2

    def generator(self) -> np.ndarray:
        S = self.size
        sf, fail = S - 2, S - 1
        Q = np.zeros((S, S))
        s = 0
        for j, m in enumerate(self.means):
            rate = self.stages / m
            for _ in range(self.stages):
                Q[s, s + 1 if s + 1 < sf else sf] = rate
                Q[s, fail] += self.mu[j]
                s += 1
        np.fill_diagonal(Q, 0.0)
        np.fill_diagonal(Q, -Q.sum(axis=1))
        return Q

    def success_probability(self) -> float:
        p = 1.0
        for m, mu in zip(self.means, self.mu):
            r = self.stages / m
            p *= (r / (r + mu)) ** self.stages
        return p

    def reliability(self, tau: float) -> float:
        """P(key finalized by time tau)."""
        return float(expm(self.generator() * tau)[0, self.size - 2])

    def passage_cdf(self, taus: Sequence[float]) -> np.ndarray:
        """CDF of the completion time with no failures (mu = 0)."""
        q = replace(self, mu=tuple(0.0 for _ in self.means))
        G = q.generator()
        return np.array([expm(G * x)[0, q.size - 2] if x > 0 else 0.0 for x in taus])

    def mean_completion(self) -> float:
        return float(sum(self.means))


def expected_time_to_key(p_success: float, completion_s: float, attempt_s: float) -> float:
    """Expected time until a usable key exists (decision D7e-3, replacing
    steady-state availability, which is undefined for a one-shot DKG without
    an assumed re-keying workload).  Attempts are independent; a failed
    attempt is detected when its KEY_DERIVATION window closes (`attempt_s`,
    the planned length) and a new session starts at once:
        E[T] = completion + (1/p - 1) * attempt_s."""
    if p_success <= 0.0:
        return math.inf
    return completion_s + (1.0 / p_success - 1.0) * attempt_s


def ks_validation(model: PhaseTypeCTMC, completions: Sequence[float]) -> dict:
    """Kolmogorov-Smirnov distance between the model's failure-free
    completion-time CDF and the executed runs' completion times."""
    xs = np.sort(np.asarray(completions, float))
    F = model.passage_cdf(xs)
    n = xs.size
    d = float(max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n)))
    p = float(stats.kstwo.sf(d, n))
    return {"stages": model.stages, "ks_D": d, "ks_p": p, "n": n,
            "model_mean": model.mean_completion(), "sample_mean": float(xs.mean()),
            "model_sd": float(math.sqrt(sum(m * m / model.stages for m in model.means))),
            "sample_sd": float(xs.std(ddof=1)) if n > 1 else 0.0}


@dataclass(frozen=True)
class FaultEvent:
    """Exogenous fault used for the dependability model: a partition that
    cuts `group_size` random nodes off from everyone (chain included) for
    `duration` seconds.  group_size > t exceeds the fault threshold."""
    group_size: int
    duration: float


def default_fault_event(t: int, chain: nw.ChainProfile, ctx: Context,
                        duration_factor: float = 1.0) -> FaultEvent:
    """The Sec. V fault event (decision D7e-1): a correlated outage of t+1
    random nodes -- the smallest group that exceeds the fault threshold
    (e.g. a data-centre or region outage) -- lasting `duration_factor` times
    the LONGEST planned window, so a single event can span a whole window.
    Tied to the protocol's own time scale, hence comparable across t and
    chains; sensitivity is reported for factors 0.5 and 2."""
    exp = phase_exposure(t, chain, ctx)
    return FaultEvent(group_size=t + 1, duration=duration_factor * max(exp.values()))


def _run_with_partitions(t: int, chain: nw.ChainProfile, ctx: Context,
                         parts: List[nw.Partition], seed: int) -> Tuple[bool, str, dict]:
    p = SystemParams(t=t)
    w = plan_windows(p, chain, ctx)
    r = nw.run_dkg_async(p, partitions=parts, chain=chain, net=ctx.net,
                         unit_costs=ctx.unit, windows=w, cost_model=ctx.model,
                         seed=seed)
    return r.success, r.reason, {k: list(v) for k, v in r.window_times.items()}


def _injection_job(args) -> Tuple[str, bool, float]:
    t, chain, ctx, ev, phase, rep = args
    p = SystemParams(t=t)
    wb = window_bounds(plan_windows(p, chain, ctx), chain)
    seed = stable_seed("inject", t, chain.name, ev.group_size, ev.duration, phase, rep)
    rng = random.Random(seed)
    b, e = wb[phase]
    start = rng.uniform(b, e)
    group = frozenset(rng.sample(range(1, p.n + 1), ev.group_size))
    ok, _, wt = _run_with_partitions(t, chain, ctx,
                                     [nw.Partition(start, start + ev.duration, group)], seed)
    detect = None if ok else wt["KEY_DERIVATION"][1] - start
    return phase, ok, detect


def conditional_failure(t: int, chain: nw.ChainProfile, ctx: Context, ev: FaultEvent,
                        reps: int, workers: int = 1) -> Dict[str, dict]:
    """P(run fails | one fault event starting uniformly inside phase j),
    measured by injection; also the detection delay of each failure (event
    start -> KEY_DERIVATION window close, when the missing key is certain)."""
    jobs = [(t, chain, ctx, ev, ph, i) for ph in PHASES for i in range(reps)]
    res = _map(_injection_job, jobs, workers)
    out = {}
    for ph in PHASES:
        rs = [r for r in res if r[0] == ph]
        fails = sum(not ok for _, ok, _ in rs)
        lo, hi = wilson(fails, len(rs))
        out[ph] = {"runs": len(rs), "failures": fails, "p_fail": fails / len(rs),
                   "wilson95": [lo, hi],
                   "detection_delays_s": [d for _, ok, d in rs if not ok]}
    return out


def _poisson_job(args) -> bool:
    t, chain, ctx, ev, rate, rep = args
    p = SystemParams(t=t)
    wb = window_bounds(plan_windows(p, chain, ctx), chain)
    horizon = wb["KEY_DERIVATION"][1]
    seed = stable_seed("poisson", t, chain.name, ev.group_size, ev.duration, rate, rep)
    rng = random.Random(seed)
    parts, x = [], rng.expovariate(rate)
    while x < horizon:
        parts.append(nw.Partition(x, x + ev.duration,
                                  frozenset(rng.sample(range(1, p.n + 1), ev.group_size))))
        x += rng.expovariate(rate)
    return _run_with_partitions(t, chain, ctx, parts, seed)[0]


def failure_rates(means: Dict[str, float], exposure: Dict[str, float],
                  cond: Dict[str, dict], rate_per_s: float) -> Tuple[float, ...]:
    """Per-phase CTMC failure rates.  p_fail[j] was measured with the event
    start uniform over phase j's WINDOW (length exposure[j]); the CTMC phase
    lasts means[j] on average (the KEY phase ends at finalization, before its
    window closes).  mu[j] = rate * p_fail[j] * exposure[j] / means[j] keeps
    the expected number of fatal events per phase equal to rate * p_fail[j] *
    exposure[j], i.e. consistent with how p_fail was measured."""
    return tuple(rate_per_s * cond[f]["p_fail"] * exposure[f] / means[f] for f in PHASES)


def phase_exposure(t: int, chain: nw.ChainProfile, ctx: Context) -> Dict[str, float]:
    wb = window_bounds(plan_windows(SystemParams(t=t), chain, ctx), chain)
    return {f: wb[f][1] - wb[f][0] for f in PHASES}


def validate_ctmc(t: int, chain: nw.ChainProfile, ctx: Context, ev: FaultEvent,
                  means: Dict[str, float], cond: Dict[str, dict], rate_per_s: float,
                  reps: int, stages: int = 50, workers: int = 1) -> dict:
    """HELD-OUT check: runs with Poisson fault arrivals (rate per second over
    the whole run) vs the CTMC built from measured durations and the
    separately measured conditional failure probabilities.  The CTMC assumes
    fatal events are independent (no compounding of overlapping events)."""
    exposure = phase_exposure(t, chain, ctx)
    model = PhaseTypeCTMC(tuple(means[f] for f in PHASES),
                          failure_rates(means, exposure, cond, rate_per_s), stages)
    jobs = [(t, chain, ctx, ev, rate_per_s, i) for i in range(reps)]
    oks = _map(_poisson_job, jobs, workers)
    s = sum(oks)
    lo, hi = wilson(s, reps)
    pred = model.success_probability()
    return {"rate_per_s": rate_per_s, "runs": reps, "observed_success": s / reps,
            "observed_wilson95": [lo, hi], "predicted_success": pred,
            "binom_p": float(stats.binomtest(s, reps, pred).pvalue) if 0 < pred < 1 else None,
            "prediction_inside_ci": bool(lo <= pred <= hi), "stages": stages,
            "exposure_s": {f: exposure[f] for f in PHASES}}


def dependability_curves(means: Dict[str, float], exposure: Dict[str, float],
                         cond: Dict[str, dict], rates_per_hour: Sequence[float],
                         stages: int = 50) -> List[dict]:
    """Success probability per attempt and expected time to a usable key vs
    the exogenous fault rate (a sweep, not a claim), for the
    exponential-phase (stages = 1) and window-like (stages) chains."""
    out = []
    completion = sum(means.values())
    attempt = sum(exposure.values())
    for lam_h in rates_per_hour:
        mu = failure_rates(means, exposure, cond, lam_h / 3600.0)
        row = {"fault_rate_per_hour": lam_h}
        for k in (1, stages):
            p = PhaseTypeCTMC(tuple(means[f] for f in PHASES), mu, k).success_probability()
            row[f"success_stages_{k}"] = p
            row[f"time_to_key_s_stages_{k}"] = expected_time_to_key(p, completion, attempt)
        out.append(row)
    return out


# ==========================================================================
# 8. Experiment configurations and the driver.
# ==========================================================================


@dataclass(frozen=True)
class Config:
    name: str
    exec_ts: Tuple[int, ...]
    runs: Dict[int, int]                     # t -> runs per scenario
    scenarios: Tuple[str, ...]
    chains: Tuple[str, ...]                  # for executed runs
    perf_sizes: Tuple[int, ...]
    perf_chains: Tuple[str, ...]
    ctmc_t: int
    ctmc_inject_reps: int
    ctmc_sensitivity_factors: Tuple[float, ...]   # event duration / longest window
    ctmc_sensitivity_reps: int
    ctmc_validate_reps: int
    ctmc_validate_events_per_attempt: float      # Poisson rate x attempt length
    ctmc_sweep_per_hour: Tuple[float, ...]
    ctmc_chains: Tuple[str, ...] = ("bft-1s", "ethereum-l1")
    ctmc_validate_chain: str = "bft-1s"


CHAINS = {c.name: c for c in (nw.BFT_CHAIN, nw.ETHEREUM_L1)}

QUICK = Config(
    name="quick", exec_ts=(2,), runs={2: 3}, scenarios=tuple(SCENARIOS),
    chains=("bft-1s",), perf_sizes=PAPER_SIZES, perf_chains=("bft-1s", "ethereum-l1"),
    ctmc_t=1, ctmc_inject_reps=6, ctmc_sensitivity_factors=(0.5, 2.0),
    ctmc_sensitivity_reps=3, ctmc_validate_reps=24, ctmc_validate_events_per_attempt=0.8,
    ctmc_sweep_per_hour=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0))

# Run counts: 100 failure-free runs bound the failure probability by 3% at 95%
# (runs_for_failure_bound(0.03) == 99).
PAPER = Config(
    name="paper", exec_ts=(2, 5), runs={2: 100, 5: 30}, scenarios=tuple(SCENARIOS),
    chains=("bft-1s",), perf_sizes=PAPER_SIZES, perf_chains=("bft-1s", "ethereum-l1"),
    ctmc_t=2, ctmc_inject_reps=60, ctmc_sensitivity_factors=(0.5, 2.0),
    ctmc_sensitivity_reps=30, ctmc_validate_reps=200, ctmc_validate_events_per_attempt=0.8,
    ctmc_sweep_per_hour=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0))


def dependability_for_chain(cfg: "Config", chain: nw.ChainProfile, ctx: Context,
                            workers: int, validate: bool) -> dict:
    """Sec. V on one chain profile: measured phase durations, conditional
    failure by injection (primary event + duration sensitivity), KS check,
    curves, and -- if `validate` -- the held-out Poisson validation."""
    honest = run_jobs([("honest", cfg.ctmc_t, 0, rep, chain, ctx)
                       for rep in range(max(cfg.ctmc_validate_reps // 4, 8))], workers)
    means = {f: float(np.mean(v)) for f, v in measure_phase_durations(honest).items()}
    event = default_fault_event(cfg.ctmc_t, chain, ctx)
    exposure = phase_exposure(cfg.ctmc_t, chain, ctx)
    cond = conditional_failure(cfg.ctmc_t, chain, ctx, event, cfg.ctmc_inject_reps, workers)
    sensitivity = []
    for fac in cfg.ctmc_sensitivity_factors:
        ev_f = default_fault_event(cfg.ctmc_t, chain, ctx, fac)
        cond_f = conditional_failure(cfg.ctmc_t, chain, ctx, ev_f,
                                     cfg.ctmc_sensitivity_reps, workers)
        sensitivity.append({"duration_factor": fac, "event": asdict(ev_f),
                            "conditional_failure": cond_f,
                            "curves": dependability_curves(means, exposure, cond_f,
                                                           cfg.ctmc_sweep_per_hour)})
    completions = [r.completion_time for r in honest if r.success]
    ks = [ks_validation(PhaseTypeCTMC(tuple(means[f] for f in PHASES),
                                      (0.0,) * len(PHASES), k), completions)
          for k in (1, 50)]
    rate = cfg.ctmc_validate_events_per_attempt / sum(exposure.values())
    val = (validate_ctmc(cfg.ctmc_t, chain, ctx, event, means, cond, rate,
                         cfg.ctmc_validate_reps, workers=workers) if validate else None)
    det = [d for c in cond.values() for d in c["detection_delays_s"]]
    return {
        "t": cfg.ctmc_t, "chain": chain.name, "event": asdict(event),
        "event_rule": "t+1 random nodes cut off (chain included) for the "
                      "longest planned window",
        "phase_means_s": means,
        "paper_state_of_phase": PAPER_STATE_OF_PHASE,
        "conditional_failure": cond,
        "ks_validation": ks,
        "heldout_validation": val,
        "phase_exposure_s": exposure,
        "attempt_length_s": sum(exposure.values()),
        "curves": dependability_curves(means, exposure, cond, cfg.ctmc_sweep_per_hour),
        "sensitivity": sensitivity,
        "detection_delay_s": describe(det),
        "completion_s": describe(completions),
    }


def run_all(cfg: Config, ctx: Context, workers: int = 1, log=print,
            cache_dir: Optional[str] = None) -> dict:
    """Runs every experiment of `cfg`.  With `cache_dir`, finished jobs are
    persisted and an interrupted run resumes where it stopped (the Context
    must be the one stored in that directory: see main())."""
    global _CACHE
    _CACHE = JobCache(cache_dir) if cache_dir else None
    try:
        return _run_all(cfg, ctx, workers, log)
    finally:
        _CACHE = None


def _run_all(cfg: Config, ctx: Context, workers: int, log) -> dict:
    t0 = time.time()
    jobs = [(sc, t, (t if sc in _SIMPLE or sc in ("mixed", "slow_links", "absent") else 0),
             rep, CHAINS[ch], ctx)
            for t in cfg.exec_ts for sc in cfg.scenarios for ch in cfg.chains
            for rep in range(cfg.runs[t])]
    log(f"  executing {len(jobs)} runs ...")
    records = run_jobs(jobs, workers, log)
    log(f"  done in {time.time() - t0:.0f} s")

    perf = [performance_row(n, CHAINS[ch], ctx) for ch in cfg.perf_chains
            for n in cfg.perf_sizes]

    t1 = time.time()
    dep = {ch: dependability_for_chain(cfg, CHAINS[ch], ctx, workers,
                                       validate=(ch == cfg.ctmc_validate_chain))
           for ch in cfg.ctmc_chains}
    log(f"  dependability done in {time.time() - t1:.0f} s")

    return {
        "provenance": provenance(ctx),
        "config": asdict(cfg),
        "scenarios": {k: asdict(v) for k, v in SCENARIOS.items()},
        "executed_summary": summarise(records),
        "attack_overheads": attack_overheads(records),
        "estimator_agreement": estimator_agreement(records, ctx),
        "performance": perf,
        "gas_per_transaction": [gas_per_transaction(bl.matching_t(n))
                                for n in cfg.perf_sizes],
        "table_ii": table_ii(perf),
        "scaling": scaling(perf),
        "dependability": dep,
        "runs": [asdict(r) for r in records],
    }


# ==========================================================================
# 9. Self-tests.
# ==========================================================================


def _selftest() -> None:  # pragma: no cover
    t_start = time.time()
    ctx = make_context()
    print(f"  context: unit costs measured ({ctx.unit.mul * 1e3:.2f} ms/mul), "
          f"cost model fitted at t = {ctx.model.fit_ts}")

    # 1. Statistics.
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35                       # 0/10: [0, 0.278]
    assert abs(zero_failure_upper_bound(100) - 0.0295) < 1e-3   # ~3/n
    assert runs_for_failure_bound(0.03) == 99
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    c = compare_samples("const", [40.0] * 5, [40.0] * 5)
    assert c.welch_p is None and c.diff == 0.0 and c.note
    c = compare_samples("shift", list(np.arange(20.0)), list(np.arange(20.0) + 5))
    assert c.welch_p < 0.05 and c.diff_ci95[1] < 0
    assert fisher_success(10, 10, 0, 10) < 1e-4
    print("  PASS  statistics: Wilson, zero-failure bound, Holm, bootstrap/Welch "
          "guard, Fisher")

    # 2. CTMC algebra: closed form = matrix exponential; Erlang -> deterministic.
    m = PhaseTypeCTMC((4.0, 8.0, 10.0, 8.0, 8.0, 3.0), (1e-2,) * 6, 1)
    assert abs(m.success_probability() - m.reliability(5000.0)) < 1e-6
    m50 = replace(m, stages=50)
    assert abs(m50.success_probability() - m50.reliability(5000.0)) < 1e-6
    det = math.exp(-1e-2 * sum(m.means))
    assert abs(m50.success_probability() - det) < 1e-3 < abs(m.success_probability() - det)
    cdf = m50.passage_cdf([30.0, 41.0, 52.0])
    assert cdf[0] < 0.05 and 0.4 < cdf[1] < 0.6 and cdf[2] > 0.95
    assert expected_time_to_key(1.0, 40.0, 50.0) == 40.0
    assert abs(expected_time_to_key(0.5, 40.0, 50.0) - 90.0) < 1e-12
    assert expected_time_to_key(0.0, 40.0, 50.0) == math.inf
    print("  PASS  phase-type CTMC: closed form = expm; 50 stages -> exp(-mu*T); "
          "expected time to a usable key")

    # 3. Scenario construction respects the fault model.
    p = SystemParams(t=2)
    for name in SCENARIOS:
        su = build_setup(name, p, 2 if name in _SIMPLE or name in ("mixed", "slow_links", "absent") else 0,
                         nw.BFT_CHAIN, ctx, random.Random(1))
        byz = [i for i, a in su.adversaries.items() if a.strategy is not Strategy.HONEST]
        assert len(byz) <= p.t, name
        if SCENARIOS[name].within_assumptions:
            assert len(su.faulty) <= p.t, name
        else:
            assert len(su.faulty) > p.t or name in ("undersized_windows", "slow_links"), name
    print(f"  PASS  {len(SCENARIOS)} scenarios: <= t faulty inside the assumptions, "
          "> t (or undersized windows) outside")

    # 4. Quick configuration end to end.
    res = run_all(QUICK, ctx, workers=1, log=lambda s: print("   ", s.strip()))
    summ = {(s["scenario"], s["t"]): s for s in res["executed_summary"]}
    for (sc, t), s in summ.items():
        if s["within_assumptions"]:
            assert s["successes"] == s["runs"], (sc, s["failure_reasons"])
            assert all(v == s["runs"] for v in s["invariants_held"].values()), \
                (sc, s["invariants_held"])
            assert s["honest_excluded_runs"] == 0 and s["commission_in_qual_runs"] == 0, sc
    print("  PASS  every within-assumption scenario: 100% success and every "
          "accountability invariant on every run")
    beyond = {sc: summ[(sc, 2)]["success_rate"] for sc in
              ("long_partition", "crash_beyond_t", "undersized_windows")}
    assert beyond["crash_beyond_t"] == 0.0 and beyond["undersized_windows"] < 1.0, beyond
    print(f"  PASS  outside the assumptions success drops: {beyond}")
    cured = summ[("omission_cured", 2)]["recovery_latency_s"]
    assert cured["count"] > 0
    print(f"  PASS  omission recovered through the ledger in "
          f"{cured['mean']:.1f} s (complaint -> answer inclusion)")

    for a in res["estimator_agreement"]:
        assert a["sent_mean_sim"] == a["sent_est"] and a["gas_sim"] == a["gas_est"]
        assert a["ledger_sim"] == a["ledger_est"]
        assert abs(a["completion_sim"] - a["completion_est"]) <= 2.0, a
    print("  PASS  estimator = executed runs (bytes, gas, ledger exact; completion "
          "within 2 blocks)")

    sc_ = res["scaling"]["bft-1s"]
    assert abs(sc_["sent_bytes_per_node"]["exponent"] - 2) < 0.1
    assert abs(sc_["p2p_total_bytes"]["exponent"] - 3) < 0.1
    assert abs(sc_["gas_total"]["exponent"] - 1) < 0.1
    print(f"  PASS  scaling (n >= 64): per-node bytes n^"
          f"{sc_['sent_bytes_per_node']['exponent']:.2f}, total n^"
          f"{sc_['p2p_total_bytes']['exponent']:.2f}, gas n^"
          f"{sc_['gas_total']['exponent']:.2f}, completion n^"
          f"{sc_['completion_s']['exponent']:.2f}")

    tii = res["table_ii"]
    assert tii and all(r["n_differs"] == (bl.matching_t(r["n"]) is None) for r in tii)
    print(f"  PASS  Table II pairs: {len(tii)} (anchor, DA-ADKG) rows, "
          f"{sum(r['n_differs'] for r in tii)} flagged n-differs")

    dep = res["dependability"]["bft-1s"]
    v = dep["heldout_validation"]
    print(f"  info  CTMC held-out: predicted {v['predicted_success']:.3f}, observed "
          f"{v['observed_success']:.3f} {tuple(round(x, 3) for x in v['observed_wilson95'])}"
          f"; KS D (1 stage) {dep['ks_validation'][0]['ks_D']:.2f}, "
          f"(50 stages) {dep['ks_validation'][1]['ks_D']:.2f}")
    print("        conditional failure per phase: "
          + ", ".join(f"{k[:4]} {c['failures']}/{c['runs']}"
                      for k, c in dep["conditional_failure"].items()))
    eth = res["dependability"]["ethereum-l1"]
    print(f"        ethereum-l1: attempt {eth['attempt_length_s'] / 3600:.2f} h, event "
          f"{eth['event']['duration']:.0f} s; conditional failures "
          + ", ".join(f"{k[:4]} {c['failures']}/{c['runs']}"
                      for k, c in eth["conditional_failure"].items()))

    # 5. Determinism and JSON round trip.
    r1 = execute("commission_all", 2, 2, 0, nw.BFT_CHAIN, ctx)
    r2 = execute("commission_all", 2, 2, 0, nw.BFT_CHAIN, ctx)
    assert asdict(r1) == asdict(r2)
    json.loads(dumps(res))
    assert res["provenance"]["unit_costs"]["source"].startswith("measured")
    print("  PASS  determinism (same seed -> identical record); results are "
          "JSON-serialisable with provenance")

    # 6. Resumable runs: an interrupted cache resumes to identical results.
    import tempfile
    global _CACHE
    with tempfile.TemporaryDirectory() as td:
        save_context(ctx, td)
        ctx2 = load_context(td)
        assert ctx2 == ctx                                   # exact round trip
        jobs = [("omission", 1, 1, rep, nw.BFT_CHAIN, ctx2) for rep in range(4)]
        fresh = [asdict(r) for r in run_jobs(jobs)]
        _CACHE = JobCache(td)
        try:
            run_jobs(jobs[:2])                               # "interrupted" after 2
            with open(_CACHE.path, "a") as fh:
                fh.write('["torn')                           # torn last line
            _CACHE = JobCache(td)
            assert len(_CACHE.done) == 2
            calls = []
            orig = globals()["execute"]
            globals()["execute"] = lambda *a, **k: (calls.append(a), orig(*a, **k))[1]
            resumed = [asdict(r) for r in run_jobs(jobs)]
            globals()["execute"] = orig
        finally:
            _CACHE = None
        assert resumed == fresh and len(calls) == 2
        assert len(JobCache(td).done) == 4                  # nothing lost after the tear
    print("  PASS  resumable runs: context round-trips exactly; an interrupted cache "
          "(with a torn line) re-runs only the missing jobs, identical results")

    print("\n  Sample (quick, bft-1s, honest estimator):")
    print(f"  {'n':>5} {'compl(s)':>9} {'sent/node MB':>13} {'ledger KB':>10} {'gas M':>7}")
    for r in res["performance"]:
        if r["chain"] == "bft-1s":
            print(f"  {r['n']:>5} {r['completion_s']:>9.1f} {r['sent_MB_per_node']:>13.3f} "
                  f"{r['ledger_write_bytes'] / 1e3:>10.1f} {r['gas_total'] / 1e6:>7.2f}")
    print(f"\nexperiments.py: all self-tests passed ({time.time() - t_start:.0f} s).")


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, default=_jsonable)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", choices=("quick", "paper"))
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cache", default=None,
                    help="directory for resumable runs (recommended for --run paper)")
    a = ap.parse_args(argv)
    if a.run is None:
        _selftest()
        return
    # Unit costs are measured once, before any worker starts, on an otherwise
    # idle machine; more repetitions for the paper run (medians are reported).
    # A resumed run re-uses the Context stored in its cache directory.
    ctx = load_context(a.cache) if a.cache else None
    if ctx is None:
        ctx = make_context(calib_reps=200 if a.run == "paper" else 60)
        if a.cache:
            os.makedirs(a.cache, exist_ok=True)
            save_context(ctx, a.cache)
    else:
        print(f"resuming from {a.cache} (stored unit costs and cost model)")
    res = run_all({"quick": QUICK, "paper": PAPER}[a.run], ctx, workers=a.workers,
                  cache_dir=a.cache)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(dumps(res))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
