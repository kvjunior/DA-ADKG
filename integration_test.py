"""
integration_test.py -- Cross-file checks for the DA-ADKG v2 artefact.

Each module has its own self-test (`python3 <module>.py`).  This suite checks
the SEAMS between modules, regression-tests every defect found in the v1
code (D7 R1-R7), and runs the adversarial checks that used to live in
xcheck_protocol.py and xcheck_network.py (now folded in here).

    python3 integration_test.py                       # all checks (~5 min)
    python3 integration_test.py --results R.json      # also: is R.json fresh?
    python3 integration_test.py --results R.json --results-only
    python3 integration_test.py --texts DIR --ledger D4.md   # also: anchors

SEAMS
  I1  protocol.run_dkg (synchronous) and network.run_dkg_async agree on
      success, QUAL, honest holders, complaints and dispute verdicts.
  I2  Gas: metered = estimator = transaction counts x protocol.gas_* schedule.
  I3  Bytes: network's size formulas = the real serialised messages of
      protocol.py; bytes sent per node, P2P total and ledger writes exact.
  I4  experiments.execute = a direct run_dkg_async with the same setup.
  I5  Cost model held out at t = 4; planned windows reject no honest
      transaction; estimator = simulator; adversarial async run accounted.
  I6  Determinism across processes (different PYTHONHASHSEED).
  I7  Pipeline: experiments -> results.json -> figures -> LaTeX compiles;
      a results file whose source hashes do not match is detected.
  I8  Gas-accounting report (methods text; F9).
REGRESSIONS (v1 defects, D7 Sec. 2)
  R1  a replayed dispute never transfers stake twice
  R2  a selective dealer is excluded and >= 2t+1 honest holders complete
  R3  F1: the printed vote rule excludes an honest dealer whose row is late;
      the complaint rule does not
  R4  an omission terminates with exclusion (v1 hung)
  R5  bytes are counted per recipient (Theta(n) messages per dealer)
  R6  protocol group is BN254 G1 (EVM precompiles), not secp256k1
  R7  P-1: a party that never registers does not stop the others; fewer
      than n - t registrations void the session
  P5  the finalised key is g^s for the audited secret s
ADVERSARIAL (from xcheck_protocol)
  A1  non-frameability: 10 single-field mutations of evidence against an
      honest dealer are never upheld
  A2  forged answers rejected; the genuine one accepted once
  A3  t = 3 with three Byzantine strategies at once
  A4  t wrong key votes never finalise; t+1 matching votes do
  A5  a dispute after the DISPUTE window is rejected
  A6  decoder fuzz: 400 malformed payloads never crash the receiver
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import random
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from typing import Callable, List, Optional, Tuple

import crypto
import experiments as ex
import figures as fg
import network as nw
from crypto import DLEQProof, G, H, Q, ecdsa_sign, point_add, point_mul, ser_point_evm
from protocol import (
    Answer,
    KeyShare,
    Participant,
    QualRule,
    SmartContract,
    Strategy,
    SystemParams,
    Window,
    calldata_bytes,
    gas_answer,
    gas_anchor,
    gas_complaint,
    gas_dispute,
    gas_register,
    gas_submit_key,
    run_dkg,
)

S = Strategy
BN254_P = 21888242871839275222246405745257275088696311157297823662689037894645226208583
BN254_R = 21888242871839275222246405745257275088548364400416034343698204186575808495617
CTX: Optional[ex.Context] = None
RESULTS: List[Tuple[str, str, float]] = []


def ctx() -> ex.Context:
    global CTX
    if CTX is None:
        CTX = ex.make_context()
    return CTX


def run_async(p: SystemParams, adversaries=None, seed: int = 0, **kw) -> nw.AsyncResult:
    c = ctx()
    return nw.run_dkg_async(p, adversaries=adversaries, unit_costs=c.unit,
                            cost_model=c.model, net=c.net, seed=seed, **kw)


def check(name: str):
    def deco(fn: Callable[[], str]):
        fn.check_name = name
        return fn
    return deco


# ==========================================================================
# Seams.
# ==========================================================================

I1_CASES = (
    ("honest", {}),
    ("corrupt-all", {7: (S.CORRUPT_ALL, ())}),
    ("selective", {6: (S.SELECTIVE, (1, 2))}),
    ("silent + selective", {5: (S.SILENT, ()), 6: (S.SELECTIVE, (1,))}),
    ("withhold", {7: (S.WITHHOLD, (1, 2))}),
    ("withhold then answer", {7: (S.WITHHOLD_THEN_ANSWER, (1,))}),
    ("false accuser", {7: (S.FALSE_ACCUSER, (2,))}),
    ("bad PoK", {7: (S.BAD_POK, ())}),
)


@check("I1 sync and async drivers agree")
def i1() -> str:
    p = SystemParams(t=2)
    for name, byz in I1_CASES:
        a = run_dkg(p, byzantine=byz)
        b = run_async(p, {i: nw.AdversaryProfile(s, t) for i, (s, t) in byz.items()}, seed=11)
        assert a.success == b.success, name
        assert sorted(a.qual) == sorted(b.qual), (name, a.qual, b.qual)
        assert sorted(a.honest_complete_holders) == sorted(b.honest_complete_holders), name
        assert set(a.complaints) == set(b.complaints), name
        va = Counter((d["kind"], d["verdict"]) for d in a.disputes)
        vb = Counter((d["kind"], d["verdict"]) for d in b.disputes)
        assert va == vb, (name, va, vb)
    return f"{len(I1_CASES)} adversary configurations: success, QUAL, holders, complaints, verdicts"


@check("I2 gas: metered = estimator = schedule")
def i2() -> str:
    for t in (1, 2, 3):
        p = SystemParams(t=t)
        r = run_async(p, seed=5)
        assert r.success
        e = nw.estimate_performance(p, unit_costs=ctx().unit, cost_model=ctx().model)
        acc = r.tx_accepted
        assert acc == {"register": p.n, "anchor": p.n, "submit_public_key": p.n}, acc
        schedule = (p.n * (gas_register() + gas_anchor())
                    + (p.n - 1) * gas_submit_key(False) + gas_submit_key(True))
        assert r.gas_total == e.gas_total == schedule == sum(r.gas_by_category.values()), t
    # accountability transactions priced exactly by the schedule too
    p = SystemParams(t=2)
    r = run_async(p, {7: nw.AdversaryProfile(S.WITHHOLD_THEN_ANSWER, (1, 2))}, seed=3)
    assert r.success
    assert r.gas_by_category["complaint"] == 2 * gas_complaint()
    assert r.gas_by_category["answer"] == 2 * gas_answer(p)
    r = run_async(p, {7: nw.AdversaryProfile(S.SELECTIVE, (1,))}, seed=3)
    upheld = [d for d in r.disputes if d["verdict"] == "upheld"]
    assert len(upheld) == 1 and r.gas_by_category["dispute"] == gas_dispute(p)
    return "honest t=1..3 exact; complaint, answer and upheld dispute priced by the schedule"


@check("I3 bytes: formulas = serialised messages")
def i3() -> str:
    for t in (1, 2, 3):
        p = SystemParams(t=t)
        parts = {i: Participant(i, p) for i in range(1, p.n + 1)}
        _, _, msgs = parts[1].deal({i: q.pk for i, q in parts.items()})
        pl = parts[1]._make_payload(2, parts[1]._f.eval(2), parts[1]._fh.eval(2))
        assert len(pl.to_bytes()) == nw.share_payload_bytes(p), t
        r = run_async(p, seed=7)
        assert r.success
        q = Participant(2, p)
        q.final_share = (12345, 67890)            # any share: the size is fixed
        ks = q.key_share()
        assert KeyShare.from_bytes(ks.to_bytes()) == ks
        assert len(ks.to_bytes()) == nw.key_share_bytes()
        tx = (calldata_bytes("register", p) + calldata_bytes("anchor", p)
              + calldata_bytes("submit_public_key", p))
        per_node = (p.n - 1) * (nw.share_payload_bytes(p) + nw.key_share_bytes()) + tx
        assert set(r.comm.sent.values()) == {per_node}, (t, set(r.comm.sent.values()))
        assert r.comm.p2p_total == p.n * (p.n - 1) * (nw.share_payload_bytes(p)
                                                      + nw.key_share_bytes())
        assert r.comm.ledger_writes == p.n * tx
    return "share and key-share wire sizes; sent per node, P2P total, ledger writes exact"


@check("I4 experiments.execute = direct run")
def i4() -> str:
    for scenario, k in (("commission_all", 2), ("omission_cured", 2), ("long_partition", 0)):
        rec = ex.execute(scenario, 2, k, 0, nw.BFT_CHAIN, ctx())
        p = SystemParams(t=2)
        seed = ex.stable_seed(scenario, 2, k, nw.BFT_CHAIN.name, 0)
        su = ex.build_setup(scenario, p, k, nw.BFT_CHAIN, ctx(), random.Random(seed))
        r = nw.run_dkg_async(p, adversaries=su.adversaries, partitions=su.partitions,
                             chain=nw.BFT_CHAIN, net=ctx().net, unit_costs=ctx().unit,
                             windows=su.windows, cost_model=ctx().model, seed=seed,
                             audit_reconstruct=True)
        assert (rec.success, rec.qual, rec.completion_time, rec.gas_total,
                rec.p2p_total_bytes, rec.ledger_write_bytes) == (
            r.success, r.qual, r.completion_time, r.gas_total, r.comm.p2p_total,
            r.comm.ledger_writes), scenario
    return "3 scenarios: identical outcome, completion, gas and bytes"


@check("I5 cost model held out; windows; estimator; adversarial async")
def i5() -> str:
    c = ctx()
    p4 = SystemParams(t=4)
    r4 = nw.run_dkg_async(p4, unit_costs=c.unit, seed=21,
                          windows=nw.WindowSchedule(5, 40, 60, 40, 40, 60))
    assert r4.success
    meas = nw._per_node_action_counts(r4, p4.n)
    worst = 0.0
    for a in nw._ACTIONS:
        pred, got = c.model.counts(a, 4), Counter(meas[a])
        if a == "key_verify":       # small/full split depends on arrival order
            for cc in (pred, got):
                cc["mul_all"] = cc.pop("mul", 0) + cc.pop("mul_small", 0)
        for k in set(pred) | set(got):
            m, q = got.get(k, 0), pred.get(k, 0)
            if m or q:
                worst = max(worst, abs(m - q) / max(m, q))
    assert worst < 1e-9, worst
    rejected = 0
    for t, seeds in ((2, range(6)), (3, range(2))):
        for sd in seeds:
            r = run_async(SystemParams(t=t), seed=100 + sd)
            assert r.success, (t, sd, r.reason)
            rejected += sum(r.tx_rejected.values())
    assert rejected == 0
    for t in (3, 4):
        p = SystemParams(t=t)
        r = run_async(p, seed=5)                     # planned windows
        e = nw.estimate_performance(p, unit_costs=c.unit, cost_model=c.model)
        assert abs(r.completion_time - e.completion_time) <= 2 * nw.BFT_CHAIN.block_time
        assert set(r.comm.sent.values()) == {e.per_node_sent_bytes}
        assert r.gas_total == e.gas_total
    r = run_async(SystemParams(t=3), {
        8: nw.AdversaryProfile(S.CORRUPT_ALL), 9: nw.AdversaryProfile(S.WITHHOLD, (1, 2)),
        10: nw.AdversaryProfile(S.HONEST, crash_time=20.0)},
        partitions=[nw.Partition(5.0, 9.0, frozenset({3, 4}))], seed=9,
        audit_reconstruct=True)
    assert r.success and 8 not in r.qual and 9 not in r.qual
    assert r.complaints == {(1, 9): False, (2, 9): False} and not r.tx_rejected
    assert r.audit and all(r.audit.values())
    return ("op counts exact at held-out t=4; 8 runs with planned windows reject "
            "nothing; estimator = simulator at n=10, 13; t=3 corrupt+withhold+crash+"
            "partition accounted")


_CHILD = r"""
import json, pickle, sys
sys.path.insert(0, {here!r})
import experiments as ex, network as nw
from dataclasses import asdict
ctx = pickle.load(open({ctxp!r}, "rb"))
rec = ex.execute("mixed", 2, 2, 3, nw.BFT_CHAIN, ctx)
print(json.dumps(asdict(rec), sort_keys=True))
"""


@check("I6 determinism across processes")
def i6() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as td:
        ctxp = os.path.join(td, "ctx.pkl")
        with open(ctxp, "wb") as fh:
            pickle.dump(ctx(), fh)
        outs = []
        for hs in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=hs)
            r = subprocess.run([sys.executable, "-c", _CHILD.format(here=here, ctxp=ctxp)],
                               capture_output=True, text=True, env=env, check=True)
            outs.append(r.stdout)
    assert outs[0] == outs[1] and outs[0].strip()
    return "the same run in two processes (PYTHONHASHSEED 1, 2) gives byte-identical records"


MINI = ex.Config(
    name="integration-mini", exec_ts=(1,), runs={1: 2},
    scenarios=("honest", "commission_all", "omission_cured", "crash_beyond_t"),
    chains=("bft-1s",), perf_sizes=ex.PAPER_SIZES, perf_chains=("bft-1s", "ethereum-l1"),
    ctmc_t=1, ctmc_inject_reps=2, ctmc_sensitivity_factors=(2.0,),
    ctmc_sensitivity_reps=1, ctmc_validate_reps=8, ctmc_validate_events_per_attempt=0.8,
    ctmc_sweep_per_hour=(0.1, 10.0, 1000.0), ctmc_chains=("bft-1s",))


def check_results_fresh(path: str) -> List[str]:
    """Problems that make a results file unusable for the paper: schema,
    missing unit-cost source, or source code changed since it was produced."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    prov = d.get("provenance", {})
    probs = []
    if prov.get("schema") != ex.SCHEMA:
        probs.append(f"schema {prov.get('schema')!r}")
    if not prov.get("unit_costs", {}).get("source"):
        probs.append("unit-cost source missing")
    here = os.path.dirname(os.path.abspath(ex.__file__))
    for f, h in prov.get("source_sha256", {}).items():
        with open(os.path.join(here, f), "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != h:
                probs.append(f"{f} changed since the run")
    return probs


@check("I7 pipeline experiments -> results -> figures")
def i7() -> str:
    with tempfile.TemporaryDirectory() as td:
        res = ex.run_all(MINI, ctx(), workers=1, log=lambda *_: None)
        rp = os.path.join(td, "r.json")
        with open(rp, "w", encoding="utf-8") as fh:
            fh.write(ex.dumps(res))
        assert check_results_fresh(rp) == []
        out = fg.generate_all(rp, os.path.join(td, "out"))
        assert all(os.path.getsize(v) > 300 for v in out.values())
        compiled = "pdflatex not installed"
        if shutil.which("pdflatex"):
            ok, log = fg.compile_tables([out[n] for n in fg.TABLES], os.path.join(td, "out"))
            assert ok, log
            compiled = "tables compile"
        d = json.load(open(rp))
        d["provenance"]["source_sha256"]["protocol.py"] = "0" * 64
        with open(rp, "w") as fh:
            json.dump(d, fh)
        assert check_results_fresh(rp) == ["protocol.py changed since the run"]
    return f"{len(out)} outputs; {compiled}; a stale results file is detected"


# Submitted manuscript, Table III, n = 128 (finding F9).  Used ONLY to state the
# discrepancy in the report below; nothing is calibrated to it.
MANUSCRIPT_TOTAL_GAS_N128 = 623_000


@check("I8 gas-accounting report (F9)")
def i8() -> str:
    p = SystemParams(t=42)
    e = nw.estimate_performance(p, unit_costs=ctx().unit, cost_model=ctx().model)
    share = MANUSCRIPT_TOTAL_GAS_N128 / e.gas_total
    return (f"n=127 honest run, every participant submits its own transactions: "
            f"{e.gas_total / 1e6:.2f} M gas in total, {e.gas_total / p.n / 1e3:.0f} K per "
            f"participant (register {gas_register():,}, anchor {gas_anchor():,}, key vote "
            f"{gas_submit_key(False):,}). The manuscript's 623 K (n=128) is {share:.1%} "
            f"of this; the revision must state its gas accounting")


# ==========================================================================
# Regressions of the v1 defects.
# ==========================================================================


def _setup(strategy_for_6: Strategy, p: Optional[SystemParams] = None):
    """Dealer 6 deals to node 1 under `strategy_for_6`; contract at COMPLAINT."""
    p = p or SystemParams(t=2)
    parts = {i: Participant(i, p, *((strategy_for_6, (1,)) if i == 6 else (S.HONEST, ())))
             for i in range(1, p.n + 1)}
    c = SmartContract(p)
    for q in parts.values():
        c.register(*q.registration_tx())
    c.advance_to(c.bounds[Window.DEALING][0])
    mr, ph, msgs = parts[6].deal({i: parts[i].pk for i in parts})
    c.anchor(6, mr, ph)
    parts[1].receive_share(6, msgs[1], c)
    c.advance_to(c.bounds[Window.COMPLAINT][0])
    return parts, c


@check("R1 replayed dispute transfers once")
def r1() -> str:
    parts, c = _setup(S.SELECTIVE)
    pl = parts[1].provable_bad[6]
    ev = parts[1]._evidence(6, pl.commits, pl.c, pl.c_hat, pl.sig, c)
    verdicts = [c.dispute(ev) for _ in range(10)]
    assert verdicts == ["upheld"] + ["duplicate"] * 9, verdicts
    assert c.registry[6].stake == 0.0 and len(c.ledger) == 1
    return "10 identical disputes: 1 upheld + 9 duplicate, one ledger transfer"


@check("R2 selective dealer excluded, holders complete")
def r2() -> str:
    p = SystemParams(t=2)
    r = run_async(p, {6: nw.AdversaryProfile(S.SELECTIVE, (1, 2))}, seed=4)
    assert r.success and 6 not in r.qual
    assert len(r.honest_complete_holders) >= p.recon_threshold
    return f"QUAL {r.qual}; {len(r.honest_complete_holders)} honest holders >= 2t+1"


@check("R3 F1: vote rule vs complaint rule")
def r3() -> str:
    legacy = run_dkg(SystemParams(t=2, qual_rule=QualRule.VOTES_LEGACY),
                     byzantine={7: (S.VOTE_AGAINST, (5,))}, late=frozenset({1, 2}))
    fixed = run_dkg(SystemParams(t=2), late=frozenset({1, 2}))
    assert 5 not in legacy.qual, legacy.qual
    assert fixed.success and all(i in fixed.qual for i in range(1, 8)), fixed.qual
    return (f"printed vote rule drops honest dealer 5 (QUAL {legacy.qual}); "
            f"complaint rule keeps every honest dealer (QUAL {fixed.qual})")


@check("R4 omission terminates with exclusion")
def r4() -> str:
    r = run_async(SystemParams(t=2), {7: nw.AdversaryProfile(S.WITHHOLD, (1,))}, seed=2)
    assert r.success and 7 not in r.qual and r.complaints == {(1, 7): False}
    assert not any(d["dealer"] == 7 and d["slashed"] for d in r.disputes)
    return "withholding dealer excluded, not slashed (F14); run terminated"


@check("R5 bytes counted per recipient")
def r5() -> str:
    rows = [nw.estimate_performance(SystemParams(t=t), unit_costs=ctx().unit,
                                    cost_model=ctx().model) for t in (21, 42)]
    ratio = rows[1].per_node_sent_bytes / rows[0].per_node_sent_bytes
    assert 3.5 < ratio < 4.5, ratio                # Theta(n^2) per node
    assert rows[1].p2p_total_bytes == 127 * 126 * (
        nw.share_payload_bytes(SystemParams(t=42)) + nw.key_share_bytes())
    return f"n 64 -> 127: bytes per node x{ratio:.2f} (Theta(n^2)); total = n(n-1) messages"


@check("R7 participant set = registered set (P-1)")
def r7() -> str:
    p = SystemParams(t=2)
    gone = {6: nw.AdversaryProfile(S.HONEST, crash_time=0.0),
            7: nw.AdversaryProfile(S.HONEST, crash_time=0.0)}
    r = run_async(p, gone, seed=12, audit_reconstruct=True)
    assert r.success and set(r.qual) == {1, 2, 3, 4, 5}, (r.reason, r.qual)
    assert r.audit and all(r.audit.values())
    gone[5] = nw.AdversaryProfile(S.HONEST, crash_time=0.0)    # t+1 absent
    r = run_async(p, gone, seed=12)
    assert not r.success and r.reason == "registration-below-n-t", r.reason
    assert not r.qual and r.gas_by_category.get("commitment", 0) == 0
    return ("t parties never register: run succeeds with the registered set; "
            "t+1 absent: session void (registration below n - t), nothing anchored")


@check("R6 protocol group is BN254 G1")
def r6() -> str:
    assert crypto.P == BN254_P and crypto.Q == BN254_R and crypto.G == (1, 2)
    x, y = crypto.G
    assert (y * y - x ** 3 - 3) % crypto.P == 0
    assert len(ser_point_evm(point_mul(7, G))) == 64
    return "p, r, generator and 64-byte EVM encoding (EIP-196) match BN254"


@check("P5 finalised key is g^s")
def p5() -> str:
    r = run_async(SystemParams(t=2), {7: nw.AdversaryProfile(S.CORRUPT_ALL)}, seed=8,
                  audit_reconstruct=True)
    assert r.success and r.audit == {"secret_matches_dealer_sum": True,
                                     "public_key_is_g_to_s": True}
    return "reconstructed s = sum of QUAL secrets and the on-chain key equals g^s"


# ==========================================================================
# Adversarial protocol checks (formerly xcheck_protocol.py).
# ==========================================================================


@check("A1 non-frameability under evidence mutation")
def a1() -> str:
    def evidence():
        parts, c = _setup(S.HONEST)
        pl = parts[1].valid_payload[6]
        return parts, c, parts[1]._evidence(6, pl.commits, pl.c, pl.c_hat, pl.sig, c)

    _, _, ev = evidence()
    muts = {
        "commit[0]": dict(commits=(point_add(ev.commits[0], G),) + ev.commits[1:]),
        "commit[-1]": dict(commits=ev.commits[:-1] + (point_add(ev.commits[-1], H),)),
        "c": dict(c=(ev.c + 1) % Q), "c_hat": dict(c_hat=(ev.c_hat + 1) % Q),
        "sig byte": dict(sig=ev.sig[:10] + bytes([ev.sig[10] ^ 1]) + ev.sig[11:]),
        "K": dict(K=point_add(ev.K, G)),
        "dleq.c": dict(dleq=DLEQProof((ev.dleq.c + 1) % Q, ev.dleq.z)),
        "dleq.z": dict(dleq=DLEQProof(ev.dleq.c, (ev.dleq.z + 1) % Q)),
        "accuser": dict(accuser=2), "dealer": dict(dealer=5),
    }
    verdicts = {}
    for name, m in muts.items():
        _, c, e0 = evidence()
        verdicts[name] = c.dispute(dataclasses.replace(e0, **m))
        assert c.registry[6].stake == 1000.0, name
    assert set(verdicts.values()) <= {"dismissed", "rejected"}, verdicts
    _, c, e0 = evidence()
    assert c.dispute(e0) == "false-accusation" and c.registry[6].stake == 1000.0
    return f"{len(muts)} mutations -> {sorted(set(verdicts.values()))}; honest stake intact"


@check("A2 forged answers rejected")
def a2() -> str:
    parts, c = _setup(S.WITHHOLD)
    assert parts[1].receive_share(6, None, c) == "unusable"
    assert c.complain(1, 6)
    c.advance_to(c.bounds[Window.ANSWER][0])
    d = parts[6]
    good = d._make_payload(1, d._f.eval(1), d._fh.eval(1))

    def mk(**kw):
        base = dict(dealer=6, recipient=1, commits=d.commits, poly_pok_bytes=d.poly_pok_bytes,
                    c=good.c, c_hat=good.c_hat, sig=good.sig)
        return Answer(**{**base, **kw})

    assert not c.answer(mk(sig=ecdsa_sign(parts[2].ecdsa_sk, b"forged")))
    assert not c.answer(mk(commits=d.commits[::-1]))
    assert not c.answer(mk(poly_pok_bytes=d.poly_pok_bytes + b"\x00"))
    assert not c.answer(mk(recipient=2))
    assert c.answer(mk()) and not c.answer(mk())
    return "wrong signer, reordered commitments, PoK mismatch, no complaint, duplicate"


@check("A3 three Byzantine strategies at t = 3")
def a3() -> str:
    r = run_dkg(SystemParams(t=3), byzantine={8: (S.CORRUPT_ALL, ()),
                                              9: (S.WITHHOLD, (1, 2, 3)),
                                              10: (S.FALSE_ACCUSER, (4,))},
                audit_reconstruct=True)
    assert r.success and r.qual == [1, 2, 3, 4, 5, 6, 7, 10], r.qual
    assert sorted(r.honest_complete_holders) == list(range(1, 8))
    assert r.audit and all(r.audit.values())
    assert abs(sum(e["amount"] - e["reward"] for e in r.ledger) - r.treasury) < 1e-9
    return f"QUAL {r.qual}; 7 honest holders; audit and conservation hold"


@check("A4 key finalisation needs t+1 matching votes")
def a4() -> str:
    p = SystemParams(t=2)
    c = SmartContract(p)
    for q in (Participant(i, p) for i in range(1, 8)):
        c.register(*q.registration_tx())
    c.advance_to(c.bounds[Window.DISPUTE][1])
    c.close_qualification()
    c.advance_to(c.bounds[Window.KEY_DERIVATION][0])
    wrong, right = ser_point_evm(point_mul(12345, G)), ser_point_evm(point_mul(777, G))
    assert c.submit_public_key(6, wrong) and c.submit_public_key(7, wrong)
    assert c.public_key is None
    assert c.submit_public_key(1, right) and c.submit_public_key(2, right)
    assert c.public_key is None and c.submit_public_key(3, right)
    assert c.public_key == right and not c.submit_public_key(6, right)
    return "t wrong votes never finalise; t+1 do; double vote rejected"


@check("A5 late dispute rejected")
def a5() -> str:
    parts, c = _setup(S.SELECTIVE)
    pl = parts[1].provable_bad[6]
    ev = parts[1]._evidence(6, pl.commits, pl.c, pl.c_hat, pl.sig, c)
    c.advance_to(c.bounds[Window.DISPUTE][1])
    assert c.dispute(ev) == "rejected" and c.registry[6].stake == 1000.0
    return "after the DISPUTE window: rejected, no slashing"


@check("A6 decoder fuzz")
def a6() -> str:
    parts, c = _setup(S.HONEST)
    d = parts[6]
    good = d._make_payload(1, d._f.eval(1), d._fh.eval(1)).to_bytes()
    rng = random.Random(6)
    seen = Counter()
    for k in range(400):
        b = bytearray(good)
        if k % 3 == 0:
            b = b[: rng.randrange(len(b))]
        elif k % 3 == 1:
            b[rng.randrange(len(b))] ^= 1 << rng.randrange(8)
        else:
            b = bytearray(secrets.token_bytes(rng.randrange(1, 900)))
        seen[parts[1].receive_share(6, bytes(b), c)] += 1
    assert set(seen) <= {"unusable", "provable-invalid", "valid", "pok-fraud"}
    return f"400 malformed payloads -> {dict(seen)}; no exception"


@check("B1 published anchors verified against the PDFs")
def b1() -> str:
    import baselines as bl
    if not (ARGS.texts and os.path.isdir(ARGS.texts)):
        return "SKIP (no --texts directory)"
    fails = bl.verify_against_texts(ARGS.texts)
    if ARGS.ledger:
        fails += bl.check_evidence_ids(ARGS.ledger)
    assert not fails, fails
    return f"{len(bl.ANCHORS)} anchors, testbed quotes and table grids verified"


CHECKS = (i1, i2, i3, i4, i5, i6, i7, i8, r1, r2, r3, r4, r5, r6, r7, p5,
          a1, a2, a3, a4, a5, a6, b1)
ARGS = argparse.Namespace(texts=None, ledger=None, results=None, only=None,
                          results_only=False)


def main(argv=None) -> int:
    global ARGS
    ap = argparse.ArgumentParser(description="DA-ADKG v2 cross-file checks")
    ap.add_argument("--results", help="results.json to check for freshness")
    ap.add_argument("--texts", help="directory of extracted PDF texts (baselines)")
    ap.add_argument("--ledger", help="D4 evidence ledger (baselines)")
    ap.add_argument("--only", nargs="*", help="run only these check ids (e.g. I1 R3)")
    ap.add_argument("--results-only", action="store_true",
                    help="only check the --results file, run no other check")
    ARGS = ap.parse_args(argv)
    print("DA-ADKG v2 integration suite\n")
    failed = 0
    for fn in CHECKS:
        cid = fn.check_name.split()[0]
        if ARGS.results_only or (ARGS.only and cid not in ARGS.only):
            continue
        t0 = time.time()
        try:
            msg = fn()
            status = "SKIP" if msg.startswith("SKIP") else ("INFO" if cid == "I8" else "PASS")
        except Exception as e:                       # report and continue
            msg, status = f"{type(e).__name__}: {e}"[:300], "FAIL"
            failed += 1
        dt = time.time() - t0
        print(f"  {status}  {fn.check_name:<52} {dt:6.1f} s\n        {msg}")
    if ARGS.results:
        probs = check_results_fresh(ARGS.results)
        print(f"\n  results file {ARGS.results}: "
              + ("FRESH (schema, unit costs and source hashes match)" if not probs
                 else "NOT USABLE: " + "; ".join(probs)))
        failed += bool(probs)
    print("\nintegration_test.py: " + ("all checks passed." if not failed
                                        else f"{failed} check(s) FAILED."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
