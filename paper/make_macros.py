"""
make_macros.py -- every number the manuscript text quotes from the experiments
is generated here from the results file, so the prose can never drift from
the measured data.  Re-run after any new experiment run:

    python3 make_macros.py RESULTS.json > results_macros.tex

Published numbers (other papers) come from baselines.ANCHORS, never typed.
"""
import json
import sys

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import baselines as bl  # noqa: E402

R = json.load(open(sys.argv[1]))
out = []


def m(name, value):
    out.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def num(x, nd=1):
    return f"{x:,.{nd}f}".replace(",", "{,}")


prov = R["provenance"]
m("RunCPU", prov["cpu_model"].replace("(R)", "").replace("@", "at"))
m("UnitMulMs", num(prov["unit_costs"]["mul"] * 1e3, 2))
m("LatBoundMs", num(prov["latency_bound_s"] * 1e3, 0))

# executed scenarios -----------------------------------------------------
S = R["executed_summary"]
within = [s for s in S if s["within_assumptions"]]
m("NexecRuns", f"{sum(s['runs'] for s in S):,}".replace(",", "{,}"))
m("NwithinRuns", f"{sum(s['runs'] for s in within):,}".replace(",", "{,}"))
m("NwithinSucc", f"{sum(s['successes'] for s in within):,}".replace(",", "{,}"))
m("NwithinScen", str(len({s['scenario'] for s in within})))
m("NscenAll", str(len({s['scenario'] for s in S})))
ts = sorted({s["t"] for s in S})
m("ExecTs", " and ".join(f"$t={t}$" for t in ts))
runs = {s["t"]: s["runs"] for s in S}
m("ExecRunsSmallT", str(runs[ts[0]]))
m("ExecRunsLargeT", str(runs[ts[-1]]))
m("NwithinFail", str(sum(s['runs'] - s['successes'] for s in within)))
def sc_t(name, t):
    return next(x for x in S if x["scenario"] == name and x["t"] == t)
for name, tag in (("absent", "Absent"), ("slow_links", "SlowLinks")):
    for t, tt in ((ts[0], "SmallT"), (ts[-1], "LargeT")):
        x = sc_t(name, t)
        m(f"{tag}Succ{tt}", f"{x['successes']}/{x['runs']}")
inv_ok = all(all(v == (x["successes"] if k == "audit_secret_and_key" else x["runs"])
                 for k, v in x["invariants_held"].items()) for x in within)
m("InvariantsAllHeldCap", "Every" if inv_ok else "Not every")
delays = [abs(o["completion"]["diff"]) for o in R["attack_overheads"]
          if next(x for x in S if x["scenario"] == o["scenario"]
                  and x["t"] == o["t"])["within_assumptions"]]
m("MaxAttackDelayS", num(max(delays), 1))
rec = [s["recovery_latency_s"]["mean"] for s in S
       if s["scenario"] == "omission_cured" and s["recovery_latency_s"].get("count")]
m("OmissionRecoveryS", num(sum(rec) / len(rec), 1))
small = [s for s in S if s["t"] == ts[0]]
def rate(sc):
    s = next(x for x in small if x["scenario"] == sc)
    return 100.0 * s["runs_with_rewards_to_faulty"] / s["runs"]
m("ColluderRewardMin", num(min(rate(x) for x in ("commission_all", "bad_pok", "mixed")), 0))
m("ColluderRewardMax", num(max(rate(x) for x in ("commission_all", "bad_pok", "mixed")), 0))
ov = {(o["scenario"], o["t"]): o for o in R["attack_overheads"]}
m("ExtraGasBadPokLargeT", num(ov[("bad_pok", ts[-1])]["extra_gas_mean"] / 1e6, 1))
m("ExtraGasCommAllLargeT", num(ov[("commission_all", ts[-1])]["extra_gas_mean"] / 1e6, 1))

# performance -------------------------------------------------------------
P = {(r["n"], r["chain"]): r for r in R["performance"]}
b127, e127 = P[(127, "bft-1s")], P[(127, "ethereum-l1")]
m("MBnodeOneTwoSeven", num(b127["sent_MB_per_node"], 2))
m("MBnodeSixtyFour", num(P[(64, "bft-1s")]["sent_MB_per_node"], 2))
m("ComplBFTOneTwoSeven", num(b127["completion_s"], 0))
m("ComplEthOneTwoSeven", num(e127["completion_s"] / 3600, 2))
m("ComplBFTKilo", num(P[(1024, "bft-1s")]["completion_s"], 0))
m("GasNodeOneTwoSeven", num(b127["gas_per_participant"] / 1e3, 0))
m("GasTotalOneTwoSeven", num(b127["gas_total"] / 1e6, 1))
m("LedgerKBOneTwoSeven", num(b127["ledger_write_bytes"] / 1e3, 1))
sc = R["scaling"]["bft-1s"]
m("ExpBytesNode", num(sc["sent_bytes_per_node"]["exponent"], 2))
m("ExpBytesTotal", num(sc["p2p_total_bytes"]["exponent"], 2))
m("ExpGas", num(sc["gas_total"]["exponent"], 2))
m("ExpComplBFT", num(sc["completion_s"]["exponent"], 2))
m("ExpComplEth", num(R["scaling"]["ethereum-l1"]["completion_s"]["exponent"], 2))
g256 = next(g for g in R["gas_per_transaction"] if g["n"] == 256)
m("GasDisputeTwoFiveSix", num(g256["dispute"] / 1e6, 2))
m("GasPokFraudTwoFiveSix", num(g256["pok_fraud"] / 1e6, 2))
m("GasRegister", f"{g256['register']:,}".replace(",", "{,}"))

# published anchors (as reported) -----------------------------------------
A = {a.id: a for a in bl.ANCHORS}
m("DasTwentyThreeMBOneTwoEight", num(A["das23.t3.das23.bw.128"].value, 2))
m("HtwoCTMBOneTwoEight", num(A["h2ct.t3.h2ct.bw.128"].value, 2))
m("DasTwentyThreeMBSixtyFour", num(A["das23.t3.das23.bw.64"].value, 2))
m("EthDKGDisputeM", num(A["ethdkg.gas.dispute.256"].value / 1e6, 1))
m("EthDKGRegister", f"{int(A['ethdkg.gas.registration'].value):,}".replace(",", "{,}"))

# dependability -----------------------------------------------------------
D = R["dependability"]["bft-1s"]
E = R["dependability"]["ethereum-l1"]
v = D["heldout_validation"]
m("HeldRuns", str(v["runs"]))
m("HeldPred", num(v["predicted_success"], 3))
m("HeldObs", num(v["observed_success"], 3))
m("HeldLo", num(v["observed_wilson95"][0], 3))
m("HeldHi", num(v["observed_wilson95"][1], 3))
m("HeldBinomP", num(v["binom_p"], 2))
m("AttemptBFT", num(D["attempt_length_s"], 0))
m("AttemptEthH", num(E["attempt_length_s"] / 3600, 2))
m("ExposureRatio", num(E["attempt_length_s"] / D["attempt_length_s"], 0))
one = lambda d: next(c for c in d["curves"] if c["fault_rate_per_hour"] == 1.0)
m("SuccOnePerHourBFT", num(one(D)["success_stages_50"], 4))
m("SuccOnePerHourEth", num(one(E)["success_stages_50"], 3))
m("TTKOnePerHourEthH", num(one(E)["time_to_key_s_stages_50"] / 3600, 2))
m("CtmcT", str(D["t"]))
m("InjectReps", str(next(iter(D["conditional_failure"].values()))["runs"]))
m("EthKeyFailPct", num(100 * E["conditional_failure"]["KEY_DERIVATION"]["p_fail"], 0))

print("% Generated by make_macros.py from", sys.argv[1], "-- do not edit by hand.")
print(f"% results sha256 = {__import__('hashlib').sha256(open(sys.argv[1], 'rb').read()).hexdigest()}")
print("\n".join(out))
