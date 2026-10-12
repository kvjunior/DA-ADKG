r"""
figures.py -- Figure 1 and the LaTeX tables, generated ONLY from a results
file written by experiments.py (plus the published anchors in baselines.py).

    python3 figures.py RESULTS.json [OUTDIR]     # generate everything
    python3 figures.py                           # self-tests

This module computes nothing that is reported: no model, no constant, no
statistical test.  Every plotted value and every table cell is read from the
results file (fresh DA-ADKG measurements) or from baselines.ANCHORS
(published numbers, labelled "as reported").  Each output file starts with
the SHA-256 of the results file it came from and the unit-cost source.

OUTPUTS
    figure1.pdf / figure1.png        3 x 3 evaluation grid (panels below)
    table_I_properties.tex           Table I  (qualitative, cited cells)
    table_II_comparison.tex          Table II (published anchors + DA-ADKG)
    table_III_scalability.tex        Table III (estimator, both chains)
    table_scenarios.tex              Sec. VII-B: executed scenarios
    table_gas.tex                    Gas per transaction vs EthDKG / Sober'23
    manifest.json                    inputs, hashes and outputs
    (the tables need \usepackage{multirow,array} in the paper preamble)

PANELS
    (a) completion time vs n, two chain profiles (+ executed runs)
    (b) bytes sent per node: DA-ADKG (P2P, ledger) vs Das'22/Das'23 (as reported)
    (c) success per scenario with Wilson 95% intervals
    (d) gas per scenario: protocol path vs accountability transactions
    (e) gas per transaction vs n, with EthDKG / Sober'23 (as reported)
    (f) success per attempt vs exogenous fault rate (+ held-out validation)
    (g) conditional failure probability per phase (Wilson 95%)
    (h) expected time to a usable key vs fault rate, two chains
    (i) growth: per-node bytes, P2P total, ledger bytes with n, n^2, n^3 guides

STYLE
    IEEE two-column width (7.16 in), 8 pt text, black text.  Series are told
    apart by colour (Wong palette, validated: CVD separation >= 6 with the
    secondary encoding used here) AND by marker and line style, so the figure
    survives grayscale printing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import baselines as bl  # noqa: E402

SCHEMA = "dadkg-experiments/2"

# Categorical colours in FIXED order (never cycled); each series also has
# its own marker and line style.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
INK, MUTED = "black", "#6b6b6b"
CHAIN_LABEL = {"bft-1s": "BFT chain (1 s, 1-block finality)",
               "ethereum-l1": "Ethereum L1 (12 s, 96-slot finality)"}
PHASE_SHORT = {"REGISTRATION": "REG", "DEALING": "DEAL", "COMPLAINT": "COMPL",
               "ANSWER": "ANS", "DISPUTE": "DISP", "KEY_DERIVATION": "KEY"}
PROTOCOL_GAS = ("registration", "commitment", "key_derivation")


def _style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4,
        "legend.frameon": False, "figure.dpi": 150, "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _series(i: int) -> dict:
    return {"color": COLORS[i], "marker": MARKERS[i], "linestyle": LINES[i]}


# ==========================================================================
# 1. Loading and provenance.
# ==========================================================================


class Results:
    def __init__(self, path: str) -> None:
        with open(path, "rb") as fh:
            raw = fh.read()
        self.path = path
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.d = json.loads(raw)
        prov = self.d.get("provenance", {})
        if prov.get("schema") != SCHEMA:
            raise ValueError(f"{path}: schema {prov.get('schema')!r}, expected {SCHEMA!r}")
        if not str(prov.get("unit_costs", {}).get("source", "")):
            raise ValueError(f"{path}: unit-cost source missing; refusing to report")

    def __getitem__(self, k):
        return self.d[k]

    @property
    def prov(self) -> dict:
        return self.d["provenance"]

    def header(self, comment: str = "%") -> str:
        u = self.prov["unit_costs"]
        src = hashlib.sha256(json.dumps(self.prov["source_sha256"],
                                        sort_keys=True).encode()).hexdigest()[:16]
        return "\n".join(f"{comment} {line}" for line in (
            "Generated by figures.py -- do not edit by hand.",
            f"results: {os.path.basename(self.path)} sha256={self.sha256}",
            f"config: {self['config']['name']}; created {self.prov['created_utc']}; "
            f"source digest {src}",
            f"unit costs: {u['source']}; mul = {u['mul'] * 1e3:.3f} ms; "
            f"CPU: {self.prov.get('cpu_model', 'unknown')}",
        )) + "\n"

    def perf(self, chain: str) -> List[dict]:
        return sorted((r for r in self["performance"] if r["chain"] == chain),
                      key=lambda r: r["n"])

    def summary(self, t: Optional[int] = None) -> List[dict]:
        t = t if t is not None else max(self["config"]["exec_ts"])
        return [s for s in self["executed_summary"] if s["t"] == t]


# ==========================================================================
# 2. Panels.
# ==========================================================================


def panel_a(ax, R: Results) -> None:
    for i, ch in enumerate(R["config"]["perf_chains"]):
        rows = R.perf(ch)
        ax.plot([r["n"] for r in rows], [r["completion_s"] for r in rows],
                label=f"estimator, {ch}", **_series(i))
    ex = [(s["n"], s["completion_s"]["mean"]) for s in R["executed_summary"]
          if s["scenario"] == "honest" and s["completion_s"].get("count")]
    if ex:
        ax.plot(*zip(*ex), linestyle="none", marker="x", color=INK, markersize=6,
                label="executed (honest)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("network size $n$")
    ax.set_ylabel("completion time (s)")
    ax.set_title("(a) Completion time, honest")
    ax.legend(loc="center left")


def panel_b(ax, R: Results) -> None:
    rows = R.perf("bft-1s")
    ns = [r["n"] for r in rows]
    ax.plot(ns, [r["sent_MB_per_node"] for r in rows], label="DA-ADKG P2P sent", **_series(0))
    ax.plot(ns, [r["ledger_read_bytes_per_node"] / 1e6 for r in rows],
            label="DA-ADKG ledger read", **_series(1))
    published = (("das23.t3.das22-lt.bw", "Das'22, $\\ell=t$"),
                 ("das23.t3.das22-ht.bw", "Das'22, $\\ell=2t$"),
                 ("das23.t3.das23.bw", "Das'23, $\\ell=2t$"))
    for j, (prefix, lab) in enumerate(published):
        pts = sorted((a.n, a.value) for a in bl.ANCHORS if a.id.startswith(prefix + "."))
        ax.plot(*zip(*pts), linestyle="none", fillstyle="none", markersize=5,
                color=COLORS[2 + j], marker=MARKERS[2 + j],
                label=lab + " (reported)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("network size $n$")
    ax.set_ylabel("MB per node")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo / 300, hi)             # room under the curves for the legend
    ax.set_title("(b) Bytes per node")
    ax.legend(loc="lower right", fontsize=5.5)


def _scenario_order(R: Results, rows: Sequence[dict]) -> List[dict]:
    names = list(R["scenarios"])
    return sorted(rows, key=lambda s: (not s["within_assumptions"], names.index(s["scenario"])))


def panel_c(ax, R: Results) -> None:
    rows = _scenario_order(R, R.summary())
    y = np.arange(len(rows))[::-1]
    for yi, s in zip(y, rows):
        lo, hi = s["success_wilson95"]
        c = COLORS[0] if s["within_assumptions"] else COLORS[1]
        m = MARKERS[0] if s["within_assumptions"] else MARKERS[1]
        ax.plot([100 * lo, 100 * hi], [yi, yi], color=c, linewidth=1.0)
        ax.plot([100 * s["success_rate"]], [yi], marker=m, color=c, linestyle="none")
    ax.set_yticks(y)
    ax.set_yticklabels([s["scenario"].replace("_", " ") for s in rows], fontsize=5.5)
    n_in = sum(s["within_assumptions"] for s in rows)
    if 0 < n_in < len(rows):
        ax.axhline(y[n_in - 1] - 0.5, color=MUTED, linewidth=0.6, linestyle="--")
    ax.plot([], [], marker=MARKERS[0], color=COLORS[0], label="within assumptions")
    ax.plot([], [], marker=MARKERS[1], color=COLORS[1], label="beyond assumptions")
    ax.set_xlim(-3, 103)
    ax.set_xlabel("success (%), Wilson 95%")
    t, runs = rows[0]["t"], rows[0]["runs"]
    ax.set_title(f"(c) Success, $t={t}$, {runs} runs each")
    ax.legend(loc="center left", fontsize=5.5)


def panel_d(ax, R: Results) -> None:
    rows = [s for s in _scenario_order(R, R.summary()) if s["within_assumptions"]]
    y = np.arange(len(rows))[::-1]
    base = np.array([sum(v for k, v in s["gas_by_category_mean"].items()
                         if k in PROTOCOL_GAS) for s in rows]) / 1e6
    acc = np.array([sum(v for k, v in s["gas_by_category_mean"].items()
                        if k not in PROTOCOL_GAS) for s in rows]) / 1e6
    ax.barh(y, base, color=COLORS[0], edgecolor="white", linewidth=0.8,
            label="protocol txs")                 # register, anchor, key
    ax.barh(y, acc, left=base, color=COLORS[1], edgecolor="white", linewidth=0.8,
            hatch="////", label="accountability txs")  # complaint, answer, dispute, fraud
    ax.set_xlim(0, float(np.max(base + acc)) * 1.5)   # legend sits right of the bars
    ax.set_yticks(y)
    ax.set_yticklabels([s["scenario"].replace("_", " ") for s in rows], fontsize=5.5)
    ax.set_xlabel("gas per run (millions)")
    ax.set_title(f"(d) Gas per run, $t={rows[0]['t']}$")
    ax.legend(loc="upper right", fontsize=5.5)


def panel_e(ax, R: Results) -> None:
    g = sorted(R["gas_per_transaction"], key=lambda r: r["n"])
    ns = [r["n"] for r in g]
    for i, (k, lab) in enumerate((("register", "register"), ("anchor", "anchor"),
                                  ("dispute", "dispute"), ("pok_fraud", "PoK fraud"))):
        ax.plot(ns, [r[k] for r in g], label=f"DA-ADKG {lab}", **_series(i))
    eth = [(256, a.value) for a in bl.anchors(source="schindler2019ethdkg", n=256)
           if a.operation == "dispute"]
    eth += [(n, a.value) for a in bl.anchors(source="schindler2019ethdkg")
            if a.operation == "registration" for n in (4, 256)]
    ax.plot(*zip(*sorted(eth)), linestyle="none", marker=MARKERS[4], fillstyle="none",
            markersize=6, color=COLORS[4], label="EthDKG reg./dispute (reported)")
    sob = sorted((a.n, a.value) for a in bl.anchors(source="sober2023distributed",
                                                    metric="gas")
                 if a.operation == "dispute")
    ax.plot(*zip(*sob), linestyle="none", marker=MARKERS[5], fillstyle="none",
            markersize=6, color=COLORS[5], label="Sober'23 dispute (reported)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("network size $n$")
    ax.set_ylabel("gas per transaction")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 40)              # head-room so the legend clears the data
    ax.set_title("(e) Gas per transaction")
    ax.legend(loc="upper left", fontsize=5.5)


def panel_f(ax, R: Results) -> None:
    d = R["dependability"]["bft-1s"]
    rates = [c["fault_rate_per_hour"] for c in d["curves"]]
    ax.plot(rates, [100 * c["success_stages_1"] for c in d["curves"]],
            label="CTMC, exponential phases", **_series(0))
    ax.plot(rates, [100 * c["success_stages_50"] for c in d["curves"]],
            label="CTMC, 50-stage phases", **_series(1))
    v = d["heldout_validation"]
    if v:
        r_h = v["rate_per_s"] * 3600
        lo, hi = v["observed_wilson95"]
        ax.errorbar([r_h], [100 * v["observed_success"]],
                    yerr=[[max(0.0, 100 * (v["observed_success"] - lo))],
                          [max(0.0, 100 * (hi - v["observed_success"]))]],
                    fmt="x", color=INK, capsize=2, markersize=6,
                    label=f"held-out runs ({v['runs']})")
    ax.set_xscale("log")
    ax.set_xlabel("fault events per hour")
    ax.set_ylabel("success per attempt (%)")
    ax.set_title(f"(f) Reliability, BFT chain, $t={d['t']}$")
    ax.legend(loc="lower left", fontsize=5.5)


def panel_g(ax, R: Results) -> None:
    d = R["dependability"]["bft-1s"]
    sets = [("1x window", d["conditional_failure"])]
    sets += [(f"{s['duration_factor']:g}x window", s["conditional_failure"])
             for s in d["sensitivity"]]
    x = np.arange(len(d["conditional_failure"]))
    width = 0.8 / len(sets)
    for i, (lab, cond) in enumerate(sets):
        ps = [cond[f]["p_fail"] * 100 for f in cond]
        lo = [max(0.0, 100 * (cond[f]["p_fail"] - cond[f]["wilson95"][0])) for f in cond]
        hi = [max(0.0, 100 * (cond[f]["wilson95"][1] - cond[f]["p_fail"])) for f in cond]
        xs = x - 0.4 + width * (i + 0.5)
        ax.errorbar(xs, ps, yerr=[lo, hi], fmt=MARKERS[i], color=COLORS[i], capsize=1.5,
                    markersize=4, linewidth=0.8,
                    label=f"{lab} ({next(iter(cond.values()))['runs']}/phase)")
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_SHORT[f] for f in d["conditional_failure"]], fontsize=6,
                       rotation=30)
    ax.set_ylim(-3, 103)
    ax.set_ylabel("P(fail | one event) (%)")
    ev = d["event"]
    ax.set_title(f"(g) Failure per phase, {ev['group_size']} nodes cut off")
    ax.legend(loc="upper left", fontsize=5.5)


def panel_h(ax, R: Results) -> None:
    for i, (ch, d) in enumerate(R["dependability"].items()):
        rates = [c["fault_rate_per_hour"] for c in d["curves"]]
        ys = [c["time_to_key_s_stages_50"] / 3600 for c in d["curves"]]
        keep = [(r, y) for r, y in zip(rates, ys) if np.isfinite(y) and y < 1e4]
        if keep:
            ax.plot(*zip(*keep), label=ch, **_series(i))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fault events per hour")
    ax.set_ylabel("expected time to key (h)")
    ax.set_title("(h) Time to a usable key")
    ax.legend(loc="upper left", fontsize=5.5)


def panel_i(ax, R: Results) -> None:
    rows = R.perf("bft-1s")
    ns = np.array([r["n"] for r in rows], float)
    series = (("sent_bytes_per_node", "P2P sent per node"),
              ("p2p_total_bytes", "P2P total"),
              ("ledger_write_bytes", "ledger writes"))
    for i, (k, lab) in enumerate(series):
        ax.plot(ns, [r[k] for r in rows], label=lab, **_series(i))
    n_min = R["scaling"]["bft-1s"]["sent_bytes_per_node"]["n_range"][0]
    big = ns[ns >= n_min]
    for k, power, lab in (("sent_bytes_per_node", 2, "$n^2$"),
                          ("p2p_total_bytes", 3, "$n^3$"),
                          ("ledger_write_bytes", 1, "$n$")):
        ref = rows[-1][k] * (big / ns[-1]) ** power        # guide over the fit range
        ax.plot(big, ref, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.annotate(lab, (big[0], ref[0]), textcoords="offset points", xytext=(-3, 4),
                    ha="right", va="bottom", fontsize=6, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("network size $n$")
    ax.set_ylabel("bytes")
    ax.set_title(f"(i) Growth; guides for $n \\geq {n_min:.0f}$")
    ax.legend(loc="upper left", fontsize=5.5)


PANELS = (panel_a, panel_b, panel_c, panel_d, panel_e, panel_f, panel_g, panel_h, panel_i)


def make_figure1(R: Results, pdf: str, png: str) -> plt.Figure:
    _style()
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 7.4), constrained_layout=True)
    for fn, ax in zip(PANELS, axes.flat):
        fn(ax, R)
    fig.savefig(pdf, metadata={"Subject": f"results sha256 {R.sha256}"})
    fig.savefig(png, dpi=300)
    return fig


# ==========================================================================
# 3. LaTeX tables.
# ==========================================================================

_UNI = {
    "κ": r"\kappa", "λ": r"\lambda", "ℓ": r"\ell", "Θ": r"\Theta", "Δ": r"\Delta",
    "α": r"\alpha", "γ": r"\gamma", "≥": r"\geq", "≤": r"\leq", "−": "-",
    "→": r"\rightarrow", "·": r"\cdot", "⌈": r"\lceil", "⌉": r"\rceil",
    "²": "^{2}", "³": "^{3}", "⁴": "^{4}", "Õ": r"\tilde{O}", "∈": r"\in",
}
_TEXT_UNI = {"–": "--", "§": r"\S{}"}


def tex(text: str) -> str:
    """Plain text with Unicode maths -> LaTeX.  Each special character is put
    in its own math group, so the output is valid in any text cell."""
    s = str(text)
    out, i = [], 0
    while i < len(s):
        m = re.match(r"\^\{([^}]*)\}", s[i:])
        if m:
            out.append(f"$^{{{m.group(1)}}}$")
            i += m.end()
            continue
        ch = s[i]
        if ch in _UNI:
            out.append(f"${_UNI[ch]}$")
        elif ch in _TEXT_UNI:
            out.append(_TEXT_UNI[ch])
        elif ch in "&%#_$":
            out.append("\\" + ch)
        elif ch in "{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ord(ch) > 127:
            raise ValueError(f"no LaTeX mapping for {ch!r} in {text!r}")
        else:
            out.append(ch)
        i += 1
    return "".join(out).replace("$$", "")


def f_num(x: Optional[float], nd: int = 1) -> str:
    if x is None:
        return "--"
    return f"{x:,.{nd}f}".replace(",", "{,}")


def table_I(R: Results) -> str:
    # widths sum to 16.25 cm + padding: fits the 7.16 in two-column text width
    cols = (("network", "Network model", "1.7cm"), ("shares", "Shares", "1.25cm"),
            ("comm_total", "Comm. (total)", "1.75cm"), ("rounds", "Rounds", "1.15cm"),
            ("setup", "Setup", "1.5cm"), ("adversary", "Adversary", "1.45cm"),
            ("onchain_disputes", "On-chain disputes", "1.7cm"),
            ("penalty", "Penalty", "1.6cm"), ("trusted_setup", "Trusted setup", "1.1cm"),
            ("dependability_model", "Depend. model", "1.5cm"))
    lines = [R.header(), r"\begin{table*}[!t]", r"\centering", r"\scriptsize",
             r"\caption{Properties of asynchronous and blockchain-based DKG protocols. "
             r"Each cell follows the cited paper; the DA-ADKG row describes the "
             r"revised protocol as implemented.}",
             r"\label{tab:properties}",
             r"\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{|>{\raggedright\arraybackslash}p{1.55cm}|"
             + "|".join(rf">{{\raggedright\arraybackslash}}p{{{w}}}" for *_, w in cols)
             + "|}",
             r"\hline",
             r"\textbf{Protocol} & " + " & ".join(rf"\textbf{{{h}}}" for _, h, _ in cols)
             + r" \\", r"\hline"]
    for row in bl.TABLE_I:
        name = tex(row.protocol)
        if row.key != "this-work":
            name += rf" \cite{{{row.key}}}"
        else:
            name = rf"\textbf{{{name}}}"
        lines.append(name + " & " + " & ".join(tex(row.cells[c]) for c, *_ in cols)
                     + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def table_II(R: Results) -> str:
    """Published numbers on their own testbeds, side by side with DA-ADKG at
    the same n (or the closest n = 3t+1, marked *).  No significance test."""
    perf = {(r["n"], r["chain"]): r for r in R["performance"]}

    def da(n: int) -> Tuple[str, dict, dict]:
        """DA-ADKG rows at n, or at the closest n = 3t+1 below it (marked *)."""
        exact = bl.matching_t(n) is not None
        n_da = n if exact else 3 * ((n - 1) // 3) + 1
        return (f"{n_da}{'' if exact else '*'}", perf[(n_da, "bft-1s")],
                perf[(n_da, "ethereum-l1")])

    def anchor(prefix: str, n: int) -> Optional[float]:
        hit = [x.value for x in bl.ANCHORS if x.id == f"{prefix}.{n}"]
        return hit[0] if hit else None

    L = [R.header(), r"\begin{table*}[!t]", r"\centering", r"\scriptsize",
         r"\caption{Published measurements (as reported, on each paper's testbed) and "
         r"fresh DA-ADKG results at the same $n$. Times are not comparable across "
         r"rows: published values are wall-clock on the stated testbed; DA-ADKG "
         r"values are simulated completion under the stated chain, latency model and "
         r"unit costs measured on our machine. MB as printed by each source; DA-ADKG "
         r"MB $=10^6$ bytes. * closest $n=3t+1$.}",
         r"\label{tab:comparison}",
         r"\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}", r"\hline",
         r"\multirow{2}{*}{$n$} & \multicolumn{2}{c|}{Das'22, $\ell=t$} & "
         r"\multicolumn{2}{c|}{Das'22, $\ell=2t$} & \multicolumn{2}{c|}{Das'23, $\ell=2t$} & "
         r"\multicolumn{4}{c|}{DA-ADKG (this work, degree $2t$)} \\",
         r"\cline{2-11}",
         r" & s & MB & s & MB & s & MB & $n$ & s (BFT) & h (Eth.) & MB P2P \\",
         r"\hline",
         r"\multicolumn{11}{|l|}{\textit{Das'23 Table 3: AWS t3a.medium, 8 regions, "
         r"Python + Rust, curve25519 \cite{das2023practical}}} \\", r"\hline"]
    for n in (16, 32, 64, 128):
        lab, b, e = da(n)
        cells = []
        for pre in ("das22-lt", "das22-ht", "das23"):
            cells += [f_num(anchor(f"das23.t3.{pre}.rt", n), 2),
                      f_num(anchor(f"das23.t3.{pre}.bw", n), 2)]
        L.append(f"{n} & " + " & ".join(cells) + f" & {lab} & {f_num(b['completion_s'], 0)}"
                 f" & {f_num(e['completion_s'] / 3600, 2)} & "
                 f"{f_num(b['sent_MB_per_node'], 3)} \\\\")
        L.append(r"\hline")
    L += [r"\multicolumn{11}{|l|}{\textit{Zhang'23 Fig.~4a: AWS t3a.medium, 4 regions, Go; "
          r"reconstruction threshold $2t+1$ \cite{zhang2023practical}}} \\", r"\hline",
          r"$n$ & \multicolumn{2}{c|}{Zhang'23 (s)} & \multicolumn{2}{c|}{DYX+ (s)} & "
          r"\multicolumn{2}{c|}{} & $n$ & s (BFT) & h (Eth.) & MB P2P \\", r"\hline"]
    for n in (4, 13, 25, 49):
        lab, b, e = da(n)
        L.append(f"{n} & \\multicolumn{{2}}{{c|}}{{{f_num(anchor('zhang23.f4a.zhang23.rt', n), 2)}}}"
                 f" & \\multicolumn{{2}}{{c|}}{{{f_num(anchor('zhang23.f4a.dyx.rt', n), 2)}}}"
                 f" & \\multicolumn{{2}}{{c|}}{{}} & {lab} & {f_num(b['completion_s'], 0)} & "
                 f"{f_num(e['completion_s'] / 3600, 2)} & {f_num(b['sent_MB_per_node'], 3)} \\\\")
        L.append(r"\hline")
    lab, b, e = da(128)
    h = {a.system: a.value for a in bl.anchors(group="h2ct-eval", n=128)}
    lab2, b2, e2 = da(1024)
    h2 = [a.value for a in bl.anchors(group="h2ct-eval", n=1024) if a.curve == "curve25519"]
    L += [r"\multicolumn{11}{|l|}{\textit{H$^2$CT Sec.~VI: AWS t3.medium, Docker, Python; "
          r"$\ell=2t$, curve25519 \cite{xiang2024h2ct}}} \\", r"\hline",
          f"128 & \\multicolumn{{6}}{{l|}}{{MB per node: Das'22 {f_num(h['Das' + chr(39) + '22'], 2)}, "
          f"Das'23 {f_num(h['Das' + chr(39) + '23'], 2)}, H$^2$CT {f_num(h['H2CT'], 2)}}} & {lab} & "
          f"{f_num(b['completion_s'], 0)} & {f_num(e['completion_s'] / 3600, 2)} & "
          f"{f_num(b['sent_MB_per_node'], 3)} \\\\", r"\hline",
          f"1024 & \\multicolumn{{6}}{{l|}}{{H$^2$CT runtime: {f_num(h2[0], 2)} s}} & {lab2} & "
          f"{f_num(b2['completion_s'], 0)} & {f_num(e2['completion_s'] / 3600, 2)} & "
          f"{f_num(b2['sent_MB_per_node'], 3)} \\\\", r"\hline",
          r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def table_III(R: Results) -> str:
    sc = R["scaling"]
    L = [R.header(), r"\begin{table*}[!t]", r"\centering", r"\scriptsize",
         r"\caption{DA-ADKG scalability (honest case, estimator validated against "
         r"executed runs). Completion on two chain profiles; bytes are payload bytes; "
         r"gas from the EVM schedule (EIP-2929 prices).}",
         r"\label{tab:scalability}",
         r"\begin{tabular}{|r|r|r|r|r|r|r|r|r|}", r"\hline",
         r"$n$ & $t$ & Compl. BFT (s) & Compl. Eth. (h) & MB/node sent & "
         r"MB P2P total & Ledger (KB) & Gas (M) & Gas/node (K) \\", r"\hline"]
    eth = {r["n"]: r for r in R.perf("ethereum-l1")}
    for r in R.perf("bft-1s"):
        e = eth.get(r["n"])
        L.append(f"{r['n']} & {r['t']} & {f_num(r['completion_s'], 0)} & "
                 f"{f_num(e['completion_s'] / 3600, 2) if e else '--'} & "
                 f"{f_num(r['sent_MB_per_node'], 3)} & {f_num(r['p2p_total_bytes'] / 1e6, 3)} & "
                 f"{f_num(r['ledger_write_bytes'] / 1e3, 1)} & {f_num(r['gas_total'] / 1e6, 2)} & "
                 f"{f_num(r['gas_per_participant'] / 1e3, 1)} \\\\")
        L.append(r"\hline")
    b = sc["bft-1s"]

    def ex(k: str) -> str:
        return f"$n^{{{b[k]['exponent']:.2f}\\pm{b[k]['stderr']:.2f}}}$"
    lo, hi = b["sent_bytes_per_node"]["n_range"]
    L.append(r"\multicolumn{9}{|p{15.5cm}|}{Growth for $" + f"{lo:.0f}" + r"\le n\le " + f"{hi:.0f}"
             + r"$ (log-log slope $\pm$ s.e.): bytes/node " + ex("sent_bytes_per_node")
             + ", P2P total " + ex("p2p_total_bytes") + ", gas " + ex("gas_total")
             + ", completion (BFT) " + ex("completion_s") + r".} \\")
    L += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def table_scenarios(R: Results) -> str:
    rows = _scenario_order(R, R.summary())
    over = {(o["scenario"], o["t"]): o for o in R["attack_overheads"]}
    L = [R.header(), r"\begin{table*}[!t]", r"\centering", r"\scriptsize",
         r"\caption{Executed scenarios (real cryptography). Success with Wilson 95\% "
         r"interval; invariants are the accountability checks of Sec.~\ref{sec:evaluation}, counted "
         r"over runs; extra gas and completion difference are relative to the honest "
         r"runs with the same $t$.}",
         r"\label{tab:scenarios}",
         r"\begin{tabular}{|l|c|c|c|c|c|c|c|}", r"\hline",
         r"Scenario & Assumptions & Runs & Success (\%) & Invariants held & "
         r"Extra gas (M) & $\Delta$ completion (s) & Recovery (s) \\", r"\hline"]
    for s in rows:
        lo, hi = s["success_wilson95"]
        inv = s["invariants_held"]
        n_ok = sum(v == s["runs"] for v in inv.values())
        o = over.get((s["scenario"], s["t"]))
        rec = s["recovery_latency_s"]
        L.append(f"{tex(s['scenario'].replace('_', ' '))} & {'within' if s['within_assumptions'] else 'beyond'} & "
                 f"{s['runs']} & {f_num(100 * s['success_rate'], 0)} "
                 f"[{f_num(100 * lo, 0)}, {f_num(100 * hi, 0)}] & {n_ok}/{len(inv)} & "
                 f"{f_num(o['extra_gas_mean'] / 1e6, 2) if o else '--'} & "
                 f"{f_num(o['completion']['diff'], 1) if o else '--'} & "
                 f"{f_num(rec['mean'], 1) if rec.get('count') else '--'} \\\\")
        L.append(r"\hline")
    L += [r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def table_gas(R: Results) -> str:
    g = {r["n"]: r for r in R["gas_per_transaction"]}
    d = g.get(256)
    eth = {a.operation: a.value for a in bl.anchors(source="schindler2019ethdkg")}
    sob = {a.operation: a.value for a in bl.anchors(source="sober2023distributed", metric="gas")
           if a.n in (256, None)}
    L = [R.header(), r"\begin{table}[!t]", r"\centering", r"\scriptsize",
         r"\caption{Gas per transaction at $n=256$. DA-ADKG: EVM schedule with EIP-2929 "
         r"prices, $n=3t+1$, commitment of $2t+1$ points. EthDKG: as reported, after "
         r"Istanbul, $n=2t+1$. Sober'23: as reported, local Hardhat.}",
         r"\label{tab:gas}",
         r"\begin{tabular}{|l|r|r|r|}", r"\hline",
         r"Transaction & DA-ADKG & EthDKG \cite{schindler2019ethdkg} & "
         r"Sober'23 \cite{sober2023distributed} \\", r"\hline"]
    rows = (("Registration", d["register"], eth.get("registration"), None),
            ("Commitment / share", d["anchor"], None, sob.get("share")),
            ("Dispute", d["dispute"], eth.get("dispute"), sob.get("dispute")),
            ("Answer to complaint", d["answer"], None, None),
            ("PoK fraud proof", d["pok_fraud"], None, None),
            ("Key submission (finalising)", d["submit_key_finalizing"],
             eth.get("master_key_submission"), sob.get("derive")))
    for lab, a, e, s in rows:
        L.append(f"{lab} & {f_num(a, 0)} & {f_num(e, 0)} & {f_num(s, 0)} \\\\")
        L.append(r"\hline")
    L += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


TABLES = {"table_I_properties.tex": table_I, "table_II_comparison.tex": table_II,
          "table_III_scalability.tex": table_III, "table_scenarios.tex": table_scenarios,
          "table_gas.tex": table_gas}


# ==========================================================================
# 4. Driver.
# ==========================================================================


def generate_all(results_path: str, outdir: str) -> Dict[str, str]:
    R = Results(results_path)
    os.makedirs(outdir, exist_ok=True)
    out = {"figure_pdf": os.path.join(outdir, "figure1.pdf"),
           "figure_png": os.path.join(outdir, "figure1.png")}
    fig = make_figure1(R, out["figure_pdf"], out["figure_png"])
    plt.close(fig)
    for name, fn in TABLES.items():
        p = os.path.join(outdir, name)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(fn(R))
        out[name] = p
    manifest = {"results": os.path.abspath(results_path), "results_sha256": R.sha256,
                "provenance": R.prov,
                "outputs": {k: hashlib.sha256(open(v, "rb").read()).hexdigest()
                            for k, v in out.items()}}
    out["manifest"] = os.path.join(outdir, "manifest.json")
    with open(out["manifest"], "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return out


def compile_tables(paths: Sequence[str], workdir: str) -> Tuple[bool, str]:
    """Compile every table in a minimal IEEEtran-like document (pdflatex)."""
    doc = [r"\documentclass{article}",
           r"\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=7.16in,"
           r"textheight=9.5in]{geometry}",
           r"\usepackage{multirow}", r"\usepackage{array}", r"\usepackage{amsmath}",
           r"\begin{document}"]
    doc += [r"\input{" + os.path.abspath(p).replace("\\", "/") + "}" for p in paths]
    doc += [r"\end{document}"]
    tex_path = os.path.join(workdir, "tables_check.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(doc))
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "tables_check.tex"], cwd=workdir, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    log = r.stdout
    over = [float(x) for x in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)]
    return r.returncode == 0 and all(x <= 1.0 for x in over), (
        f"overfull boxes (pt): {over}\n" + log[-2000:])


# ==========================================================================
# 5. Self-tests.
# ==========================================================================


def _selftest(results_path: Optional[str] = None) -> None:  # pragma: no cover
    tmp = tempfile.mkdtemp()
    if results_path is None:
        results_path = os.path.join(tmp, "quick.json")
        here = os.path.dirname(os.path.abspath(__file__))
        print("  generating a quick results file with experiments.py (about 2 min) ...")
        subprocess.run([sys.executable, os.path.join(here, "experiments.py"), "--run", "quick",
                        "--out", results_path, "--workers", "2"], check=True,
                       capture_output=True)
    R = Results(results_path)

    # 1. Refuse results without provenance / with another schema.
    bad = os.path.join(tmp, "bad.json")
    with open(bad, "w") as fh:
        json.dump({"provenance": {"schema": "dadkg-experiments/1"}}, fh)
    try:
        Results(bad)
        raise AssertionError("schema not checked")
    except ValueError:
        pass
    print("  PASS  results without the v2 schema / unit-cost source are refused")

    # 2. Everything is generated.
    out = generate_all(results_path, os.path.join(tmp, "out"))
    for k, p in out.items():
        assert os.path.getsize(p) > 300, k
    print(f"  PASS  generated {len(out)} files (figure, {len(TABLES)} tables, manifest)")

    # 3. Figure structure: 9 panels with data; >= 2 series -> a legend.
    _style()
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 7.4), constrained_layout=True)
    for fn, ax in zip(PANELS, axes.flat):
        fn(ax, R)
        has = ax.lines or ax.patches or ax.collections
        assert has, fn.__name__
        n_series = len([h for h in ax.get_legend_handles_labels()[1]])
        assert n_series >= 2 and ax.get_legend() is not None, fn.__name__
    labels = [t.get_text() for t in axes[2, 2].texts]
    assert {"$n$", "$n^2$", "$n^3$"} <= set(labels)                   # G4
    plt.close(fig)
    print("  PASS  9 panels, each with data and a legend; growth guides n, n^2, n^3")

    # 4. Traceability: table cells equal the results file.
    t3 = open(out["table_III_scalability.tex"]).read()
    for r in R.perf("bft-1s"):
        row = f"{r['n']} & {r['t']} & {f_num(r['completion_s'], 0)} & "
        assert row in t3, row
        assert f_num(r["sent_MB_per_node"], 3) in t3
    t2 = open(out["table_II_comparison.tex"]).read()
    for a in bl.anchors(group="das23-table3"):
        assert f_num(a.value, 2) in t2, a.id
    for a in bl.anchors(group="zhang23-fig4a"):
        assert f_num(a.value, 2) in t2, a.id
    tg = open(out["table_gas.tex"]).read()
    g256 = next(r for r in R["gas_per_transaction"] if r["n"] == 256)
    assert f_num(g256["dispute"], 0) in tg and f_num(1.5e6, 0) in tg
    ts = open(out["table_scenarios.tex"]).read()
    for s in R.summary():
        assert tex(s["scenario"].replace("_", " ")) in ts
    print("  PASS  table cells trace to the results file and to baselines.ANCHORS")

    # 5. Nothing from v1 survives (G1, G2).
    allout = "".join(open(p).read() for k, p in out.items() if k.endswith(".tex"))
    for banned in ("p<0.001", "Cohen", "Bonferroni", "attack probability",
                   "analytical models", "Wang"):
        assert banned not in allout, banned
    print("  PASS  no significance claims against models, no deterrence curve, "
          "no Wang'25")

    # 6. Every table carries the provenance header.
    for name in TABLES:
        head = open(out[name]).read().splitlines()[1]
        assert R.sha256 in head, name
    print("  PASS  every table starts with the results SHA-256 and the unit-cost source")

    # 7. Unicode -> LaTeX and compilation.
    assert tex("Õ(κn³)") == r"$\tilde{O}$($\kappa$n$^{3}$)"
    assert tex("O(λ²n^{2+1/k})") == r"O($\lambda$$^{2}$n$^{2+1/k}$)".replace("$$", "")
    if shutil.which("pdflatex"):
        ok, log = compile_tables([out[n] for n in TABLES], os.path.join(tmp, "out"))
        if not ok:
            print(log)
        assert ok
        print("  PASS  all five tables compile with pdflatex and fit the IEEE "
              "two-column text width (7.16 in), no overfull box")
    else:
        print("  SKIP  pdflatex not installed")

    print(f"\n  outputs in {os.path.join(tmp, 'out')}")
    print("figures.py: all self-tests passed.")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] not in ("--selftest",):
        outdir = sys.argv[2] if len(sys.argv) > 2 else "."
        for k, p in generate_all(sys.argv[1], outdir).items():
            print(f"  {k:<26} {p}")
    else:
        _selftest(sys.argv[2] if len(sys.argv) > 2 else None)
