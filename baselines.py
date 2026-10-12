"""
baselines.py -- Published reference numbers ("anchors") and the Table I rows.

WHAT THIS MODULE IS
    A data module.  It holds only numbers that the comparison papers PUBLISH,
    each with its source, PDF page, locator, testbed, threshold convention, the
    evidence-ledger ID (D4) and a verbatim fragment of the sentence or caption
    it comes from.  `verify_against_texts()` checks every anchor against the
    extracted text of its PDF, so no number here can drift from its source.

    It also holds the rows of the revised Table I (qualitative properties),
    where every cell cites its evidence IDs.

WHAT THIS MODULE IS NOT
    * It is NOT a performance model of the baselines.  The v1 module modelled
      five protocols with invented constants (success rates, MTTR, failure
      rates, a 50x DCR cost factor, "consensus rounds", a Groth16 proving
      constant) and asserted that DA-ADKG wins.  All of that is removed
      (D7 B1-B5).  None of the baselines is re-implemented in this artifact.
    * It produces NO DA-ADKG numbers.  Those come only from fresh runs of
      network.run_dkg_async / estimate_performance, made by experiments.py.
    * It never interpolates or extrapolates a published series.  A value is
      returned only at an n the source actually reports.

HOW THE ANCHORS MAY BE USED (the rules Table II must follow)
    1. Report each anchor "as reported", with its testbed.  Different testbeds
       (machine, language, regions, curve) are not directly comparable.  Only
       the series inside one comparability group were measured together.
    2. "MB" is as printed by each source; none of them states whether it
       means 10^6 or 2^20 bytes.  DA-ADKG bytes must be reported in bytes or
       with the unit stated (see `mb_both`).
    3. Threshold conventions differ: Das'23 and H2CT use l = polynomial degree
       (l+1 shares reconstruct), Das'22 and Zhang'23 state the reconstruction
       threshold directly.  Each anchor records both `degree` and `shares`.
    4. DA-ADKG fixes n = 3t+1.  Of the published sizes, n = 16, 64 and 49 are
       of that form; 32 and 128 are not (see `matching_t`).
    5. Gas depends on the EVM fork schedule.  EthDKG states its figures are
       after Istanbul (EIP-1108); DA-ADKG's gas model (protocol.py) uses the
       EIP-2929 cold-access prices.  State this next to any gas comparison.
    6. Byte counts are not the whole story for DA-ADKG: it buys agreement from
       the ledger (F12).  Ledger writes/reads and the chain profile must be
       reported alongside P2P bytes.

Run `python3 baselines.py [TEXT_DIR] [D4_LEDGER]` for the self-tests.  With
TEXT_DIR (the folder of extracted PDF texts, page markers
"===== PDF PAGE n =====") every anchor is re-verified against its page.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ==========================================================================
# 1. Sources.
# ==========================================================================


@dataclass(frozen=True)
class Source:
    key: str                 # BibTeX key used in the manuscript / D1
    short: str               # label used in tables
    venue: str
    text_file: str           # stem of the extracted-text file in TEXT_DIR
    testbed: str             # as stated by the source (verbatim fragments)
    implementation: str      # as stated by the source
    testbed_quotes: Tuple[Tuple[int, str], ...] = ()   # (page, verbatim)


SOURCES: Dict[str, Source] = {s.key: s for s in (
    Source(
        key="das2022practical", short="Das'22", venue="IEEE S&P 2022",
        text_file="Practical Asynchronous Distributed Key Generation",
        testbed="AWS t3a.medium, one node per VM, 2 vCPUs, 4 GB RAM, "
                "Ubuntu 20.04, nodes spread evenly over 8 AWS regions",
        implementation="Python 3.7.6 on hbACSS, Rust EC libraries",
        testbed_quotes=(
            (12, "t3a.medium virtual machines (VM) with one node per VM. Each "
                 "VM has two vCPUs and 4GB RAM and runs Ubuntu 20.04."),
            (12, "We place nodes evenly across eight different AWS regions"),
            (11, "using python 3.7.6 on top of the open-source hbACSS library"),
        )),
    Source(
        key="das2023practical", short="Das'23", venue="USENIX Security 2023",
        text_file="Practical Asynchronous High-threshold Distributed Key",
        testbed="AWS t3a.medium, one node per VM, 2 vCPUs, 4 GB RAM, "
                "Ubuntu 20.04, nodes spread evenly over 8 AWS regions",
        implementation="Python 3.7.13 on the Das'22 code, Rust EC libraries",
        testbed_quotes=(
            (13, "t3a.medium virtual machines (VM) with one node per VM. Each "
                 "VM has two vCPUs and 4GB RAM and runs Ubuntu 20.04."),
            (13, "We place nodes evenly across eight AWS regions"),
            (12, "We have implemented a prototype of our ADKG protocol in "
                 "python 3.7.13"),
        )),
    Source(
        key="zhang2023practical", short="Zhang'23", venue="IEEE/IFIP DSN 2023",
        text_file="Practical Asynchronous Distributed Key Generation (2)",
        testbed="AWS EC2 t3a.medium (2 vCPUs, 4 GB RAM, Ubuntu 20.04), up to "
                "49 VMs evenly over 4 EC2 regions",
        implementation="Go (gRPC)",
        testbed_quotes=(
            (12, "We implement our ADKG protocol using Golang."),
            (12, "evenly distributed in four EC2 regions. Each VM is a "
                 "t3a.medium instance (with two virtual CPUs and 4GB RAM)"),
        )),
    Source(
        key="xiang2024h2ct", short="H2CT", venue="IEEE IoT-J 2024",
        text_file="H²CT",
        testbed="AWS EC2 t3.medium with Docker (2 vCPUs, 4 GB RAM, Ubuntu "
                "22.04); the paper also states benchmarks run in a single "
                "vCPU inside a Docker container",
        implementation="Python on the Das'23 ADKG and ACSS code, Rust pairing "
                       "library",
        testbed_quotes=(
            (11, "we deploy all nodes on AWS EC2 t3.medium virtual machines "
                 "(VMs) with Docker, where each node per VM has two vCPUs and "
                 "4-GB RAM and runs Ubuntu 22.04"),
            (11, "all benchmarks only run in a single vCPU and is done inside "
                 "a Docker container"),
        )),
    Source(
        key="schindler2019ethdkg", short="EthDKG", venue="IACR ePrint 2019/985",
        text_file="EthDKG",
        testbed="Gas measured per transaction; worst case n = 2t+1 with t "
                "adversarial parties; after the Istanbul hard fork (EIP-1108); "
                "contract also deployed on the Ropsten testnet",
        implementation="Solidity contract + Python client",
        testbed_quotes=(
            (11, "We use a setup with n = 2t + 1 participants, where t parties "
                 "are executing adversarial actions"),
            (12, "as part of Ethereum’s Istanbul hardfork"),
            (1, "execution of our DKG contract in the Ethereum Ropsten testnet"),
        )),
    Source(
        key="sober2023distributed", short="Sober'23", venue="ACM SAC 2023",
        text_file="Distributed Key Generation with Smart Contracts",
        testbed="Amazon EC2, Intel Xeon Platinum 8259CL, 8 cores at 2.50 GHz, "
                "32 GB RAM, Ubuntu 22.04; local Hardhat network 2.11.1",
        implementation="Solidity contracts, Go client, ZoKrates Groth16 over "
                       "Baby Jubjub",
        testbed_quotes=(
            (7, "Intel(R) Xeon(R) Platinum 8259CL running Ubuntu 22.04 using "
                "eight cores with a clock speed of 2.50GHz"),
            (7, "we spin up a local blockchain instance using Hardhat network "
                "2.11.1"),
            (7, "a prototypical implementation of the client software written "
                "in the Go programming language"),
        )),
    Source(
        key="kemmoe2023leveraging", short="Kemmoe'23", venue="IEEE TDSC 2023",
        text_file="Leveraging Smart Contracts for Secure",
        testbed="Truffle local Ethereum instance on one Windows 10 PC "
                "(4.00 GHz Intel Core i7, 32 GB RAM)",
        implementation="Solidity 0.5.16 contract, Python 3.6.7 client "
                       "(Web3.py)",
        testbed_quotes=(
            (15, "The simulation was performed on a Windows 10 computer with a "
                 "4.00 GHz Intel core i7 and 32 GB of RAM."),
            (15, "We used Truffle Suite [34] to simulate a local instance of "
                 "the Ethereum blockchain"),
        )),
)}


# ==========================================================================
# 2. Anchors.
# ==========================================================================

METRICS = {
    "runtime_s": "s",           # end-to-end ADKG time as defined by the source
    "sent_per_node_MB": "MB",   # data sent by one node; MB as printed
    "gas": "gas",               # per transaction, EVM
    "proving_time_s": "s",      # zk-SNARK proof generation per participant
}


@dataclass(frozen=True)
class Anchor:
    id: str                  # stable identifier, e.g. "das23.t3.das23.rt.64"
    source: str              # key into SOURCES
    system: str              # protocol the number is ABOUT
    metric: str              # key into METRICS
    n: Optional[int]         # None = the source states it is constant in n
    value: float
    printed: str             # the value exactly as printed on the page
    degree: Optional[str]    # polynomial degree, e.g. "t", "2t"
    shares: Optional[str]    # shares needed to reconstruct, e.g. "2t+1"
    curve: Optional[str]
    operation: Optional[str]  # for gas / proving: which transaction
    page: int                # PDF page (1-based) of the extracted text
    locator: str             # table / figure / section on that page
    evidence: str            # D4 evidence ID
    quote: str               # verbatim fragment on `page`
    group: str               # comparability group (same testbed + method)
    kind: str                # "table" | "text" | "figure-label"
    note: str = ""
    scale: float = 1.0       # value = printed x scale (kGas: 1e3, "million": 1e6)


# Comparability groups: series inside one group were measured on one testbed
# by one team with one method.  Nothing is comparable ACROSS groups without
# the caveats in the module docstring.
GROUPS: Dict[str, str] = {
    "das23-table3": "Das'23 Table 3: Das'22 vs Das'23, same testbed (AWS, "
                    "8 regions, Python+Rust)",
    "das22-text": "Das'22 Sec. VIII-C text, n = 64, same testbed as das23-table3",
    "zhang23-fig4a": "Zhang'23 Fig. 4a: Zhang'23 vs DYX+ (Das'22), Go, "
                     "4 regions, reconstruction threshold 2t+1",
    "h2ct-eval": "H2CT Sec. VI: H2CT, Das'22, Das'23 on the H2CT testbed",
    "ethdkg-gas": "EthDKG App. A gas, post-Istanbul, worst case n = 2t+1",
    "sober23-gas": "Sober'23 Fig. 1 gas, local Hardhat",
    "sober23-proving": "Sober'23 Fig. 2 proving time, EC2 8-core Xeon",
    "kemmoe23-gas": "Kemmoe'23 Sec. 7.3 gas (group key exchange, not a DKG; "
                    "order-of-magnitude reference only)",
}

_DAS23_T3_CAPTION = "Table 3: Evaluation of high-threshold ADKG schemes."
_DAS23_T3_BW = ("Bandwidth usage is the amount of data a node sends during "
                "the ADKG protocol.")


def _series(prefix: str, source: str, system: str, metric: str,
            ns: Sequence[int], printed: Sequence[str], **common) -> List[Anchor]:
    """One anchor per reported cell.  A printed "—" (not run / not reported)
    produces NO anchor: missing cells are never filled."""
    out = []
    for n, s in zip(ns, printed, strict=True):
        if s in ("—", "-", ""):
            continue
        out.append(Anchor(id=f"{prefix}.{n}", source=source, system=system,
                          metric=metric, n=n, value=float(s), printed=s,
                          **common))
    return out


_N_DAS = (16, 32, 64, 128)
_N_ZHANG = (4, 13, 25, 49)

ANCHORS: Tuple[Anchor, ...] = tuple(
    # ---- E037: Das'23 Table 3 (PDF p.13), curve25519 columns -------------
    _series("das23.t3.das22-lt.rt", "das2023practical", "Das'22", "runtime_s",
            _N_DAS, ("1.10", "2.96", "9.56", "39.56"),
            degree="t", shares="t+1", curve="curve25519", operation=None,
            page=13, locator="Table 3, Das et al. [22], curve25519, l = t",
            evidence="E037", quote=_DAS23_T3_CAPTION, group="das23-table3",
            kind="table")
    + _series("das23.t3.das22-ht.rt", "das2023practical", "Das'22",
              "runtime_s", _N_DAS, ("10.64", "37.76", "152.56", "—"),
              degree="2t", shares="2t+1", curve="curve25519", operation=None,
              page=13, locator="Table 3, Das et al. [22], curve25519, l = 2t",
              evidence="E037", quote=_DAS23_T3_CAPTION, group="das23-table3",
              kind="table", note="n = 128 not reported (printed as a dash)")
    + _series("das23.t3.das23.rt", "das2023practical", "Das'23", "runtime_s",
              _N_DAS, ("1.29", "3.19", "12.48", "46.55"),
              degree="2t", shares="2t+1", curve="curve25519", operation=None,
              page=13, locator="Table 3, This work, curve25519, l = 2t",
              evidence="E037", quote=_DAS23_T3_CAPTION, group="das23-table3",
              kind="table")
    + _series("das23.t3.das22-lt.bw", "das2023practical", "Das'22",
              "sent_per_node_MB", _N_DAS, ("0.17", "0.68", "2.78", "11.24"),
              degree="t", shares="t+1", curve="curve25519", operation=None,
              page=13, locator="Table 3, Das et al. [22], curve25519, l = t",
              evidence="E037", quote=_DAS23_T3_BW, group="das23-table3",
              kind="table")
    + _series("das23.t3.das22-ht.bw", "das2023practical", "Das'22",
              "sent_per_node_MB", _N_DAS, ("0.95", "4.20", "18.97", "—"),
              degree="2t", shares="2t+1", curve="curve25519", operation=None,
              page=13, locator="Table 3, Das et al. [22], curve25519, l = 2t",
              evidence="E037", quote=_DAS23_T3_BW, group="das23-table3",
              kind="table", note="n = 128 not reported (printed as a dash)")
    + _series("das23.t3.das23.bw", "das2023practical", "Das'23",
              "sent_per_node_MB", _N_DAS, ("0.20", "0.82", "3.32", "13.10"),
              degree="2t", shares="2t+1", curve="curve25519", operation=None,
              page=13, locator="Table 3, This work, curve25519, l = 2t",
              evidence="E037", quote=_DAS23_T3_BW, group="das23-table3",
              kind="table")
    # ---- E038: Das'22's own text, n = 64 (runtime p.12, bandwidth p.13) ---
    + [
        Anchor("das22.text.lt.rt.64", "das2022practical", "Das'22",
               "runtime_s", 64, 10.0, "10", "t", "t+1", None, None, 12,
               "Sec. VIII-C runtime paragraph (Figure 5)", "E038",
               "our ADKG protocol takes approximately 10 seconds for 64 nodes",
               "das22-text", "text",
               note="'approximately'; curve not stated in this sentence"),
        Anchor("das22.text.ht.rt.64", "das2022practical", "Das'22",
               "runtime_s", 64, 160.0, "160", "2t", "2t+1", None, None, 12,
               "Sec. VIII-C runtime paragraph (Figure 5)", "E038",
               "160 seconds for 64 nodes", "das22-text", "text",
               note="curve not stated in this sentence"),
        Anchor("das22.text.lt.bw.64", "das2022practical", "Das'22",
               "sent_per_node_MB", 64, 2.96, "2.96", "t", "t+1", None, None,
               13, "Sec. VIII-C bandwidth paragraph (Figure 4)", "E038",
               "each node only sends 2.96 Megabytes", "das22-text", "text",
               note="Differs from 2.78 MB for the same configuration in "
                    "Das'23 Table 3; report the source used"),
        Anchor("das22.text.ht.bw.64", "das2022practical", "Das'22",
               "sent_per_node_MB", 64, 19.2, "19.2", "2t", "2t+1", None, None,
               13, "Sec. VIII-C bandwidth paragraph (Figure 4)", "E038",
               "each node sends 19.2 Megabytes", "das22-text", "text",
               note="Das'23 Table 3 prints 18.97 and its text 18.9 for this "
                    "configuration"),
    ]
    # ---- E040: Zhang'23 Fig. 4a (PDF p.12), threshold 2t+1 ---------------
    + _series("zhang23.f4a.zhang23.rt", "zhang2023practical", "Zhang'23",
              "runtime_s", _N_ZHANG, ("0.34", "1.88", "9.50", "61.76"),
              degree="2t", shares="2t+1", curve=None, operation=None, page=12,
              locator="Fig. 4a (data labels), our ADKG", evidence="E040",
              quote="(a) Average latency for different network sizes.",
              group="zhang23-fig4a", kind="figure-label",
              note="Series assignment of the labels is corroborated by the "
                   "Fig. 4b totals and the stated 2x-4.6x ratio (self-test)")
    + _series("zhang23.f4a.dyx.rt", "zhang2023practical", "Das'22 (DYX+)",
              "runtime_s", _N_ZHANG, ("1.56", "8.65", "28.75", "124.51"),
              degree="2t", shares="2t+1", curve=None, operation=None, page=12,
              locator="Fig. 4a (data labels), DYX+ ADKG", evidence="E040",
              quote="(a) Average latency for different network sizes.",
              group="zhang23-fig4a", kind="figure-label",
              note="DYX+ as run by Zhang et al. on their testbed")
    # ---- E078: H2CT (PDF p.12) ------------------------------------------
    + [
        Anchor("h2ct.text.h2ct.rt.1024.c25519", "xiang2024h2ct", "H2CT",
               "runtime_s", 1024, 1540.47, "1540.47", "2t", "2t+1",
               "curve25519", None, 12, "Sec. VI text on Fig. 6", "E078",
               "For n = 1024, our H2 CT takes approximately 1540.47 s and "
               "1655.94 s for curve25519 and bls12381, respectively.",
               "h2ct-eval", "text"),
        Anchor("h2ct.text.h2ct.rt.1024.bls", "xiang2024h2ct", "H2CT",
               "runtime_s", 1024, 1655.94, "1655.94", "2t", "2t+1",
               "bls12381", None, 12, "Sec. VI text on Fig. 6", "E078",
               "For n = 1024, our H2 CT takes approximately 1540.47 s and "
               "1655.94 s for curve25519 and bls12381, respectively.",
               "h2ct-eval", "text"),
    ]
    + [
        Anchor(f"h2ct.t3.{tag}.bw.128", "xiang2024h2ct", system,
               "sent_per_node_MB", 128, float(s), s, "2t", "2t+1",
               "curve25519", None, 12, "Sec. VI text on Table III", "E078",
               "each node in [10] and [11] sends 77.03 Megabytes and 13.55 "
               "Megabytes of messages, respectively; whereas our H2 CT "
               "protocol only sends 11.41 Megabytes",
               "h2ct-eval", "text",
               note="[10] = Das'22, [11] = Das'23 in the H2CT reference list")
        for tag, system, s in (("das22", "Das'22", "77.03"),
                               ("das23", "Das'23", "13.55"),
                               ("h2ct", "H2CT", "11.41"))
    ]
    # ---- E042: EthDKG App. A (PDF p.11) ----------------------------------
    + [
        Anchor(f"ethdkg.gas.{op}", "schindler2019ethdkg", "EthDKG", "gas",
               None, float(v), printed, None, "t+1", "BN254 (alt_bn128)", op,
               11, "App. A, Computational costs", "E042",
               "gas costs for contract deployment (2 551 221), for "
               "registration (106 385), and (recovered) key share submission "
               "(222 510) do not depend on the number of participants",
               "ethdkg-gas", "text", note="stated to be independent of n")
        for op, v, printed in (("deployment", 2551221, "2 551 221"),
                               ("registration", 106385, "106 385"),
                               ("key_share_submission", 222510, "222 510"))
    ]
    + [
        Anchor("ethdkg.gas.dispute.256", "schindler2019ethdkg", "EthDKG",
               "gas", 256, 1.5e6, "1.5", None, "t+1", "BN254 (alt_bn128)",
               "dispute", 11, "App. A, Computational costs", "E042",
               "a dispute consumes approximately 1.5 million gas",
               "ethdkg-gas", "text", note="'approximately'; printed as 1.5 million",
               scale=1e6),
        Anchor("ethdkg.gas.master_key.256", "schindler2019ethdkg", "EthDKG",
               "gas", 256, 1.7e6, "1.7", None, "t+1", "BN254 (alt_bn128)",
               "master_key_submission", 11, "App. A, Computational costs",
               "E042", "the master key submission requires around 1.7 million "
               "gas", "ethdkg-gas", "text",
               note="'around'; printed as 1.7 million", scale=1e6),
    ]
    # ---- E043: Sober'23 Fig. 1 text (PDF p.7) -----------------------------
    + [
        Anchor(f"sober23.gas.{op}.{n}", "sober2023distributed", "Sober'23",
               "gas", n, float(v) * 1000.0, str(v), None, "ceil((n+1)/2)",
               "Baby Jubjub (Groth16 over BN254)", op, 7,
               "Sec. 5.1 Costs, text on Fig. 1", "E043", quote, "sober23-gas",
               "text", note="printed in kGas", scale=1e3)
        for op, n, v, quote in (
            ("share", 4, 117, "The gas consumption of the share function "
             "averages 117 kGas for 4 participants reaching 505 kGas for 256 "
             "participants."),
            ("share", 256, 505, "The gas consumption of the share function "
             "averages 117 kGas for 4 participants reaching 505 kGas for 256 "
             "participants."),
            ("dispute", 4, 137, "Here we get 137 kGas for 4 participants and "
             "269 kGas for 256 participants."),
            ("dispute", 256, 269, "Here we get 137 kGas for 4 participants and "
             "269 kGas for 256 participants."),
            ("derive", 4, 290, "executing the derive function consumes 290 "
             "kGas for 4 participants up to 932 kGas for 256 participants."),
            ("derive", 256, 932, "executing the derive function consumes 290 "
             "kGas for 4 participants up to 932 kGas for 256 participants."),
        )
    ]
    + [
        Anchor("sober23.gas.justify", "sober2023distributed", "Sober'23",
               "gas", None, 343000.0, "343", None, "ceil((n+1)/2)",
               "Baby Jubjub (Groth16 over BN254)", "justify", 7,
               "Sec. 5.1 Costs, text on Fig. 1", "E043",
               "the gas consumption of the justify function is constant at "
               "343 kGas", "sober23-gas", "text",
               note="stated constant in n; printed in kGas", scale=1e3),
    ]
    # ---- E044: Sober'23 Fig. 2 text (PDF p.8) -----------------------------
    + [
        Anchor(f"sober23.prove.{op}.{n}", "sober2023distributed", "Sober'23",
               "proving_time_s", n, float(v), str(v), None, "ceil((n+1)/2)",
               "Baby Jubjub (Groth16 over BN254)", op, 8,
               "Sec. 5.2 Performance, text on Fig. 2", "E044",
               quote, "sober23-proving", "text")
        for op, n, v, quote in (
            ("justify", 4, 17, "The generation of the proofs with 4 "
             "participants takes 17 and 6 seconds, respectively."),
            ("derive", 4, 6, "The generation of the proofs with 4 "
             "participants takes 17 and 6 seconds, respectively."),
            ("justify", 256, 292, "it already takes 292 and 451 seconds with "
             "256 participants."),
            ("derive", 256, 451, "it already takes 292 and 451 seconds with "
             "256 participants."),
        )
    ]
    # ---- E103: Kemmoe'23 Sec. 7.3 (PDF p.15), 100 members ------------------
    + [
        Anchor(f"kemmoe23.gas.{op}.100", "kemmoe2023leveraging", "Kemmoe'23",
               "gas", 100, float(v), str(v), None, None,
               "BN curve + secp256k1 keys", op, 15,
               "Sec. 7.3, lite GKSC with distributed storage", "E103", quote,
               "kemmoe23-gas", "text",
               note="group key exchange contract, not a DKG")
        for op, v, quote in (
            ("CreateGrp", 1685263, "the group administrator pays approximately "
             "1685263 gas for the execution of CreateGrp"),
            ("AddMbr", 772963, "772963 gas for the execution of AddMbr"),
            ("UpdGKey", 764771, "a group member pays 764771 gas"),
        )
    ]
)

_ANCHOR_BY_ID: Dict[str, Anchor] = {a.id: a for a in ANCHORS}

# Zhang'23 Fig. 4b (same page): per-n totals of the latency breakdown of
# "our ADKG".  Used only to corroborate which Fig. 4a labels belong to which
# series; verified on the page like the anchors.
_ZHANG_FIG4B: Dict[int, str] = {4: "0.345", 13: "1.884", 25: "9.506",
                                49: "61.767"}


# ==========================================================================
# 3. Structural checks for tables (cell-to-series mapping).
# ==========================================================================


@dataclass(frozen=True)
class GridCheck:
    """A table printed row-major in the extracted text.  `start` locates the
    first row; the anchors listed under `columns` must equal the given column
    of the parsed grid exactly, cell by cell (dash cells included)."""
    source: str
    page: int
    start: Tuple[str, ...]
    ncols: int
    nrows: int
    columns: Dict[int, str]      # column index -> anchor-id prefix
    ns: Tuple[int, ...] = _N_DAS  # row labels


GRID_CHECKS: Tuple[GridCheck, ...] = (
    # Das'23 Table 3 runtime, Das et al. [22]: columns curve25519/t,
    # bls12381/t, curve25519/2t, bls12381/2t; rows n = 16, 32, 64, 128.
    GridCheck("das2023practical", 13, ("1.10", "1.32", "10.64", "11.50"), 4, 4,
              {0: "das23.t3.das22-lt.rt", 2: "das23.t3.das22-ht.rt"}),
    GridCheck("das2023practical", 13, ("0.17", "0.17", "0.95", "0.97"), 4, 4,
              {0: "das23.t3.das22-lt.bw", 2: "das23.t3.das22-ht.bw"}),
    # Das'23 Table 3, This work: columns curve25519/2t, bls12381/2t.
    GridCheck("das2023practical", 13, ("1.29", "1.47"), 2, 4,
              {0: "das23.t3.das23.rt"}),
    GridCheck("das2023practical", 13, ("0.20", "0.21"), 2, 4,
              {0: "das23.t3.das23.bw"}),
)


# ==========================================================================
# 4. Verification against the extracted PDF texts.
# ==========================================================================


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def load_pages(text_dir: str, stem: str) -> Dict[int, str]:
    path = os.path.join(text_dir, stem + ".txt")
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    parts = re.split(r"===== PDF PAGE (\d+) =====", raw)
    return {int(parts[i]): _norm(parts[i + 1]) for i in range(1, len(parts), 2)}


def _contains_token(page: str, printed: str) -> bool:
    # the printed value must appear as a whole token (not inside 12.96 etc.)
    pat = r"(?<![\d.])" + re.escape(printed) + r"(?![\d])"
    return re.search(pat, page) is not None


def verify_against_texts(text_dir: str) -> List[str]:
    """Return a list of failures (empty = every anchor verified).

    For each anchor: the quote occurs verbatim on the stated page, and the
    printed value occurs on that page as a whole token (for text anchors,
    inside the quoted sentence itself).  For each source: the
    testbed quotes occur on their pages.  For each GridCheck: the table cells
    parsed from the page equal the anchors exactly."""
    failures: List[str] = []
    cache: Dict[str, Dict[int, str]] = {}

    def pages(src_key: str) -> Dict[int, str]:
        if src_key not in cache:
            cache[src_key] = load_pages(text_dir, SOURCES[src_key].text_file)
        return cache[src_key]

    for a in ANCHORS:
        pg = pages(a.source).get(a.page, "")
        if _norm(a.quote) not in pg:
            failures.append(f"{a.id}: quote not on p.{a.page}")
        if not _contains_token(pg, a.printed):
            failures.append(f"{a.id}: value {a.printed!r} not on p.{a.page}")
        if a.kind == "text" and not _contains_token(_norm(a.quote), a.printed):
            failures.append(f"{a.id}: value {a.printed!r} not inside its quote")
    zp = pages("zhang2023practical").get(12, "")
    for n, printed in _ZHANG_FIG4B.items():
        if not _contains_token(zp, printed) or f"n = {n}" not in zp:
            failures.append(f"Zhang'23 Fig. 4b total {printed} (n={n}) not on p.12")
    for s in SOURCES.values():
        for page, q in s.testbed_quotes:
            if _norm(q) not in pages(s.key).get(page, ""):
                failures.append(f"{s.key}: testbed quote not on p.{page}: {q[:40]}")
    for g in GRID_CHECKS:
        toks = pages(g.source).get(g.page, "").split(" ")
        starts = [i for i in range(len(toks) - len(g.start) + 1)
                  if tuple(toks[i:i + len(g.start)]) == g.start]
        if len(starts) != 1:
            failures.append(f"grid {g.start}: found {len(starts)} times")
            continue
        cells = toks[starts[0]:starts[0] + g.ncols * g.nrows]
        grid = [cells[r * g.ncols:(r + 1) * g.ncols] for r in range(g.nrows)]
        for col, prefix in g.columns.items():
            series = [a for a in ANCHORS if a.id.startswith(prefix + ".")]
            by_n = {a.n: a.printed for a in series}
            if not series:
                failures.append(f"grid {prefix}: no anchors")
            for r, n in enumerate(g.ns):
                printed = grid[r][col]
                if printed == "—":
                    if n in by_n:
                        failures.append(f"grid {prefix}: n={n} is a dash in "
                                        "the table but has an anchor")
                elif by_n.get(n) != printed:
                    failures.append(f"grid {prefix}: n={n} table {printed!r} "
                                    f"vs anchor {by_n.get(n)!r}")
    return failures


def check_evidence_ids(d4_path: str) -> List[str]:
    """Every evidence ID cited here must exist in the D4 evidence ledger."""
    with open(d4_path, encoding="utf-8") as fh:
        ledger = fh.read()
    cited = {a.evidence for a in ANCHORS}
    for row in TABLE_I:
        for ids in row.evidence.values():
            cited.update(i for i in ids if i.startswith("E"))
    return [f"{e} not in D4" for e in sorted(cited)
            if f"**{e}**" not in ledger]


# ==========================================================================
# 5. Query helpers for Table II (exact values only).
# ==========================================================================


def anchors(system: Optional[str] = None, metric: Optional[str] = None,
            n: Optional[int] = None, group: Optional[str] = None,
            source: Optional[str] = None) -> List[Anchor]:
    """Exact filter over ANCHORS.  No interpolation: an n that the source does
    not report returns nothing."""
    return [a for a in ANCHORS
            if (system is None or a.system == system)
            and (metric is None or a.metric == metric)
            and (n is None or a.n == n)
            and (group is None or a.group == group)
            and (source is None or a.source == source)]


def matching_t(n: int) -> Optional[int]:
    """t such that DA-ADKG's n = 3t+1 equals n, else None."""
    return (n - 1) // 3 if n >= 4 and (n - 1) % 3 == 0 else None


def mb_both(nbytes: float) -> Tuple[float, float]:
    """(decimal MB, binary MiB).  The sources do not state which they use."""
    return nbytes / 1e6, nbytes / 2 ** 20


def table_ii_anchor_rows(metrics: Iterable[str] = ("runtime_s",
                                                  "sent_per_node_MB", "gas"),
                         ) -> List[Dict[str, object]]:
    """Rows for the published-numbers block of Table II, labelled "as
    reported".  DA-ADKG rows are NOT produced here (experiments.py)."""
    wanted = set(metrics)
    unknown = wanted - set(METRICS)
    if unknown:
        raise ValueError(f"unknown metric(s): {sorted(unknown)}")
    rows = []
    for a in ANCHORS:
        if a.metric not in wanted:
            continue
        s = SOURCES[a.source]
        rows.append({
            "system": a.system, "reported_in": s.short, "metric": a.metric,
            "unit": METRICS[a.metric], "n": a.n, "value": a.value,
            "printed": a.printed, "degree": a.degree, "shares": a.shares,
            "curve": a.curve, "operation": a.operation,
            "testbed": s.testbed, "implementation": s.implementation,
            "group": a.group, "locator": f"{s.short}, p.{a.page}, {a.locator}",
            "evidence": a.evidence, "note": a.note,
        })
    return rows


# ==========================================================================
# 6. Table I (qualitative), every cell with evidence IDs.
# ==========================================================================

TABLE_I_COLUMNS = (
    "network", "resilience", "shares", "comm_per_node", "comm_total",
    "rounds", "setup", "adversary", "proof", "onchain_disputes", "penalty",
    "reporter_incentive", "external_verifiability", "trusted_setup",
    "dependability_model", "implementation",
)


@dataclass(frozen=True)
class TableIRow:
    protocol: str
    key: str                          # BibTeX key, or "this-work"
    cells: Dict[str, str]
    evidence: Dict[str, Tuple[str, ...]] = field(default_factory=dict)


def _row(protocol: str, key: str, **spec: Tuple[str, Tuple[str, ...]]
         ) -> TableIRow:
    cells = {c: spec[c][0] for c in TABLE_I_COLUMNS}
    ev = {c: spec[c][1] for c in TABLE_I_COLUMNS}
    return TableIRow(protocol, key, cells, ev)


# Evidence codes: "E###" = D4 ledger entry; "D3:<matrix>" = D3 comparison
# matrix (which itself cites D2 locators); "code:<file>" = the v2 artifact.
_NONE_ACC = ("No", ("D3:M3",))
TABLE_I: Tuple[TableIRow, ...] = (
    _row("Abraham'21", "abraham2021reaching",
         network=("Asynchronous", ("E006",)),
         resilience=("f < n/3", ("E006",)),
         shares=("f+1", ("D3:M1",)),
         comm_per_node=("Not stated", ("D3:M1",)),
         comm_total=("Õ(κn³) (SoK: λn³ log n)", ("E006", "E048")),
         rounds=("O(1) expected", ("E006",)),
         setup=("PKI (Das'23 lists SXDH, RO & PKI)", ("E006", "E008")),
         adversary=("Static", ("E007",)),
         proof=("Definitional; preservation deferred", ("D3:M2",)),
         onchain_disputes=_NONE_ACC, penalty=_NONE_ACC,
         reporter_incentive=_NONE_ACC, external_verifiability=_NONE_ACC,
         trusted_setup=("No", ("E006",)),
         dependability_model=("None", ("D3:M3c",)),
         implementation=("None", ("D3:M5",))),
    _row("Das'22", "das2022practical",
         network=("Asynchronous", ("E001",)),
         resilience=("t < n/3", ("E001",)),
         shares=("ℓ ∈ [t+1, n−t]", ("E005",)),
         comm_per_node=("O(κn²)", ("E002",)),
         comm_total=("O(κn³)", ("E001", "E003")),
         rounds=("O(log n) expected", ("E003",)),
         setup=("RO + PKI", ("E002",)),
         adversary=("Static", ("E004",)),
         proof=("Theorem 1", ("E003",)),
         onchain_disputes=_NONE_ACC, penalty=_NONE_ACC,
         reporter_incentive=_NONE_ACC, external_verifiability=_NONE_ACC,
         trusted_setup=("No", ("E002",)),
         dependability_model=("None", ("D3:M3c",)),
         implementation=("Python + Rust (public)", ("D3:M5",))),
    _row("Zhang'23", "zhang2023practical",
         network=("Asynchronous", ("E010",)),
         resilience=("n = 3t+1", ("E010",)),
         shares=("t+1 or 2t+1", ("D3:M1",)),
         comm_per_node=("Not stated", ("D3:M1",)),
         comm_total=("O(λn⁴) own HACSS; O(λn³) with DYX+ HACSS "
                     "(SoK lists λn³)", ("E009", "E057", "E048")),
         rounds=("O(1)", ("E009",)),
         setup=("None (standard model, DL)", ("E010",)),
         adversary=("Not stated", ("E011",)),
         proof=("Theorems 1–6", ("D3:M2",)),
         onchain_disputes=_NONE_ACC, penalty=_NONE_ACC,
         reporter_incentive=_NONE_ACC, external_verifiability=_NONE_ACC,
         trusted_setup=("No", ("E010",)),
         dependability_model=("None", ("D3:M3c",)),
         implementation=("Go (ADKG code not given)", ("D3:M5",))),
    _row("Das'23", "das2023practical",
         network=("Asynchronous", ("E012",)),
         resilience=("n ≥ 3t+1", ("E012",)),
         shares=("ℓ+1, ℓ ∈ [t, n−t−1]", ("E012",)),
         comm_per_node=("O(κn²)", ("D3:M1",)),
         comm_total=("O(κn³)", ("E012", "E013")),
         rounds=("O(log n) expected; O(1) common case", ("E013", "D3:M1")),
         setup=("RO + PKI", ("E014", "D3:M2")),
         adversary=("Static (+ ℓ−t leaked shares)", ("E013", "E014")),
         proof=("F_ADKG + Theorem 1", ("E025", "E026")),
         onchain_disputes=_NONE_ACC, penalty=_NONE_ACC,
         reporter_incentive=_NONE_ACC, external_verifiability=_NONE_ACC,
         trusted_setup=("No", ("E014",)),
         dependability_model=("None", ("D3:M3c",)),
         implementation=("Python + Rust (public)", ("D3:M5",))),
    _row("H²CT'24", "xiang2024h2ct",
         network=("Asynchronous", ("E076",)),
         resilience=("t < n/3", ("D3:M1b",)),
         shares=("ℓ+1, ℓ ∈ [t+1, 2t]", ("D3:M1b",)),
         comm_per_node=("Not stated", ("D3:M1b",)),
         comm_total=("O(λn³)", ("E076",)),
         rounds=("O(log n)", ("E076",)),
         setup=("PKI + CRS", ("E076", "D3:M1b")),
         adversary=("Static (+ ℓ−t leaked shares)", ("E076",)),
         proof=("Theorem 1 (DDH)", ("D3:M2b",)),
         onchain_disputes=("No", ("E080",)), penalty=("No", ("E080",)),
         reporter_incentive=("No", ("E080",)),
         external_verifiability=("pvACSS transcripts", ("D3:M3b",)),
         trusted_setup=("No", ("D3:M1b",)),
         dependability_model=("None", ("E080",)),
         implementation=("Python (no H²CT repository given)", ("E077",))),
    _row("Abraham'25", "abraham2025nearly",
         network=("Asynchronous", ("E055",)),
         resilience=("f < n/3", ("E055",)),
         shares=("Low threshold", ("D3:M1b",)),
         comm_per_node=("Not stated", ("D3:M1b",)),
         comm_total=("O(λ²n^{2+1/k}) expected", ("E056",)),
         rounds=("O(k) expected", ("E056",)),
         setup=("Plain PKI + RO + erasures", ("E055",)),
         adversary=("Strongly adaptive", ("E055",)),
         proof=("Theorem 7 (OMDL)", ("E056",)),
         onchain_disputes=("No", ("D3:M3b",)), penalty=("No", ("D3:M3b",)),
         reporter_incentive=("No", ("D3:M3b",)),
         external_verifiability=("APVSS transcripts", ("D3:M3b",)),
         trusted_setup=("No", ("E055",)),
         dependability_model=("None", ("D3:M3c",)),
         implementation=("None", ("E058",))),
    _row("Feng–Tang'25", "feng2025asymptotically",
         network=("Asynchronous", ("E059",)),
         resilience=("f < n/3", ("D3:M1b",)),
         shares=("Low threshold", ("D3:M1b",)),
         comm_per_node=("Not stated", ("D3:M1b",)),
         comm_total=("O(κλn²) (Õ(n²) writes, Õ(n³) reads)", ("E060", "E054")),
         rounds=("O(1)", ("E060",)),
         setup=("Silent setup: CRS + PKI (bulletin board)", ("E059", "E054")),
         adversary=("Adaptive", ("E059",)),
         proof=("Theorem 4 (coin); ADKG via [32]", ("E062", "E060")),
         onchain_disputes=("No", ("D3:M3b",)), penalty=("No", ("D3:M3b",)),
         reporter_incentive=("No", ("D3:M3b",)),
         external_verifiability=("Not stated", ("D3:M3b",)),
         trusted_setup=("CRS", ("D3:M1b",)),
         dependability_model=("None", ("D3:M3c",)),
         implementation=("None", ("E063",))),
    _row("EthDKG", "schindler2019ethdkg",
         network=("Synchronous (block-height phases)", ("E015",)),
         resilience=("t < n/2", ("E016",)),
         shares=("t+1", ("D3:M1",)),
         comm_per_node=("Not stated", ("D3:M1",)),
         comm_total=("O(n³) network; O(n²) on-chain", ("E017",)),
         rounds=("Fixed number of phases", ("E015",)),
         setup=("Ethereum chain + keys", ("D3:M2",)),
         adversary=("Adaptive (claimed)", ("E016",)),
         proof=("Deferred to Gennaro et al. and Neji et al.", ("D3:M2",)),
         onchain_disputes=("Yes: contract decrypts disclosed key + DLEQ",
                           ("E029",)),
         penalty=("Optional", ("E032",)),
         reporter_incentive=("Fee refund", ("E032",)),
         external_verifiability=("Yes", ("D3:M3",)),
         trusted_setup=("No", ("D3:M3",)),
         dependability_model=("None", ("D3:M3c",)),
         implementation=("Solidity + Python (public)", ("D3:M5",))),
    _row("Sober'23", "sober2023distributed",
         network=("Synchronous (Δ)", ("E018",)),
         resilience=("Not stated", ("D3:M1",)),
         shares=("⌈(n+1)/2⌉", ("E020",)),
         comm_per_node=("Not stated", ("D3:M1",)),
         comm_total=("Not stated", ("D3:M1",)),
         rounds=("Fixed number of phases", ("D3:M1",)),
         setup=("Chain + Groth16 trusted setup", ("E019",)),
         adversary=("Not stated", ("D3:M2",)),
         proof=("None", ("D3:M2",)),
         onchain_disputes=("Yes: dealer zk-justifies", ("E035",)),
         penalty=("Slash collateral", ("E034", "E035")),
         reporter_incentive=("Not stated", ("D3:M3",)),
         external_verifiability=("Yes (stated aim)", ("E034",)),
         trusted_setup=("Yes", ("E019",)),
         dependability_model=("None", ("D3:M3c",)),
         implementation=("Solidity + Go + ZoKrates (public)", ("D3:M5",))),
    # DA-ADKG as implemented in code_v2 (NOT the submitted text).  Cells that
    # depend on open paper decisions say so.
    _row("DA-ADKG (this work, revised)", "this-work",
         network=("Asynchronous P2P dissemination; QUAL and key fixed by "
                  "ledger windows (partial synchrony to the chain)",
                  ("code:network.py", "D3:M1")),
         resilience=("n = 3t+1", ("code:protocol.py",)),
         shares=("2t+1 (degree 2t)", ("code:protocol.py",)),
         comm_per_node=("Θ(κn²) sent", ("code:network.py",)),
         comm_total=("Θ(κn³) P2P; Θ(n) ledger writes without faults",
                     ("code:network.py", "D3:M1")),
         rounds=("6 ledger windows; time set by finality and Δ",
                 ("code:protocol.py", "code:network.py")),
         setup=("On-chain registration (PKI) + RO", ("code:protocol.py",)),
         adversary=("Static", ("code:protocol.py", "D3:M2")),
         proof=("Theorems for inclusion, exclusion, non-frameability, key "
                "correctness; secrecy sketch", ("code:protocol.py",)),
         onchain_disputes=("Yes: disclosed DH key + DLEQ; complaint/answer "
                           "for omissions", ("code:protocol.py", "E029")),
         penalty=("α·S for provable faults (α = 1); omission → "
                  "exclusion only", ("code:protocol.py",)),
         reporter_incentive=("Share ρ of slash (ρ = 0.1)", ("code:protocol.py",)),
         external_verifiability=("Yes (on-chain record + g^s)",
                                 ("code:protocol.py",)),
         trusted_setup=("No", ("code:protocol.py",)),
         dependability_model=("Phase-type CTMC, validated on held-out runs",
                              ("code:experiments.py",)),
         implementation=("Python reference + EVM gas model (this artifact)",
                         ("code:protocol.py",))),
)


def table_i_rows() -> List[Dict[str, str]]:
    return [{"protocol": r.protocol, **r.cells} for r in TABLE_I]


# ==========================================================================
# 7. Self-tests.
# ==========================================================================


def _consistency_report() -> List[str]:
    """Cross-source and internal consistency checks [DERIVED].  Returns
    human-readable lines; asserts only what must hold exactly."""
    out = []
    v = {a.id: a.value for a in ANCHORS}

    # E041: DYX+/Zhang'23 ratios vs the stated "about 2x-4.6x".
    ratios = [v[f"zhang23.f4a.dyx.rt.{n}"] / v[f"zhang23.f4a.zhang23.rt.{n}"]
              for n in _N_ZHANG]
    assert 1.95 <= min(ratios) and max(ratios) <= 4.65, ratios
    out.append("E041 Zhang'23 Fig. 4a ratios DYX+/Zhang'23 = "
               + ", ".join(f"{r:.2f}" for r in ratios)
               + "  (paper: 'about 2x-4.6x')")

    # Fig. 4a labels are the Fig. 4b totals truncated to 2 decimals.
    for n, s in _ZHANG_FIG4B.items():
        assert f"{float(s):.3f}"[:-1] == _ANCHOR_BY_ID[
            f"zhang23.f4a.zhang23.rt.{n}"].printed, n
    out.append("Zhang'23 Fig. 4a labels = Fig. 4b totals truncated "
               "(0.345, 1.884, 9.506, 61.767): series assignment corroborated")

    # E079: Das'23 bytes per node at n = 128 on two independent testbeds.
    a, b = v["das23.t3.das23.bw.128"], v["h2ct.t3.das23.bw.128"]
    out.append(f"E079 Das'23 MB/node at n=128: {a} (Das'23) vs {b} (H2CT): "
               f"{100 * (b - a) / a:.1f}% apart")

    # Das'22 at n = 64 as reported by itself vs by Das'23.
    out.append("Das'22 n=64 MB/node, degree t: "
               f"{v['das22.text.lt.bw.64']} (own text) vs "
               f"{v['das23.t3.das22-lt.bw.64']} (Das'23 Table 3); degree 2t: "
               f"{v['das22.text.ht.bw.64']} vs {v['das23.t3.das22-ht.bw.64']}"
               " -> cite one source per number, do not mix")
    out.append("Das'22 n=64 runtime, degree t: "
               f"~{v['das22.text.lt.rt.64']:.0f} s (own) vs "
               f"{v['das23.t3.das22-lt.rt.64']} s (Das'23); degree 2t: "
               f"{v['das22.text.ht.rt.64']:.0f} s vs "
               f"{v['das23.t3.das22-ht.rt.64']} s")

    # Quadratic growth of bytes per node (Das'23 own claim, p.13).
    s = [v[f"das23.t3.das23.bw.{n}"] for n in _N_DAS]
    out.append("Das'23 MB/node growth per doubling of n: "
               + ", ".join(f"x{s[i + 1] / s[i]:.2f}" for i in range(3)))
    return out


def _selftest(text_dir: Optional[str], d4_path: Optional[str]) -> None:
    # 1. Schema and uniqueness.
    ids = [a.id for a in ANCHORS]
    assert len(ids) == len(set(ids)), "duplicate anchor id"
    for a in ANCHORS:
        assert a.source in SOURCES and a.metric in METRICS, a.id
        assert a.group in GROUPS and a.kind in ("table", "text",
                                                "figure-label"), a.id
        assert a.page >= 1 and a.evidence.startswith("E") and a.quote, a.id
        assert a.value > 0 and (a.n is None or a.n >= 4), a.id
        # value = printed form x stated scale (no rounding, no conversion)
        assert float(a.printed.replace(" ", "")) * a.scale == a.value, a.id
    print(f"  PASS  {len(ANCHORS)} anchors from {len(SOURCES)} sources; schema, "
          "unique ids, printed form = value")

    # 2. Missing cells are not filled; no interpolation.
    got = anchors(system="Das'22", metric="runtime_s", n=128,
                  group="das23-table3")
    assert [a.degree for a in got] == ["t"]          # the 2t cell is a dash
    assert anchors(system="Das'23", n=100) == []      # not reported -> nothing
    assert len(anchors(group="das23-table3")) == 22   # 24 cells - 2 dashes
    print("  PASS  dash cells produce no anchor; queries are exact-match only")

    # 3. Every anchor verified against its PDF page.
    if text_dir and os.path.isdir(text_dir):
        failures = verify_against_texts(text_dir)
        for f in failures:
            print("   FAIL", f)
        assert not failures
        nq = sum(len(s.testbed_quotes) for s in SOURCES.values())
        print(f"  PASS  every anchor's quote and printed value found on its "
              f"page; {nq} testbed quotes found; {len(GRID_CHECKS)} table "
              "grids match cell by cell")
    else:
        print("  SKIP  text verification (no TEXT_DIR given)")

    # 4. Table I: complete, cited, and free of the v1 errors.
    for r in TABLE_I:
        assert set(r.cells) == set(TABLE_I_COLUMNS), r.protocol
        for c in TABLE_I_COLUMNS:
            assert r.cells[c] and r.evidence[c], (r.protocol, c)
    rows = {r["protocol"]: r for r in table_i_rows()}
    assert "Wang'25" not in rows                                   # DEC-2
    assert rows["Das'22"]["rounds"] == "O(log n) expected"         # v1: O(1)
    assert rows["Sober'23"]["penalty"] == "Slash collateral"       # v1: "no"
    assert rows["EthDKG"]["network"].startswith("Synchronous")
    dadkg = rows["DA-ADKG (this work, revised)"]
    assert "Θ(κn³)" in dadkg["comm_total"]                         # F6
    assert "secrecy sketch" in dadkg["proof"]                       # F11 scope
    print(f"  PASS  Table I: {len(TABLE_I)} rows x {len(TABLE_I_COLUMNS)} "
          "columns, every cell cited; v1 errors absent (Wang'25, Das'22 "
          "rounds, Sober'23 penalty)")

    if d4_path and os.path.isfile(d4_path):
        missing = check_evidence_ids(d4_path)
        assert not missing, missing
        print("  PASS  every cited evidence ID exists in the D4 ledger")
    else:
        print("  SKIP  evidence-ID check (no D4 ledger path given)")

    # 5. Helpers.
    assert matching_t(16) == 5 and matching_t(64) == 21 and matching_t(49) == 16
    assert matching_t(32) is None and matching_t(128) is None
    assert mb_both(2 ** 20) == (1.048576, 1.0)
    rows2 = table_ii_anchor_rows()
    assert all(r["testbed"] and r["locator"] for r in rows2)
    print("  PASS  matching_t (16, 64, 49 representable; 32, 128 not), "
          "mb_both, table_ii_anchor_rows")

    # 6. This module must not depend on the simulator (it holds no model).
    import ast
    with open(os.path.abspath(__file__), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    imported = {a.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for a in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert not imported & {"protocol", "network", "crypto", "random"}, imported
    print("  PASS  data-only module: no protocol/network import, no model")

    print("\n  Consistency notes [DERIVED]:")
    for line in _consistency_report():
        print("   -", line)
    print("\nbaselines.py: all self-tests passed.")


if __name__ == "__main__":
    _selftest(sys.argv[1] if len(sys.argv) > 1 else None,
              sys.argv[2] if len(sys.argv) > 2 else None)
