# DA-ADKG: Code, Results and Reproduction Scripts

This repository accompanies the article *"DA-ADKG: Dependable Asynchronous Distributed Key Generation with Blockchain-Enforced Accountability"* (IEEE Transactions on Dependable and Secure Computing, manuscript TDSC-2025-08-1249, revised version).

It contains everything needed to check the paper's experimental claims:

- the reference implementation of the protocol and of its smart-contract logic;
- the ledger-driven asynchronous simulator and the experiment harness;
- the results file from which **every** number, table and figure of the evaluation was generated;
- the scripts that regenerate those tables, the figure and the in-text numbers.

A reader can verify the published results in about five minutes without re-running anything (Section 3), or repeat the full measurement in about two hours on a two-core machine (Section 4).

---

## 1. Repository layout

```
DA-ADKG/
├── README.md                  this file
├── code/
│   ├── crypto.py              BN254 G1, Pedersen commitments, Schnorr and polynomial PoKs,
│   │                          Chaum-Pedersen DLEQ, Merkle tree, Shamir/Lagrange, ECIES,
│   │                          DH-masked shares, ECDSA secp256k1 (ecrecover form), Keccak-256
│   ├── protocol.py            participants and smart-contract state machines (six windows,
│   │                          complaint/answer QUAL rule, disputes, slashing, PoK fraud proofs,
│   │                          key finalization), EVM gas schedule, synchronous reference driver
│   ├── network.py             discrete-event simulator: blocks, finality, latency, uplink queues,
│   │                          partitions, adversarial scheduling, crashes; window planning;
│   │                          validated honest-case estimator for large n
│   ├── baselines.py           58 published reference numbers from 7 papers, each with page,
│   │                          testbed and verbatim quote; rows of Table I with per-cell sources
│   ├── experiments.py         executed scenarios and invariants, large-n performance, scaling fits,
│   │                          dependability model (phase-type CTMC, fault injection, held-out
│   │                          validation), provenance, resumable runs
│   ├── figures.py             Fig. 2 and the generated LaTeX tables, from a results file only
│   ├── integration_test.py    23 cross-file checks and the freshness check of a results file
│   └── requirements.txt
├── results/
│   ├── results_paper.json     the results of the paper run (2,080 executed runs + dependability)
│   ├── run.log                console log of that run
│   └── paper_outputs/         figure1.pdf/.png (Fig. 2 of the paper), table_*.tex, manifest.json
├── paper/
│   ├── make_macros.py         turns the results file into LaTeX macros for the in-text numbers
│   └── results_macros.tex     the macros used in the paper
└── docs/
    ├── CHANGELOG.md           changes from the submitted version to this one
    └── VERIFICATION.md        record of the end-to-end verification pass
```

---

## 2. Requirements

- Python 3.10 or later. The paper run used Python 3.13.
- Install the packages:

  ```
  pip install -r code/requirements.txt
  ```

| Package | Purpose |
|---|---|
| `cryptography` | ECIES transport (HKDF + AES-256-GCM) |
| `numpy`, `scipy` | statistics, matrix exponential, regression fits |
| `matplotlib` | Fig. 2 |
| `pycryptodome` (optional) | faster Keccak-256; cross-checked against the pure-Python version |
| `py_ecc` (optional, tests only) | independent BN254 implementation used by the crypto self-tests |

`pdflatex` is optional. When it is present, `figures.py` also compiles every generated table as a check.

All cryptography that the protocol uses is implemented in `code/crypto.py` in pure Python. No external elliptic-curve library is needed to run the protocol.

---

## 3. Verify the published results without re-running (about 5 minutes)

Run all commands from `code/`.

**Step 1: the results file belongs to this code.**

```
python3 integration_test.py --results ../results/results_paper.json --results-only
```

Expected output: `FRESH (schema, unit costs and source hashes match)`.

The results file records the SHA-256 of every source file. A file produced by different or modified sources is reported as stale.

**Step 2: the tables and the figure follow from the results file.**

```
python3 figures.py ../results/results_paper.json /tmp/out
diff -r /tmp/out ../results/paper_outputs
```

The `.tex` tables are identical byte for byte. `figure1.pdf` differs only in its embedded creation date, and `manifest.json` records the output paths and file hashes, which change with them.

**Step 3: the numbers quoted in the text follow from the results file.**

```
cd ../paper
python3 make_macros.py ../results/results_paper.json | diff - results_macros.tex
```

Every experimental number in the paper's prose is one of these macros. None was typed by hand. For example, `\NwithinSucc` = 1,559 runs that produced a key, `\HeldPred` = 0.955 predicted success and `\MBnodeOneTwoSeven` = 1.62 MB sent per node at n = 127.

**Step 4 (optional): look inside.** `results_paper.json` holds per-scenario summaries, attack overheads, the large-n performance rows, scaling fits, gas per transaction, the dependability analysis of both chain profiles, and the provenance record:

- source hashes;
- unit costs and how they were obtained;
- CPU model, Python, numpy and scipy versions;
- network parameters.

---

## 4. Re-run everything

**Self-tests and cross-file checks** (about 8 minutes on two cores):

```
cd code
for f in crypto protocol network baselines experiments; do python3 $f.py; done
python3 figures.py
python3 integration_test.py
```

`integration_test.py` runs the 23 checks below.

| Group | What it checks |
|---|---|
| I1–I8 | interfaces between the modules |
| R1–R7 | regressions of defects found in the submitted version, including the participant-set rule (R7) |
| P5 | the finalized key equals g^s |
| A1–A6 | adversarial checks |
| B1 | published anchors verified against the cited PDFs (needs `--texts`) |

The adversarial checks cover:

- non-frameability under evidence mutation;
- forged answers;
- three Byzantine strategies at t = 3;
- key finalization needing t+1 matching votes;
- late disputes;
- a decoder fuzz with 400 malformed payloads.

Check B1, which verifies the published anchors against the PDF texts of the cited papers, is skipped unless `--texts DIR` points to a directory of extracted PDF texts. The PDFs are not redistributed here.

**The full paper measurement:**

```
python3 experiments.py --run paper --workers 2 --cache run_cache --out results.json
python3 integration_test.py --results results.json --results-only
python3 figures.py results.json out/
python3 ../paper/make_macros.py results.json > results_macros.tex
```

| Configuration | Content | Time on 2 cores (Intel Xeon @ 2.10 GHz) |
|---|---|---|
| `--run quick` | 3 runs per scenario, small dependability study | about 3 minutes |
| `--run paper` | t = 2 (n = 7, 100 runs per scenario) and t = 5 (n = 16, 30 runs) for 16 scenarios; performance for n = 4 … 1024 on two chain profiles; dependability at t = 2 with 60 injections per phase and 200 held-out runs | about 1.8 hours (4,902 s executed scenarios + 1,661 s dependability) |

**The run is resumable.** If it is interrupted, repeat the same command: finished jobs are read from `run_cache` and only missing jobs run.

**The cache refuses mixed versions.** A cache produced by different sources is rejected, so one run can never mix two versions of the code.

**What changes between machines:**

- *Timing numbers.* The unit costs (time per elliptic-curve operation) are measured once at the start of a run on the machine at hand. Completion times therefore scale with your CPU speed.
- *Outcomes.* Success, QUAL, slashing, gas, bytes and ledger data are identical on every machine for the same seeds.

---

## 5. Where each part of the paper comes from

| Paper element | Produced by | File |
|---|---|---|
| Table I (protocol properties) | `baselines.TABLE_I` → `figures.py` | `table_I_properties.tex` |
| Table II (ledger windows), Fig. 1 (architecture), Algorithms 1–4 | written from `protocol.py` (`Window`, `SmartContract`, `Participant`) | — |
| Table III (gas per transaction) | `protocol.py` gas schedule → `experiments.gas_per_transaction` | `table_gas.tex` |
| Table IV (executed scenarios) | `experiments.py` executed runs | `table_scenarios.tex` |
| Table V (scalability) | `network.py` honest-case estimator, validated against the executed runs | `table_III_scalability.tex` |
| Table VI (comparison with published work) | `baselines.ANCHORS` (as reported, with testbeds) + our rows | `table_II_comparison.tex` |
| Fig. 2 (evaluation, panels a–i) | `figures.py` | `figure1.pdf` |
| Section VI (dependability) | `experiments.dependability_for_chain`, `PhaseTypeCTMC`, `validate_ctmc` | `results_paper.json` → `dependability` |
| Numbers in the text | `paper/make_macros.py` | `results_macros.tex` |

The generated table files keep the names they had during development. For example, `table_III_scalability.tex` is Table V of the published layout.

---

## 6. Model and assumptions used by the simulator

| Quantity | How it is obtained | Assumption |
|---|---|---|
| Success, QUAL, slashing, invariants | executed runs with real cryptography | windows planned by `experiments.plan_windows`; scenario labelled *within* or *beyond* the model's assumptions |
| Completion time | simulated: network + ledger + computation | latency log-normal (μ = 4.2, σ = 0.8 in log-ms, median ≈ 67 ms), 100 Mbit/s uplinks; Δ = 99.9 % latency quantile × safety factor 2; computation = counted operations × unit costs measured in pure Python on the stated CPU |
| Bytes | lengths of the real serialized messages | payload only, no TCP/TLS framing |
| Gas | EVM gas schedule (EIP-2028 calldata, EIP-2929 storage, EIP-1108 BN254 precompiles) | the contract logic is executed in Python and priced; it is not deployed on a chain |
| Chain profiles | `bft-1s`: 1 s blocks, 1-block finality; `ethereum-l1`: 12 s slots, 96-slot finality | Ethereum times are upper bounds under normal operation |
| Dependability | phase durations from honest runs; P(fail \| fault event in phase) by injection; held-out Poisson validation | fault event = t + 1 random parties cut off (ledger included) for the longest window; sensitivity at 0.5× and 2×; the fault rate is a sweep, not a claim |

Numbers of other protocols are quoted as published, with their testbeds (`baselines.py`). Wall-clock times measured on other testbeds are not comparable with our simulated completion times and are shown only for reference. Bytes sent per node are the most comparable quantity.

---

## 7. Integrity of the results

- **One run, no selection.** All DA-ADKG numbers come from one execution of `--run paper` on the code in this repository. No parameter was tuned to reproduce the numbers of the submitted version.
- **Every failure is reported.** Within the assumptions, 1 of 1,560 runs failed: a ledger transaction exceeded Δ, which is the residual risk of using a latency quantile as Δ. The paper reports it in Section VII.
- **Freshness is enforced.** The freshness check (Section 3) ties the results file to the exact source files. `docs/VERIFICATION.md` records the end-to-end pass behind the published results file.

---

## 8. Identifiers in code comments

Comments in the code refer to identifiers from our internal review of the submitted version:

- **F#** — findings;
- **R#** — reproduced defects, each with a regression check;
- **P#**, **N#** — change items;
- **D#** — documented deviations from the submitted manuscript (listed in the docstring of `protocol.py`);
- **DEC-#** — design decisions.

The review documents are not part of this repository. The docstrings state every rule that the code implements, and `docs/CHANGELOG.md` summarizes the changes.

## 9. Determinism

- **Fixed given the inputs.** Protocol outcomes, gas, bytes and simulated times are deterministic given the seed, the unit costs and the source code. This holds for any number of workers and any `PYTHONHASHSEED` (integration check I6).
- **Keys differ by design.** Cryptographic keys are drawn from the operating system's random generator, so they differ between runs. No outcome depends on their values.

---

## 10. Citation

If you use this artifact, please cite the article:

```
J. Gao et al., "DA-ADKG: Dependable Asynchronous Distributed Key Generation
with Blockchain-Enforced Accountability," IEEE Transactions on Dependable
and Secure Computing.
```
