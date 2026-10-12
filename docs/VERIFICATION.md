# Final cross-file verification — DA-ADKG v2

**Date:** 12 Oct 2026 (UTC+8).
**Machine:** Intel Xeon @ 2.80 GHz, 2 cores (the session's cloud workspace).
**Environment:** python: Python 3.13.16; numpy 2.5.3; scipy 1.18.1; matplotlib 3.11.2; cryptography 50.0.1; pdflatex: pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian)

**What was run.** The complete pipeline, end to end, on a clean copy of the nine packaged files (SHA-256 in `SHA256SUMS`). The full console output is in `VERIFICATION.log`.

| Step | Exit | Time (s) |
|---|---|---|
| 00 lint (pyflakes, all modules) | 0 (pass) | 0 |
| 01 crypto self-test | 0 (pass) | 1 |
| 02 protocol self-test | 0 (pass) | 32 |
| 03 network self-test | 0 (pass) | 64 |
| 04 baselines self-test (anchors vs PDFs, D4 IDs) | 0 (pass) | 0 |
| 05 experiments self-test | 0 (pass) | 241 |
| 06 experiments quick run (resumable cache) | 0 (pass) | 229 |
| 07 resume of a finished cache (must reuse every job) | 0 (pass) | 2 |
| 08 freshness of the results file | 0 (pass) | 1 |
| 09 figures from the results file | 0 (pass) | 6 |
| 10 figures self-test on the same results | 0 (pass) | 6 |
| 11 integration suite (22 checks, anchors on) | 0 (pass) | 208 |

## Cross-file consistency, checked on the artefacts of this pass

- **Resume:** re-running the quick configuration on its finished cache reused every job, and the results were identical apart from the timestamp.
- **Results ↔ code:** the source hashes in the results file equal the packaged sources (freshness check passed).
- **Results ↔ figures:** the figure manifest and every table header carry the results file's SHA-256, and the manifest's output hashes match the files.
- **Scenarios:** every within-assumption scenario reached 100% success with every accountability invariant on every run.
- **Estimator ↔ simulator:** bytes sent, gas and ledger bytes are equal exactly.

## Integration suite (step 11)

```
  PASS  I1 sync and async drivers agree                        45.3 s
  PASS  I2 gas: metered = estimator = schedule                 13.9 s
  PASS  I3 bytes: formulas = serialised messages                9.8 s
  PASS  I4 experiments.execute = direct run                    13.9 s
  PASS  I5 cost model held out; windows; estimator; adversarial async   61.7 s
  PASS  I6 determinism across processes                         7.5 s
  PASS  I7 pipeline experiments -> results -> figures          33.7 s
  INFO  I8 gas-accounting report (F9)                           0.0 s
  PASS  R1 replayed dispute transfers once                      0.2 s
  PASS  R2 selective dealer excluded, holders complete          2.4 s
  PASS  R3 F1: vote rule vs complaint rule                      4.6 s
  PASS  R4 omission terminates with exclusion                   2.6 s
  PASS  R5 bytes counted per recipient                          0.0 s
  PASS  R6 protocol group is BN254 G1                           0.0 s
  PASS  P5 finalised key is g^s                                 2.5 s
  PASS  A1 non-frameability under evidence mutation             1.9 s
  PASS  A2 forged answers rejected                              0.1 s
  PASS  A3 three Byzantine strategies at t = 3                  5.9 s
  PASS  A4 key finalisation needs t+1 matching votes            0.1 s
  PASS  A5 late dispute rejected                                0.2 s
  PASS  A6 decoder fuzz                                         0.5 s
  PASS  B1 published anchors verified against the PDFs          0.0 s
```

## Defect found and fixed during this pass

**Problem.** The `crypto.py` self-test failed once. It was a test bug; the library was correct.

- The Merkle tamper check replaced a random leaf's first byte with 0x01. When the leaf already began with 0x01, the "tampered" leaf was the original, so the negative check failed.
- That happens with probability 1/256 per check, about 16% of runs.

**Fix.** The check now XORs the first byte with 1, so the tampered leaf always differs.

**Confirmation.**
- 40 consecutive `crypto.py` self-tests: 0 failures.
- Repeated runs of the protocol (6×) and network (4×) self-tests found no other flaky test.
- The whole pass was then re-run from scratch on the fixed code. This record is that re-run.

## v2.1 pass (12 Oct 2026, after the P-1 fix)

**Scope.** The same 12-step pass (`run_pass.sh`) was re-run from scratch on a clean copy of v2.1. All steps passed:

| Step | Check | Exit | Time |
|---|---|---|---|
| 00 | lint | 0 | |
| 01 | crypto | 0 | 1 s |
| 02 | protocol | 0 | 27 s |
| 03 | network | 0 | 54 s |
| 04 | baselines | 0 | |
| 05 | experiments self-test | 0 | 210 s |
| 06 | quick run with cache | 0 | 199 s |
| 07 | resume of the finished cache | 0 | |
| 08 | freshness | 0 | |
| 09 | figures | 0 | |
| 10 | figures self-test | 0 | |
| 11 | integration suite, 23 checks including the new R7 | 0 | 176 s |

**After the pass:**
- Only a caption string in `figures.py` changed: the section reference is now `\ref{sec:evaluation}`.
- After that change, the lint check and the figures self-test were re-run on the paper results. Both passed.

## Paper results

The full `--run paper` was executed on exactly this code: 2,080 runs, 12 Oct 2026. Its results file passes the freshness check:
- **SHA-256:** `595d283c…`
- **Integration test:** `integration_test.py --results results_paper.json --results-only` reports FRESH.
- **Summary:** Section VII of the paper and `results/results_paper.json`.
