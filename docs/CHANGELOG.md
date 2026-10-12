# CHANGELOG — v1 (recovered reconstruction) → v2 (revision artefact)

Each file was reviewed line by line. The review records (D7a–D7g) are internal working documents and are not part of this repository; the identifiers they use (F#, R#, P#, D#, DEC-#) also appear in code comments.

## crypto.py (D7a)
- **Curve:** the protocol group moved from secp256k1/Ed25519 to **BN254 G1**, the group of the EVM precompiles (EIP-196/197, EIP-1108 prices). Dealer signatures are **ECDSA secp256k1** in ecrecover form (low-s, RFC 6979-style nonces).
- **Proofs added:** Chaum–Pedersen DLEQ, and a Schnorr PoK over an arbitrary base.
- **Shares:** DH-masked shares (EthDKG-style opening).
- **Hashing and commitments:** Keccak-256, `hash_to_scalar` with wide reduction, and a strict-position Merkle tree.
- **Tests:** known-answer tests, plus a cross-check against `py_ecc`.
- **12 Oct:** a flaky Merkle tamper self-test was fixed. It was a test defect; no library code changed.

## protocol.py (D7b)
- **QUAL rule:** QUAL = anchored ∧ no provable fault ∧ every non-delivery complaint answered. This fixes F1, R2 and R3. The printed vote rule is kept only for the ablation.
- **Disputes:** deduplicated per (accuser, dealer); a dealer is slashed at most once (fixes R1). Disputes open on-chain via the disclosed DH key plus a DLEQ proof.
- **New mechanisms:** complaint/answer for omissions (F14: excluded, never slashed) and PoK fraud proofs.
- **Public key:** g^s, finalised on t+1 matching votes. It was a Pedersen commitment (F3).
- **Gas:** a per-transaction EVM gas schedule. `calldata_bytes()` is the single source of transaction sizes.

## network.py (D7c)
- **Simulator:** ledger-driven (blocks, finality, windows) instead of oracle barriers. Fixes R4, the hang on a withheld share.
- **Compute:** every EC and hash operation is counted and priced by unit costs **measured** on the running machine.
- **Bytes:** measured from the real serialised messages. Fixes R5, the broadcast-once undercount.
- **Planning:** window planning, plus a validated honest-case estimator for large n.
- **Ethereum profile:** 12 s slots, **96-slot finality** (sources in code). Times on Ethereum are upper bounds.

## baselines.py (D7d)
- **Removed:** all modelled baselines and invented constants. The test asserting DA-ADKG superiority and Wang'25 are also gone.
- **Now:** 58 **published** numbers from 7 papers. Each carries its page, testbed, evidence ID and a verbatim quote, and is verified by code against the PDF text.
- **Table I** rows: every cell is cited.

## experiments.py (D7e)
- **Removed:** the manuscript rates, the ×4 "adversarial multiplier", the recovery model tuned to 42.1 s, and the statistics run against models.
- **Executed scenarios:** 15, run with real cryptography. Accountability invariants are checked on every run, and success is reported with Wilson intervals.
- **Performance and scaling:** large-n performance on two chain profiles; scaling by log-log slope.
- **Dependability model:**
  - a phase-type CTMC built from measured phase durations;
  - failure rates from fault injection;
  - held-out validation;
  - expected time to a usable key, replacing steady-state availability.
- **Provenance and resuming:** source hashes, unit costs and CPU model are recorded. Runs can resume (`--cache`).

## figures.py (D7f)
- **Inputs:** Figure 1 (3 × 3) and five LaTeX tables are generated **only** from a results file plus the published anchors.
- **Removed:** the invented deterrence curve and the hard-coded significance footnote.
- **Checks:** every table carries the results SHA-256, and the tables compile at the IEEE two-column width.

## integration_test.py (D7g)
- **22 checks:**
  - the joins between modules (I1–I8);
  - regressions of every v1 defect (R1–R6, P5);
  - adversarial protocol checks (A1–A6);
  - anchor verification (B1).
- **Freshness check** for results files (`--results`, `--results-only`).
- The two separate check scripts were folded in and removed.

## README.md, requirements.txt
- **README:** rewritten, with the reproduction recipe and a table of what is measured, each with the assumption the paper must state.
- **requirements.txt:** `cryptography`, `numpy`, `scipy` and `matplotlib` are required; `pycryptodome` and `py_ecc` are optional.

## v2.1 — 12 Oct 2026: participant-set rule (finding P-1 of the full run)
The first full paper run showed that one party whose registration missed the REGISTRATION window stopped every dealer (QUAL empty). This contradicted tolerance of t faults. Authors' decision: apply the fix. All results were re-run fresh afterwards.
- **protocol.py (D9):** the participant set is the registered set when REGISTRATION closes. `SmartContract.anchor` rejects anchors while fewer than n − t parties are registered (the session is then void). `Participant.deal` deals only to registered parties.
- **network.py:** the driver deals once the window closes with at least n − t registrations. A run with fewer fails with the reason `registration-below-n-t`.
- **experiments.py:**
  - new scenario `absent` (k = t parties never register; within the assumptions);
  - `slow_links` reclassified as **beyond** the assumptions, because 20× latency exceeds Δ by construction.
- **integration_test.py:** new check **R7**. With 2 absent parties at t = 2 the run succeeds with QUAL = {1..5} and the audit holds; with 3 absent the session is void and no commitment gas is spent.
- **baselines.py:** the DA-ADKG row of Table I is updated to the revised paper:
  - static adversary;
  - theorems for inclusion, exclusion, non-frameability and key correctness, plus a secrecy sketch;
  - α·S slashing, with omission leading to exclusion only;
  - reporter share ρ;
  - phase-type CTMC validated on held-out runs.
- **figures.py:** captions now reference the manuscript's sections by `\ref`.
