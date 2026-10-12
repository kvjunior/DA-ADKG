"""
protocol.py -- DA-ADKG protocol logic, v2 (Algorithms 1-3, QUAL, key derivation).

Reference: Gao et al., "DA-ADKG: Dependable Asynchronous Distributed Key
Generation with Blockchain-Enforced Accountability" (TDSC-2025-08-1249,
major revision).  Change IDs P1-P10 and findings F# refer to the revision
dossier (research/D7_code_review_and_change_plan.md, research/D5_D6_*.md);
decisions DEC-# to research/00_STATUS_and_continuation.md.

WHAT THIS FILE IS
    The participant and smart-contract state machines, plus a synchronous
    reference driver (`run_dkg`).  network.py drives the SAME state machines
    under asynchrony; every timing decision is expressed through the
    contract's block HEIGHT and its window schedule, so the two drivers
    cannot diverge in protocol logic.

PROTOCOL v2 IN ONE PARAGRAPH (DEC-1, option (a); EthDKG-style accountability)
    Registration (Alg. 1) -> DEALING: each dealer anchors MR_i = Merkle root
    of its Pedersen coefficient commitments plus H(PoK) on-chain (D3) and
    sends every recipient a dealer-SIGNED, DH-MASKED share (P4) -> COMPLAINT
    window: a recipient with a provably invalid share files a DISPUTE
    (commission, slashable); a recipient with no usable share files a
    non-delivery COMPLAINT (omission, not provable, F14); anyone can file a
    PoK fraud proof against an anchored invalid PoK -> ANSWER window: the
    accused dealer posts the complainant's signed masked share on-chain ->
    DISPUTE window: a complainant whose answered share is invalid disputes it
    -> QUAL is fixed at a finalized height (P3):
        QUAL = { i : anchored  AND  no upheld dispute / fraud proof (P1)
                     AND every complaint against i answered }
    -> KEY_DERIVATION (Das'23 Sec. 4.4, P5): every holder publishes
    g^{s_i}, h^{s^_i} with proofs of knowledge; anyone interpolates the
    group public key g^s from 2t+1 verified values; the contract finalizes
    the key once t+1 registered parties submit the same value.

DOCUMENTED DEVIATIONS FROM THE PRINTED MANUSCRIPT (each must be reconciled
in the revision text; justification in D6/D7)
  (D1) Sharing degree 2t (not "ell = 2t+1"): 2t+1 shares reconstruct (F4).
  (D2) Dealer attribution by Ethereum ECDSA signatures over the MASKED share
       (ecrecover-verifiable); without attribution disputes are unsound (F5).
  (D3) O(1) on-chain anchoring per dealer: (MR_i, H(PoK_i)).
  (D4) The output is the public key g^s (Das'23 key derivation), not the
       Pedersen commitment Y = prod C_j0 (F3).  g-parts are revealed only
       after QUAL is fixed, which also prevents the key-bias attack (F2).
  (D5) QUAL rule (P1, P3, F1, F14).  The printed rule "sum_j V[j][i] >= 2t+1"
       fails under asynchrony (an honest dealer can be excluded when
       Byzantine rows arrive first; a selective dealer can qualify while
       leaving honest victims without shares -- reproduced in D7, R2/R3).
       v2 uses the complaint/answer rule above (EthDKG Sec. IV-C a; SoK p.7
       "a single complaint being enough to disqualify the dealer").  The
       printed rule is kept as QualRule.VOTES_LEGACY for ablation only.
       The verification matrix V survives as a public record DERIVED from
       on-chain complaints/disputes: V[j][i] = 0 iff j complained about or
       disputed i, or i never anchored.
  (D6) Shares are delivered as additively masked scalars under the static
       Diffie-Hellman key K_ij = sk_i * pk_j (EthDKG), instead of ECIES.
       The mask lets the CONTRACT open a disputed share once the accuser
       discloses K_ij with a DLEQ proof; ECIES/AES-GCM could not be opened
       on-chain.
  (D7) Economic parameters (F15).  alpha (slash fraction for a provable
       fault) defaults to 1.0 (full stake, as Casper FFG); the reporter
       reward defaults to 10% of the slashed amount (Casper's "finder's
       fee"); the manuscript's values (alpha in [0.5, 1], reward alpha*S/2)
       remain reproducible via SystemParams(alpha=..., reporter_share=0.5).
       Omissions (unanswered complaints) cause exclusion WITHOUT slashing,
       because non-delivery is not provable (F14).  Stake stays locked until
       the dispute window closes plus a finality margin.
  (D8) A dealer is slashed at most once per session and a pair
       (accuser, dealer) is adjudicated at most once: v1 let the same
       evidence be replayed to drain a dealer's stake (D7, R1).
  (D9) Participant set = the parties registered when the REGISTRATION
       window closes (finding P-1 of the full evaluation run: in v2.0 a
       single party that failed to register stopped every dealer).  Dealers
       deal only to registered parties, and the contract accepts anchors
       only if at least n - t parties registered; otherwise the session is
       void and must be restarted.  Unregistered parties count as faulty.

GAS
    Gas is ESTIMATED from the public EVM schedule (constants below, with
    calldata priced at the non-zero byte rate, i.e. an upper bound) for
    every transaction type the contract would execute.  The revision must
    say so, or replace these figures by devnet measurements of a deployed
    Solidity contract.

Run `python3 protocol.py` for the self-test scenarios.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from crypto import (
    G,
    H,
    Q,
    DLEQProof,
    Point,
    PolyPoK,
    Polynomial,
    SchnorrProof,
    b2i,
    commit_poly_eval,
    commitment_root,
    dec_parts,
    deser_point,
    dh_key,
    dleq_prove,
    dleq_verify,
    double_mul,
    ecdsa_keygen,
    ecdsa_sign,
    ecdsa_verify,
    enc_parts,
    i2b,
    interpolate_at_zero_exp,
    is_on_curve,
    keccak256,
    keygen,
    lagrange_at_zero,
    mask_scalars,
    pedersen_commit,
    point_add,
    point_mul,
    pok_prove,
    pok_prove_base,
    pok_verify,
    pok_verify_base,
    poly_pok_prove,
    poly_pok_verify,
    scalar_from_bytes,
    ser_point,
    ser_point_evm,
    deser_point_evm,
    unmask_scalars,
    verify_share,
)

# ==========================================================================
# 1. Phases, windows, rules, strategies.
# ==========================================================================


class Phase(Enum):
    """Paper-level phases (Sec. IV-A) = CTMC states S0..S4, S_F, S_fail
    (Sec. V-A).  QUALIFICATION is instantaneous: the QUAL set is fixed at
    the height where the DISPUTE window closes."""

    REGISTRATION = auto()    # S0
    COMMITMENT = auto()      # S1  (window DEALING)
    VERIFICATION = auto()    # S2  (windows COMPLAINT, ANSWER, DISPUTE)
    QUALIFICATION = auto()   # S3
    KEY_DERIVATION = auto()  # S4  (paper: "reconstruction")
    SUCCESS = auto()         # S_F
    FAILED = auto()          # S_fail


class Window(Enum):
    """On-chain windows, in order.  Each lasts a number of BLOCKS given by
    the WindowSchedule; a transaction is accepted iff it is included at a
    height inside its window."""

    REGISTRATION = auto()
    DEALING = auto()
    COMPLAINT = auto()
    ANSWER = auto()
    DISPUTE = auto()
    KEY_DERIVATION = auto()
    CLOSED = auto()          # after KEY_DERIVATION


WINDOW_ORDER: Tuple[Window, ...] = (
    Window.REGISTRATION, Window.DEALING, Window.COMPLAINT, Window.ANSWER,
    Window.DISPUTE, Window.KEY_DERIVATION,
)

WINDOW_TO_PHASE: Dict[Window, Phase] = {
    Window.REGISTRATION: Phase.REGISTRATION,
    Window.DEALING: Phase.COMMITMENT,
    Window.COMPLAINT: Phase.VERIFICATION,
    Window.ANSWER: Phase.VERIFICATION,
    Window.DISPUTE: Phase.VERIFICATION,
    Window.KEY_DERIVATION: Phase.KEY_DERIVATION,
}


@dataclass(frozen=True)
class WindowSchedule:
    """Window lengths in blocks.  The defaults (1 block each) suit the
    synchronous driver; network.py sets lengths from the chain's block time
    and finality, and liveness requires the network to be synchronous for
    at least that long after GST (F12)."""

    registration: int = 1
    dealing: int = 1
    complaint: int = 1
    answer: int = 1
    dispute: int = 1
    key_derivation: int = 1

    def __post_init__(self) -> None:
        for name in ("registration", "dealing", "complaint", "answer",
                     "dispute", "key_derivation"):
            if getattr(self, name) < 1:
                raise ValueError(f"window '{name}' must last >= 1 block")

    def bounds(self, start: int = 0) -> Dict[Window, Tuple[int, int]]:
        """Half-open height intervals [begin, end) per window."""
        out: Dict[Window, Tuple[int, int]] = {}
        h = start
        for w, ln in zip(WINDOW_ORDER, (self.registration, self.dealing,
                                        self.complaint, self.answer,
                                        self.dispute, self.key_derivation)):
            out[w] = (h, h + ln)
            h += ln
        return out


class QualRule(Enum):
    COMPLAINT = auto()       # v2 rule (D5); the default
    VOTES_LEGACY = auto()    # printed Sec. IV-C rule, for ablation only


class Strategy(Enum):
    """Participant behaviours for the adversarial evaluation (Sec. VI-D).
    Timing behaviours (delays, partitions) belong to network.py."""

    HONEST = auto()
    SILENT = auto()                # registers, then crashes
    CORRUPT_ALL = auto()           # signed invalid shares to all others
    SELECTIVE = auto()             # signed invalid shares to `targets`
    WITHHOLD = auto()              # sends nothing to `targets`, never answers
    WITHHOLD_THEN_ANSWER = auto()  # sends nothing to `targets`, answers
    BAD_ANSWER = auto()            # withholds, then answers with invalid share
    BAD_POK = auto()               # anchors the hash of an invalid PoK
    FALSE_ACCUSER = auto()         # disputes the VALID share of `targets`
    FALSE_COMPLAINER = auto()      # claims non-delivery against `targets`
    VOTE_AGAINST = auto()          # legacy rule only: votes 0 for `targets`


# ==========================================================================
# 2. System parameters.
# ==========================================================================


@dataclass(frozen=True)
class SystemParams:
    t: int                                  # Byzantine tolerance, n = 3t+1
    s_min: float = 1000.0                   # minimum stake (Sec. III-F); a
                                            #   unit value: attack profits
                                            #   are scenario inputs
    alpha: float = 1.0                      # slash fraction, provable fault (D7)
    reporter_share: float = 0.10            # share of a slash paid to the
                                            #   first valid disputer (D7)
    gamma: float = 0.10                     # slash fraction for a false or
                                            #   malformed dispute (Alg. 3 l.22)
    sid: bytes = b"DA-ADKG/session-0"       # bound into every transcript
    qual_rule: QualRule = QualRule.COMPLAINT
    windows: WindowSchedule = field(default_factory=WindowSchedule)
    finality_margin: int = 1                # blocks of stake lock after DISPUTE

    def __post_init__(self) -> None:
        if not isinstance(self.t, int) or self.t < 1:
            raise ValueError("t >= 1 required")
        if self.s_min <= 0:
            raise ValueError("s_min must be positive")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must lie in (0, 1]")
        if not (0.0 <= self.reporter_share < 1.0):
            raise ValueError("reporter_share must lie in [0, 1)")
        if not (0.0 < self.gamma < 1.0):
            raise ValueError("gamma must lie in (0, 1)")
        if self.finality_margin < 0:
            raise ValueError("finality_margin must be >= 0")

    @property
    def n(self) -> int:
        return 3 * self.t + 1

    @property
    def degree(self) -> int:
        return 2 * self.t                   # (D1)

    @property
    def recon_threshold(self) -> int:
        return 2 * self.t + 1               # shares needed to reconstruct

    @property
    def qual_threshold(self) -> int:
        return 2 * self.t + 1               # legacy vote threshold only

    @property
    def min_qual(self) -> int:
        """|QUAL| >= t+1 guarantees at least one honest contribution, the
        condition for secrecy and uniformity of the group key."""
        return self.t + 1

    @property
    def key_confirmations(self) -> int:
        """Matching public-key submissions the contract needs: t+1 distinct
        registered parties include at least one honest party."""
        return self.t + 1


# ==========================================================================
# 3. Gas model (EVM schedule; estimates, see module docstring).
# ==========================================================================

GAS_TX = 21_000              # intrinsic transaction cost
GAS_CALLDATA_BYTE = 16       # non-zero calldata byte (EIP-2028); upper bound
GAS_SSTORE_SET = 22_100      # zero -> non-zero, cold slot (EIP-2929)
GAS_SSTORE_UPDATE = 5_000    # non-zero -> non-zero, cold slot
GAS_LOG_BASE = 375
GAS_LOG_TOPIC = 375
GAS_LOG_BYTE = 8
GAS_KECCAK_BASE = 30
GAS_KECCAK_WORD = 6
GAS_ECADD = 150              # BN254 precompile (EIP-1108)
GAS_ECMUL = 6_000            # BN254 precompile (EIP-1108)
GAS_ECRECOVER = 3_000

EVM_POINT = 64               # x || y
EVM_SCALAR = 32
EVM_SIG = 65                 # r || s || v
EVM_ADDR = 20
EVM_INDEX = 32               # uint256 ABI word


def _keccak_gas(nbytes: int) -> int:
    return GAS_KECCAK_BASE + GAS_KECCAK_WORD * math.ceil(nbytes / 32)


def _log_gas(topics: int, data_bytes: int) -> int:
    return GAS_LOG_BASE + GAS_LOG_TOPIC * topics + GAS_LOG_BYTE * data_bytes


def _calldata_gas(nbytes: int) -> int:
    return GAS_CALLDATA_BYTE * nbytes


def _h2s_gas(nbytes: int) -> int:
    """hash_to_scalar: two Keccak calls over (1 + preimage) bytes."""
    return 2 * _keccak_gas(1 + nbytes)


def _merkle_gas(num_leaves: int) -> int:
    """Leaves keccak(0x00 || 64B) and internal nodes keccak(0x01 || 2*32B)."""
    return num_leaves * _keccak_gas(1 + EVM_POINT) + \
        max(num_leaves - 1, 0) * _keccak_gas(1 + 64)


def _poly_pok_bytes(m: int) -> int:
    return m * EVM_POINT + 2 * m * EVM_SCALAR


def calldata_bytes(kind: str, p: Optional[SystemParams] = None) -> int:
    """ABI calldata size of each transaction type (single source for the gas
    model here and for ledger-write accounting in network.py)."""
    m = p.degree + 1 if p is not None else 0
    n = p.n if p is not None else 0
    sizes = {
        "register": EVM_INDEX + EVM_POINT + (EVM_POINT + EVM_SCALAR) + EVM_ADDR,
        "anchor": EVM_INDEX + 64,
        "complain": 2 * EVM_INDEX,
        "answer": (2 * EVM_INDEX + m * EVM_POINT + _poly_pok_bytes(m)
                   + 2 * EVM_SCALAR + EVM_SIG),
        "dispute": (2 * EVM_INDEX + m * EVM_POINT + 2 * EVM_SCALAR + EVM_SIG
                    + EVM_POINT + 2 * EVM_SCALAR),
        "pok_fraud": EVM_INDEX + m * EVM_POINT + _poly_pok_bytes(m),
        "submit_vrow": EVM_INDEX + math.ceil(n / 8),
        "submit_public_key": EVM_INDEX + EVM_POINT,
    }
    return sizes[kind]


def gas_register() -> int:
    calldata = calldata_bytes("register")
    pok_verify_gas = 2 * GAS_ECMUL + GAS_ECADD + _h2s_gas(3 * EVM_POINT + 64)
    return (GAS_TX + _calldata_gas(calldata) + pok_verify_gas
            + 4 * GAS_SSTORE_SET + _log_gas(2, EVM_POINT))


def gas_anchor() -> int:
    return (GAS_TX + _calldata_gas(calldata_bytes("anchor")) + 2 * GAS_SSTORE_SET
            + _log_gas(2, 64))


def gas_complaint() -> int:
    return (GAS_TX + _calldata_gas(calldata_bytes("complain")) + GAS_SSTORE_SET
            + _log_gas(3, 0))


def gas_answer(p: SystemParams) -> int:
    m = p.degree + 1
    calldata = calldata_bytes("answer", p)
    checks = (_merkle_gas(m) + _keccak_gas(_poly_pok_bytes(m))
              + _keccak_gas(200) + GAS_ECRECOVER)
    return (GAS_TX + _calldata_gas(calldata) + checks + GAS_SSTORE_SET
            + _log_gas(3, 2 * EVM_SCALAR))


def gas_dispute(p: SystemParams) -> int:
    m = p.degree + 1
    d = p.degree
    calldata = calldata_bytes("dispute", p)
    checks = (_merkle_gas(m) + _keccak_gas(200) + GAS_ECRECOVER
              + 4 * GAS_ECMUL + 2 * GAS_ECADD + _h2s_gas(6 * EVM_POINT + 64)  # DLEQ
              + 2 * _h2s_gas(EVM_POINT + 100)                                  # unmask
              + d * (GAS_ECMUL + GAS_ECADD) + 2 * GAS_ECMUL + GAS_ECADD)       # share eq
    return (GAS_TX + _calldata_gas(calldata) + checks
            + 3 * GAS_SSTORE_UPDATE + GAS_SSTORE_SET + _log_gas(3, 64))


def gas_pok_fraud(p: SystemParams) -> int:
    m = p.degree + 1
    calldata = calldata_bytes("pok_fraud", p)
    checks = (_merkle_gas(m) + _keccak_gas(_poly_pok_bytes(m))
              + _h2s_gas(2 * m * EVM_POINT + 64)
              + m * (3 * GAS_ECMUL + 2 * GAS_ECADD))
    return (GAS_TX + _calldata_gas(calldata) + checks
            + 3 * GAS_SSTORE_UPDATE + GAS_SSTORE_SET + _log_gas(2, 32))


def gas_submit_key(finalizes: bool) -> int:
    g = (GAS_TX + _calldata_gas(calldata_bytes("submit_public_key")) + GAS_SSTORE_SET
         + _keccak_gas(EVM_POINT) + _log_gas(2, EVM_POINT))
    return g + (2 * GAS_SSTORE_SET if finalizes else 0)


def gas_vrow(p: SystemParams) -> int:
    return (GAS_TX + _calldata_gas(calldata_bytes("submit_vrow", p))
            + GAS_SSTORE_SET + _log_gas(2, 0))


def gas_rejected(calldata_bytes: int = 64) -> int:
    """A reverted transaction still pays intrinsic + calldata gas."""
    return GAS_TX + _calldata_gas(calldata_bytes)


class GasMeter:
    """Gas per category and transaction counts (Fig. 1(e) breakdown)."""

    CATEGORIES = ("registration", "commitment", "complaint", "answer",
                  "dispute", "pok_fraud", "key_derivation", "vrow", "rejected")

    def __init__(self) -> None:
        self.by_category: Dict[str, int] = defaultdict(int)
        self.tx_count: Dict[str, int] = defaultdict(int)

    def charge(self, category: str, amount: int) -> None:
        if category not in self.CATEGORIES:
            raise KeyError(category)
        self.by_category[category] += amount
        self.tx_count[category] += 1

    @property
    def total(self) -> int:
        return sum(self.by_category.values())


# ==========================================================================
# 4. Messages.
# ==========================================================================


def share_mask_context(sid: bytes, dealer: int, recipient: int) -> bytes:
    return enc_parts(b"share-mask-v2", sid, i2b(dealer, 4), i2b(recipient, 4))


def share_sig_msg(sid: bytes, dealer: int, recipient: int, mr: bytes,
                  c: int, c_hat: int) -> bytes:
    """What a dealer signs (D2): the MASKED share, bound to the session,
    both endpoints and the dealer's anchored commitment root."""
    return enc_parts(b"share-v2", sid, i2b(dealer, 4), i2b(recipient, 4),
                     mr, i2b(c), i2b(c_hat))


def disclosure_context(sid: bytes, accuser: int, dealer: int) -> bytes:
    return enc_parts(b"dh-disclosure-v2", sid, i2b(accuser, 4), i2b(dealer, 4))


def poly_pok_context(sid: bytes, dealer: int) -> bytes:
    return enc_parts(b"poly-pok-v2", sid, i2b(dealer, 4))


def registration_context(sid: bytes, index: int) -> bytes:
    return enc_parts(b"registration-v2", sid, i2b(index, 4))


def key_share_context(sid: bytes, index: int) -> bytes:
    return enc_parts(b"key-share-v2", sid, i2b(index, 4))


@dataclass(frozen=True)
class SharePayload:
    """Off-chain dealer -> recipient message (Alg. 2 l.18, v2 form).  The
    commitment vector and PoK are identical for every recipient (network.py
    accounts for them as one dissemination per receiver)."""

    dealer: int
    recipient: int
    commits: Tuple[Point, ...]
    poly_pok: PolyPoK
    c: int                  # masked s_ij
    c_hat: int              # masked s^_ij
    sig: bytes              # ECDSA over share_sig_msg(...)

    def to_bytes(self) -> bytes:
        return enc_parts(
            i2b(self.dealer, 4), i2b(self.recipient, 4),
            enc_parts(*[ser_point(x) for x in self.commits]),
            self.poly_pok.to_bytes(), i2b(self.c), i2b(self.c_hat), self.sig)

    @classmethod
    def from_bytes(cls, b: bytes) -> "SharePayload":
        d, r, cb, pb, c, ch, sig = dec_parts(b, expected=7)
        if len(d) != 4 or len(r) != 4:
            raise ValueError("bad index encoding")
        return cls(dealer=b2i(d), recipient=b2i(r),
                   commits=tuple(deser_point(x) for x in dec_parts(cb)),
                   poly_pok=PolyPoK.from_bytes(pb),
                   c=scalar_from_bytes(c), c_hat=scalar_from_bytes(ch), sig=sig)


@dataclass(frozen=True)
class DisputeEvidence:
    """Commission evidence (Alg. 3 l.13-20, v2): the dealer-signed masked
    share, the commitment vector, and the accuser's disclosure of the DH key
    K_ij with a DLEQ proof (log_G pk_accuser == log_{pk_dealer} K)."""

    accuser: int
    dealer: int
    commits: Tuple[Point, ...]
    c: int
    c_hat: int
    sig: bytes
    K: Point
    dleq: DLEQProof


@dataclass(frozen=True)
class Answer:
    """Dealer's on-chain answer to a non-delivery complaint (D5)."""

    dealer: int
    recipient: int
    commits: Tuple[Point, ...]
    poly_pok_bytes: bytes
    c: int
    c_hat: int
    sig: bytes


@dataclass(frozen=True)
class KeyShare:
    """Das'23 Sec. 4.4 key-derivation message (D4)."""

    index: int
    A: Point                 # g^{s_i}
    Bh: Point                # h^{s^_i}
    pok_a: SchnorrProof      # knowledge of s_i (base G)
    pok_b: SchnorrProof      # knowledge of s^_i (base H)

    def to_bytes(self) -> bytes:
        return enc_parts(i2b(self.index, 4), ser_point(self.A), ser_point(self.Bh),
                         self.pok_a.to_bytes(), self.pok_b.to_bytes())

    @classmethod
    def from_bytes(cls, b: bytes) -> "KeyShare":
        i, a, bh, pa, pb = dec_parts(b, expected=5)
        if len(i) != 4:
            raise ValueError("bad index encoding")
        return cls(b2i(i), deser_point(a), deser_point(bh),
                   SchnorrProof.from_bytes(pa), SchnorrProof.from_bytes(pb))


# ==========================================================================
# 5. The smart contract (simulated; Sec. III-D).
# ==========================================================================


@dataclass
class Registration:
    pk: Point                # BN254 identity / DH key
    address: bytes           # Ethereum address (ECDSA attribution key)
    stake: float
    locked_until: int        # height from which stake may be withdrawn


class SmartContract:
    """Deterministic contract state machine.  Every method models one
    transaction included at the current `height`; window checks use the
    schedule fixed at deployment.  Rejections are recorded and charged as
    reverted transactions."""

    def __init__(self, params: SystemParams, start_height: int = 0) -> None:
        self.params = params
        self.height = start_height
        self.bounds = params.windows.bounds(start_height)
        lock = self.bounds[Window.DISPUTE][1] + params.finality_margin
        self._lock_height = lock

        self.registry: Dict[int, Registration] = {}
        self.anchors: Dict[int, Tuple[bytes, bytes]] = {}     # i -> (MR_i, H(PoK_i))
        self.complaints: Dict[Tuple[int, int], bool] = {}     # (j, i) -> answered?
        self.answers: Dict[Tuple[int, int], Answer] = {}      # (i, j) -> answer
        self.adjudicated: Set[Tuple[int, int]] = set()        # (accuser, dealer)
        self.disputed: Set[Tuple[int, int]] = set()           # upheld or not
        self.faulty: Set[int] = set()                          # provable faults
        self.slashed: Set[int] = set()
        self.vrows: Dict[int, Tuple[int, ...]] = {}           # legacy rule only
        self.qual: Optional[List[int]] = None
        self.key_votes: Dict[bytes, Set[int]] = defaultdict(set)
        self.key_voted: Set[int] = set()
        self.public_key: Optional[bytes] = None               # EVM encoding

        self.treasury = 0.0
        self.events: List[dict] = []
        self.disputes: List[dict] = []
        self.ledger: List[dict] = []
        self.gas = GasMeter()

    # -- infrastructure ---------------------------------------------------

    def window(self) -> Window:
        for w in WINDOW_ORDER:
            b, e = self.bounds[w]
            if b <= self.height < e:
                return w
        return Window.CLOSED if self.height >= self.bounds[
            Window.KEY_DERIVATION][1] else Window.REGISTRATION

    def advance_to(self, height: int) -> None:
        if height < self.height:
            raise ValueError("height cannot decrease")
        self.height = height

    def _emit(self, etype: str, **kw) -> None:
        self.events.append({"seq": len(self.events), "height": self.height,
                            "type": etype, **kw})

    def _reject(self, what: str, who: int, reason: str,
                calldata_bytes: int = 64) -> bool:
        self.gas.charge("rejected", gas_rejected(calldata_bytes))
        self._emit("Rejected", tx=what, who=who, reason=reason)
        return False

    def _in(self, w: Window) -> bool:
        return self.window() is w

    # -- Alg. 1: registration --------------------------------------------

    def register(self, i: int, pk: Point, pok: SchnorrProof, address: bytes,
                 stake: float) -> bool:
        p = self.params
        if not self._in(Window.REGISTRATION):
            return self._reject("register", i, "outside window")
        if not (1 <= i <= p.n) or i in self.registry:
            return self._reject("register", i, "bad or duplicate index")
        if pk is None or not is_on_curve(pk) or not pok_verify(
                pk, pok, context=registration_context(p.sid, i)):
            return self._reject("register", i, "invalid proof of knowledge")
        if stake < p.s_min:
            return self._reject("register", i, "stake below S_min")
        if len(address) != 20 or any(r.address == address for r in self.registry.values()):
            return self._reject("register", i, "bad or reused address")
        self.gas.charge("registration", gas_register())
        self.registry[i] = Registration(pk, address, float(stake), self._lock_height)
        self._emit("Registered", who=i)
        return True

    # -- Alg. 2 l.13: anchoring ---------------------------------------------

    def anchor(self, i: int, mr: bytes, pok_hash: bytes) -> bool:
        if not self._in(Window.DEALING):
            return self._reject("anchor", i, "outside window")
        if i not in self.registry or i in self.anchors:
            return self._reject("anchor", i, "unregistered or duplicate")
        if len(self.registry) < self.params.n - self.params.t:      # (D9)
            return self._reject("anchor", i, "registration below n - t")
        if len(mr) != 32 or len(pok_hash) != 32:
            return self._reject("anchor", i, "malformed digests")
        self.gas.charge("commitment", gas_anchor())
        self.anchors[i] = (mr, pok_hash)
        self._emit("Anchored", who=i, mr=mr.hex())
        return True

    # -- D5: non-delivery complaint (omission) ---------------------------------

    def complain(self, j: int, i: int) -> bool:
        if not self._in(Window.COMPLAINT):
            return self._reject("complain", j, "outside window")
        if j not in self.registry or i not in self.anchors or i == j:
            return self._reject("complain", j, "invalid parties")
        if (j, i) in self.complaints:
            return self._reject("complain", j, "duplicate complaint")
        self.gas.charge("complaint", gas_complaint())
        self.complaints[(j, i)] = False
        self._emit("Complaint", accuser=j, dealer=i)
        return True

    def answer(self, ans: Answer) -> bool:
        i, j = ans.dealer, ans.recipient
        p = self.params
        if not self._in(Window.ANSWER):
            return self._reject("answer", i, "outside window")
        if self.complaints.get((j, i)) is not False:
            return self._reject("answer", i, "no open complaint")
        mr, pok_hash = self.anchors[i]
        if len(ans.commits) != p.degree + 1 or commitment_root(ans.commits) != mr:
            return self._reject("answer", i, "commitments do not match anchor")
        if keccak256(ans.poly_pok_bytes) != pok_hash:
            return self._reject("answer", i, "PoK does not match anchor")
        if not ecdsa_verify(self.registry[i].address,
                            share_sig_msg(p.sid, i, j, mr, ans.c, ans.c_hat), ans.sig):
            return self._reject("answer", i, "bad dealer signature")
        self.gas.charge("answer", gas_answer(p))
        self.complaints[(j, i)] = True
        self.answers[(i, j)] = ans
        self._emit("Answered", dealer=i, recipient=j)
        return True

    # -- Alg. 3 l.13-23: commission dispute ------------------------------------

    def _slash(self, party: int, fraction: float, reason: str,
               beneficiary: Optional[int]) -> Tuple[float, float]:
        reg = self.registry[party]
        amount = fraction * reg.stake
        reg.stake -= amount
        reward = 0.0
        if beneficiary is not None:
            reward = self.params.reporter_share * amount
            self.registry[beneficiary].stake += reward
        self.treasury += amount - reward
        self.ledger.append({"height": self.height, "party": party, "amount": amount,
                            "reason": reason, "beneficiary": beneficiary,
                            "reward": reward})
        return amount, reward

    def _record(self, accuser: int, dealer: int, verdict: str,
                slashed: float, reward: float, kind: str) -> str:
        rec = {"height": self.height, "kind": kind, "accuser": accuser,
               "dealer": dealer, "verdict": verdict, "slashed": slashed,
               "reward": reward}
        self.disputes.append(rec)
        self._emit("DisputeResolved", **rec)
        return verdict

    def _penalise_provable_fault(self, dealer: int, accuser: int,
                                 reason: str) -> Tuple[float, float]:
        self.faulty.add(dealer)
        if dealer in self.slashed:                 # (D8) at most once
            return 0.0, 0.0
        self.slashed.add(dealer)
        return self._slash(dealer, self.params.alpha, reason, beneficiary=accuser)

    def dispute(self, ev: DisputeEvidence) -> str:
        """Verdicts: 'upheld' (dealer provably signed an invalid share),
        'false-accusation' (the share is valid), 'dismissed' (evidence not
        attributable or malformed), 'duplicate' (pair already adjudicated).
        Accepted in the COMPLAINT and DISPUTE windows."""
        p = self.params
        j, i = ev.accuser, ev.dealer
        if not (self._in(Window.COMPLAINT) or self._in(Window.DISPUTE)):
            self._reject("dispute", j, "outside window", 600)
            return "rejected"
        if j not in self.registry or i not in self.anchors or i == j:
            self._reject("dispute", j, "invalid parties", 600)
            return "rejected"
        self.gas.charge("dispute", gas_dispute(p))
        if (j, i) in self.adjudicated:                       # (D8)
            return self._record(j, i, "duplicate", 0.0, 0.0, "share")
        self.adjudicated.add((j, i))
        self.disputed.add((j, i))

        mr, _ = self.anchors[i]
        attributable = (
            len(ev.commits) == p.degree + 1
            and all(c is not None and is_on_curve(c) for c in ev.commits)
            and commitment_root(ev.commits) == mr
            and ecdsa_verify(self.registry[i].address,
                             share_sig_msg(p.sid, i, j, mr, ev.c, ev.c_hat), ev.sig)
            and ev.K is not None and is_on_curve(ev.K)
            and dleq_verify(G, self.registry[j].pk, self.registry[i].pk, ev.K,
                            ev.dleq, context=disclosure_context(p.sid, j, i)))
        if not attributable:
            amt, _ = self._slash(j, p.gamma, "malformed-dispute", None)
            return self._record(j, i, "dismissed", amt, 0.0, "share")

        s, s_hat = unmask_scalars(ev.K, share_mask_context(p.sid, i, j),
                                  [ev.c, ev.c_hat])
        if verify_share(j, s, s_hat, ev.commits):
            amt, _ = self._slash(j, p.gamma, "false-accusation", None)
            return self._record(j, i, "false-accusation", amt, 0.0, "share")
        amt, rew = self._penalise_provable_fault(i, j, "invalid-share")
        return self._record(j, i, "upheld", amt, rew, "share")

    def pok_fraud(self, accuser: int, dealer: int, commits: Sequence[Point],
                  poly_pok_bytes: bytes) -> str:
        """Public-fault proof: the dealer anchored H(PoK) of a PoK that does
        not verify against its anchored commitments.  The anchoring
        transaction came from the dealer's account, so the fault is
        attributable without any signature on the PoK itself."""
        p = self.params
        if not (self._in(Window.COMPLAINT) or self._in(Window.DISPUTE)):
            self._reject("pok_fraud", accuser, "outside window", 2000)
            return "rejected"
        if accuser not in self.registry or dealer not in self.anchors:
            self._reject("pok_fraud", accuser, "invalid parties", 2000)
            return "rejected"
        self.gas.charge("pok_fraud", gas_pok_fraud(p))
        if dealer in self.faulty:
            return self._record(accuser, dealer, "duplicate", 0.0, 0.0, "pok")
        mr, pok_hash = self.anchors[dealer]
        try:
            pok = PolyPoK.from_bytes(poly_pok_bytes)
        except ValueError:
            pok = None
        if (pok is None or len(commits) != p.degree + 1
                or commitment_root(commits) != mr or keccak256(poly_pok_bytes) != pok_hash):
            amt, _ = self._slash(accuser, p.gamma, "malformed-pok-fraud", None)
            return self._record(accuser, dealer, "dismissed", amt, 0.0, "pok")
        if poly_pok_verify(commits, pok, context=poly_pok_context(p.sid, dealer)):
            amt, _ = self._slash(accuser, p.gamma, "false-pok-fraud", None)
            return self._record(accuser, dealer, "false-accusation", amt, 0.0, "pok")
        amt, rew = self._penalise_provable_fault(dealer, accuser, "invalid-pok")
        return self._record(accuser, dealer, "upheld", amt, rew, "pok")

    # -- legacy rule: verification rows (Sec. IV-C as printed) ---------------

    def submit_vrow(self, j: int, row: Sequence[int]) -> bool:
        if self.params.qual_rule is not QualRule.VOTES_LEGACY:
            return self._reject("vrow", j, "rows unused under the complaint rule")
        if not self._in(Window.COMPLAINT):
            return self._reject("vrow", j, "outside window")
        if j not in self.registry or j in self.vrows or len(row) != self.params.n:
            return self._reject("vrow", j, "invalid row")
        self.gas.charge("vrow", gas_vrow(self.params))
        self.vrows[j] = tuple(1 if x else 0 for x in row)
        self._emit("VerificationRow", who=j)
        return True

    # -- Sec. IV-C: QUAL ------------------------------------------------------

    def close_qualification(self) -> List[int]:
        """Fix QUAL once the DISPUTE window has closed (callable by anyone,
        idempotent)."""
        if self.qual is not None:
            return self.qual
        if self.height < self.bounds[Window.DISPUTE][1]:
            raise RuntimeError("QUAL can only be fixed after the DISPUTE window")
        p = self.params
        qual: List[int] = []
        for i in range(1, p.n + 1):
            if i not in self.anchors:
                continue
            if p.qual_rule is QualRule.COMPLAINT:
                open_complaint = any(not answered for (j, d), answered
                                     in self.complaints.items() if d == i)
                if i in self.faulty or open_complaint:
                    continue
            else:  # VOTES_LEGACY: exactly the printed predicate
                votes = sum(row[i - 1] for row in self.vrows.values())
                if votes < p.qual_threshold:
                    continue
            qual.append(i)
        self.qual = qual
        self._emit("QualFixed", qual=list(qual))
        return qual

    def v_matrix(self) -> List[List[int]]:
        """V[j][i] (1-indexed rows/cols as lists).  Under the complaint rule
        it is DERIVED from on-chain complaints/disputes (D5)."""
        n = self.params.n
        if self.params.qual_rule is QualRule.VOTES_LEGACY:
            return [list(self.vrows.get(j, (0,) * n)) for j in range(1, n + 1)]
        V = []
        for j in range(1, n + 1):
            row = []
            for i in range(1, n + 1):
                bad = (i not in self.anchors or (j, i) in self.complaints
                       or (j, i) in self.disputed)
                row.append(0 if bad else 1)
            V.append(row)
        return V

    # -- key derivation (D4) -------------------------------------------------

    def submit_public_key(self, i: int, pk_evm: bytes) -> bool:
        p = self.params
        if not self._in(Window.KEY_DERIVATION):
            return self._reject("submit_key", i, "outside window")
        if self.qual is None:
            return self._reject("submit_key", i, "QUAL not fixed")
        if i not in self.registry or i in self.key_voted:
            return self._reject("submit_key", i, "unregistered or duplicate")
        try:
            pk_pt = deser_point_evm(pk_evm)
        except ValueError:
            return self._reject("submit_key", i, "invalid point")
        if pk_pt is None:
            return self._reject("submit_key", i, "identity key")
        self.key_voted.add(i)
        self.key_votes[pk_evm].add(i)
        finalizes = (self.public_key is None
                     and len(self.key_votes[pk_evm]) >= p.key_confirmations)
        self.gas.charge("key_derivation", gas_submit_key(finalizes))
        if finalizes:
            self.public_key = pk_evm
            self._emit("PublicKeyFinalized", key=pk_evm.hex())
        return True

    # -- stake -----------------------------------------------------------------

    def can_withdraw(self, i: int) -> bool:
        return i in self.registry and self.height >= self.registry[i].locked_until

    def total_value(self) -> float:
        return sum(r.stake for r in self.registry.values()) + self.treasury


# ==========================================================================
# 6. Participant state machine.
# ==========================================================================


class Participant:
    def __init__(self, index: int, params: SystemParams,
                 strategy: Strategy = Strategy.HONEST,
                 targets: Tuple[int, ...] = (),
                 stake: Optional[float] = None) -> None:
        if not (1 <= index <= params.n):
            raise ValueError("indices are 1..n (x = 0 encodes the secret)")
        self.index = index
        self.params = params
        self.strategy = strategy
        self.targets = tuple(targets)
        self.stake = float(stake if stake is not None else params.s_min)

        self.sk, self.pk = keygen()                     # BN254 identity
        self.ecdsa_sk, self.address = ecdsa_keygen()    # attribution key

        # dealer side
        self._f: Optional[Polynomial] = None
        self._fh: Optional[Polynomial] = None
        self.commits: Optional[Tuple[Point, ...]] = None
        self.mr: Optional[bytes] = None
        self.poly_pok_bytes: Optional[bytes] = None
        self._directory: Dict[int, Point] = {}

        # recipient side
        self.shares: Dict[int, Tuple[int, int]] = {}            # valid (s, s^)
        self.commits_of: Dict[int, Tuple[Point, ...]] = {}
        self.valid_payload: Dict[int, SharePayload] = {}
        self.provable_bad: Dict[int, SharePayload] = {}         # signed, invalid
        self.pok_bad: Dict[int, SharePayload] = {}              # anchored bad PoK
        self.complained: Set[int] = set()

        # outputs
        self.qual: Optional[List[int]] = None
        self.final_share: Optional[Tuple[int, int]] = None
        self.public_key: Optional[Point] = None
        self._agg_commits: Optional[Tuple[Point, ...]] = None

    # -- helpers ------------------------------------------------------------

    @property
    def participates(self) -> bool:
        return self.strategy is not Strategy.SILENT

    def _k_with(self, other_pk: Point) -> Point:
        return dh_key(self.sk, other_pk)

    # -- Alg. 1 ---------------------------------------------------------------

    def registration_tx(self) -> Tuple[int, Point, SchnorrProof, bytes, float]:
        pok = pok_prove(self.sk, self.pk,
                        context=registration_context(self.params.sid, self.index))
        return (self.index, self.pk, pok, self.address, self.stake)

    # -- Alg. 2 ---------------------------------------------------------------

    def _make_payload(self, j: int, s: int, s_hat: int) -> SharePayload:
        p = self.params
        K = self._k_with(self._directory[j])
        c, c_hat = mask_scalars(K, share_mask_context(p.sid, self.index, j), [s, s_hat])
        sig = ecdsa_sign(self.ecdsa_sk, share_sig_msg(p.sid, self.index, j, self.mr, c, c_hat))
        return SharePayload(self.index, j, self.commits,
                            PolyPoK.from_bytes(self.poly_pok_bytes), c, c_hat, sig)

    def _corrupt_for(self, j: int, answering: bool = False) -> bool:
        if j == self.index:
            return False
        if self.strategy is Strategy.CORRUPT_ALL:
            return True
        if self.strategy is Strategy.SELECTIVE:
            return j in self.targets
        return answering and self.strategy is Strategy.BAD_ANSWER

    def _withholds_from(self, j: int) -> bool:
        return (self.strategy in (Strategy.WITHHOLD, Strategy.WITHHOLD_THEN_ANSWER,
                                  Strategy.BAD_ANSWER) and j in self.targets)

    def deal(self, directory: Dict[int, Point]
             ) -> Optional[Tuple[bytes, bytes, Dict[int, Optional[bytes]]]]:
        """Alg. 2.  Returns (MR_i, H(PoK_i), {j: payload bytes or None}),
        or None for a silent dealer."""
        if not self.participates:
            return None
        p = self.params
        self._directory = dict(directory)
        self._f = Polynomial.random(p.degree)
        self._fh = Polynomial.random(p.degree)
        self.commits = tuple(pedersen_commit(a, b)
                             for a, b in zip(self._f.coeffs, self._fh.coeffs))
        self.mr = commitment_root(self.commits)
        pok = poly_pok_prove(self._f.coeffs, self._fh.coeffs, self.commits,
                             context=poly_pok_context(p.sid, self.index))
        if self.strategy is Strategy.BAD_POK:
            pok = PolyPoK(pok.T, ((pok.z[0] + 1) % Q,) + pok.z[1:], pok.zh)
        self.poly_pok_bytes = pok.to_bytes()

        out: Dict[int, Optional[bytes]] = {}
        for j in range(1, p.n + 1):
            if j not in self._directory:          # (D9) unregistered: no share
                out[j] = None
                continue
            if self._withholds_from(j):
                out[j] = None
                continue
            s, s_hat = self._f.eval(j), self._fh.eval(j)
            if self._corrupt_for(j):
                s = (s + 1) % Q
            out[j] = self._make_payload(j, s, s_hat).to_bytes()
        return self.mr, keccak256(self.poly_pok_bytes), out

    # -- Alg. 3 (recipient side) ------------------------------------------------

    def receive_share(self, dealer: int, blob: Optional[bytes],
                      contract: SmartContract) -> str:
        """Classify one incoming share: 'valid', 'provable-invalid',
        'pok-fraud' or 'unusable' (missing, malformed, unattributable)."""
        p = self.params
        if blob is None or dealer not in contract.anchors or dealer not in contract.registry:
            return "unusable"
        try:
            pl = SharePayload.from_bytes(blob)
        except (ValueError, IndexError):
            return "unusable"
        if pl.dealer != dealer or pl.recipient != self.index:
            return "unusable"
        mr, pok_hash = contract.anchors[dealer]
        if (len(pl.commits) != p.degree + 1
                or not all(c is not None for c in pl.commits)
                or commitment_root(pl.commits) != mr):
            return "unusable"
        if not ecdsa_verify(contract.registry[dealer].address,
                            share_sig_msg(p.sid, dealer, self.index, mr, pl.c, pl.c_hat),
                            pl.sig):
            return "unusable"
        pok_bytes = pl.poly_pok.to_bytes()
        if keccak256(pok_bytes) != pok_hash:
            return "unusable"
        if not poly_pok_verify(pl.commits, pl.poly_pok,
                               context=poly_pok_context(p.sid, dealer)):
            self.pok_bad[dealer] = pl
            return "pok-fraud"
        K = self._k_with(contract.registry[dealer].pk)
        s, s_hat = unmask_scalars(K, share_mask_context(p.sid, dealer, self.index),
                                  [pl.c, pl.c_hat])
        if verify_share(self.index, s, s_hat, pl.commits):
            self.shares[dealer] = (s, s_hat)
            self.commits_of[dealer] = pl.commits
            self.valid_payload[dealer] = pl
            return "valid"
        self.provable_bad[dealer] = pl
        return "provable-invalid"

    def _evidence(self, dealer: int, commits, c: int, c_hat: int, sig: bytes,
                  contract: SmartContract) -> DisputeEvidence:
        p = self.params
        dpk = contract.registry[dealer].pk
        return DisputeEvidence(
            accuser=self.index, dealer=dealer, commits=tuple(commits), c=c,
            c_hat=c_hat, sig=sig, K=self._k_with(dpk),
            dleq=dleq_prove(self.sk, G, dpk,
                            context=disclosure_context(p.sid, self.index, dealer)))

    def act_in_complaint_window(self, contract: SmartContract) -> None:
        """File disputes (commission), PoK fraud proofs and non-delivery
        complaints (omission).  Under the legacy rule: post a vote row."""
        if not self.participates:
            return
        p = self.params
        if p.qual_rule is QualRule.VOTES_LEGACY:
            row = [1 if d in self.shares else 0 for d in range(1, p.n + 1)]
            if self.strategy is Strategy.VOTE_AGAINST:
                for d in self.targets:
                    row[d - 1] = 0
            contract.submit_vrow(self.index, row)
        for d in sorted(contract.anchors):
            if d == self.index:
                continue
            if d in self.pok_bad:
                pl = self.pok_bad[d]
                contract.pok_fraud(self.index, d, pl.commits, pl.poly_pok.to_bytes())
            elif d in self.provable_bad:
                pl = self.provable_bad[d]
                contract.dispute(self._evidence(d, pl.commits, pl.c, pl.c_hat,
                                                pl.sig, contract))
            elif d in self.shares:
                if self.strategy is Strategy.FALSE_ACCUSER and d in self.targets:
                    pl = self.valid_payload[d]
                    contract.dispute(self._evidence(d, pl.commits, pl.c, pl.c_hat,
                                                    pl.sig, contract))
                elif self.strategy is Strategy.FALSE_COMPLAINER and d in self.targets \
                        and p.qual_rule is QualRule.COMPLAINT:
                    if contract.complain(self.index, d):
                        self.complained.add(d)
            elif p.qual_rule is QualRule.COMPLAINT:
                if contract.complain(self.index, d):
                    self.complained.add(d)

    def act_in_answer_window(self, contract: SmartContract) -> None:
        """Dealer side of D5: answer every open complaint against me."""
        if not self.participates or self.commits is None:
            return
        if self.strategy is Strategy.WITHHOLD:
            return
        for (j, d), answered in sorted(contract.complaints.items()):
            if d != self.index or answered:
                continue
            s, s_hat = self._f.eval(j), self._fh.eval(j)
            if self._corrupt_for(j, answering=True):
                s = (s + 1) % Q
            pl = self._make_payload(j, s, s_hat)
            contract.answer(Answer(self.index, j, self.commits, self.poly_pok_bytes,
                                   pl.c, pl.c_hat, pl.sig))

    def act_in_dispute_window(self, contract: SmartContract) -> None:
        """Open answers to my complaints; dispute an invalid one."""
        if not self.participates:
            return
        p = self.params
        for d in sorted(self.complained):
            ans = contract.answers.get((d, self.index))
            if ans is None or d in self.shares:
                continue
            K = self._k_with(contract.registry[d].pk)
            s, s_hat = unmask_scalars(K, share_mask_context(p.sid, d, self.index),
                                      [ans.c, ans.c_hat])
            if verify_share(self.index, s, s_hat, ans.commits):
                self.shares[d] = (s, s_hat)
                self.commits_of[d] = ans.commits
            else:
                contract.dispute(self._evidence(d, ans.commits, ans.c, ans.c_hat,
                                                ans.sig, contract))

    # -- finalization (Sec. IV-B) ----------------------------------------------

    def finalize(self, qual: Sequence[int]) -> bool:
        """s_i = sum_{d in QUAL} s_di (and the blinding sum).  Under the
        complaint rule every participating honest node holds a valid share
        from every QUAL dealer (completeness)."""
        self.qual = list(qual)
        self._agg_commits = None
        if not all(d in self.shares for d in qual):
            self.final_share = None
            return False
        self.final_share = (sum(self.shares[d][0] for d in qual) % Q,
                            sum(self.shares[d][1] for d in qual) % Q)
        return True

    def aggregated_commitments(self) -> Tuple[Point, ...]:
        """C^agg_k = sum_{d in QUAL} C_{d,k}: the commitment to the summed
        polynomial (Pedersen commitments are additively homomorphic).
        Computed once (|QUAL| * m point additions), so that each key-share
        check costs one O(m) evaluation instead of |QUAL| of them."""
        if self._agg_commits is None:
            m = self.params.degree + 1
            agg: List[Point] = [None] * m
            for d in self.qual or []:
                for k in range(m):
                    agg[k] = point_add(agg[k], self.commits_of[d][k])
            self._agg_commits = tuple(agg)
        return self._agg_commits

    def expected_commitment(self, index: int) -> Point:
        """g^{s_index} h^{s^_index} = eval(C^agg, index)."""
        return commit_poly_eval(self.aggregated_commitments(), index)

    # -- key derivation (D4; Das'23 Sec. 4.4) -------------------------------------

    def key_share(self) -> Optional[KeyShare]:
        if self.final_share is None:
            return None
        s_i, sh_i = self.final_share
        ctx = key_share_context(self.params.sid, self.index)
        A, Bh = point_mul(s_i, G), point_mul(sh_i, H)
        return KeyShare(self.index, A, Bh,
                        pok_prove_base(s_i, G, A, ctx), pok_prove_base(sh_i, H, Bh, ctx))

    def verify_key_share(self, ks: KeyShare) -> bool:
        if self.qual is None or not all(d in self.commits_of for d in self.qual):
            return False
        ctx = key_share_context(self.params.sid, ks.index)
        return (pok_verify_base(ks.A, ks.pok_a, G, ctx)
                and pok_verify_base(ks.Bh, ks.pok_b, H, ctx)
                and point_add(ks.A, ks.Bh) == self.expected_commitment(ks.index))

    def derive_public_key(self, key_shares: Iterable[KeyShare]) -> Optional[Point]:
        """Interpolate g^s from the first 2t+1 VERIFIED key shares."""
        good: List[KeyShare] = []
        for ks in sorted(key_shares, key=lambda k: k.index):
            if self.verify_key_share(ks):
                good.append(ks)
            if len(good) == self.params.recon_threshold:
                break
        return self.public_key_from_verified(good)

    def public_key_from_verified(self, verified: Sequence[KeyShare]) -> Optional[Point]:
        """Interpolation step alone, for callers that verified incrementally
        (network.py verifies each key share on arrival)."""
        if len(verified) < self.params.recon_threshold:
            return None
        chosen = sorted(verified, key=lambda k: k.index)[: self.params.recon_threshold]
        self.public_key = interpolate_at_zero_exp([(ks.index, ks.A) for ks in chosen])
        return self.public_key


# ==========================================================================
# 7. Audit reconstruction (test facility only; never run in deployment).
# ==========================================================================


def reconstruct_secret(contributions: Dict[int, Tuple[int, int]],
                       qual: Sequence[int],
                       commits_of: Dict[int, Sequence[Point]],
                       params: SystemParams) -> Tuple[int, int]:
    """Interpolate (s, s^) from >= 2t+1 contributions, each verified against
    the aggregated commitment evaluation at its index."""
    verified: List[Tuple[int, int, int]] = []
    for idx, (s_i, sh_i) in sorted(contributions.items()):
        expect: Point = None
        for d in qual:
            expect = point_add(expect, commit_poly_eval(commits_of[d], idx))
        if double_mul(s_i, G, sh_i, H) == expect:
            verified.append((idx, s_i, sh_i))
        if len(verified) == params.recon_threshold:
            break
    if len(verified) < params.recon_threshold:
        raise ValueError(f"need {params.recon_threshold} valid contributions, "
                         f"got {len(verified)}")
    s = lagrange_at_zero([(i, v) for i, v, _ in verified])
    s_hat = lagrange_at_zero([(i, v) for i, _, v in verified])
    y: Point = None
    for d in qual:
        y = point_add(y, commits_of[d][0])
    if pedersen_commit(s, s_hat) != y:
        raise ValueError("reconstruction inconsistent with the commitments")
    return s, s_hat


# ==========================================================================
# 8. Synchronous reference driver.
# ==========================================================================


@dataclass
class DKGResult:
    success: bool
    reason: str
    qual: List[int]
    complete_holders: List[int]
    honest_complete_holders: List[int]
    public_key: Optional[bytes]                 # EVM encoding of g^s
    v_matrix: List[List[int]]
    complaints: Dict[Tuple[int, int], bool]
    disputes: List[dict]
    ledger: List[dict]
    gas_by_category: Dict[str, int]
    tx_count: Dict[str, int]
    gas_total: int
    treasury: float
    cpu_seconds: Dict[str, float]               # reference-crypto CPU time only
    audit: Optional[dict] = None


def run_dkg(params: SystemParams,
            byzantine: Optional[Dict[int, Tuple[Strategy, Tuple[int, ...]]]] = None,
            late: FrozenSet[int] = frozenset(),
            audit_reconstruct: bool = False) -> DKGResult:
    """Synchronous reference run.  `late` lists HONEST nodes whose
    complaint-window transactions miss the window (models asynchrony before
    GST for the ablation of the legacy rule; D7 R3)."""
    byzantine = dict(byzantine or {})
    if len(byzantine) > params.t:
        raise ValueError("adversary exceeds Byzantine threshold t")
    if set(late) & set(byzantine):
        raise ValueError("`late` models honest nodes only")

    parts = {i: Participant(i, params, *byzantine.get(i, (Strategy.HONEST, ())))
             for i in range(1, params.n + 1)}
    contract = SmartContract(params)
    b = contract.bounds
    cpu: Dict[str, float] = {}

    def timed(name: str):
        class _T:
            def __enter__(self_):
                self_.t0 = time.perf_counter()

            def __exit__(self_, *exc):
                cpu[name] = cpu.get(name, 0.0) + time.perf_counter() - self_.t0
        return _T()

    # registration
    contract.advance_to(b[Window.REGISTRATION][0])
    with timed("registration"):
        for p in parts.values():
            if not contract.register(*p.registration_tx()):
                raise RuntimeError(f"registration of {p.index} failed")
    directory = {i: parts[i].pk for i in parts}

    # dealing
    contract.advance_to(b[Window.DEALING][0])
    mail: Dict[int, Dict[int, Optional[bytes]]] = {j: {} for j in parts}
    with timed("dealing"):
        for i, p in parts.items():
            out = p.deal(directory)
            if out is None:
                continue
            mr, pok_hash, msgs = out
            contract.anchor(i, mr, pok_hash)
            for j, blob in msgs.items():
                mail[j][i] = blob

    # complaint window (late honest nodes act after it closes -> rejected)
    with timed("verification"):
        for j, p in parts.items():
            if p.participates:
                for i in sorted(contract.anchors):
                    p.receive_share(i, mail[j].get(i), contract)
        contract.advance_to(b[Window.COMPLAINT][0])
        for j, p in parts.items():
            if j not in late:
                p.act_in_complaint_window(contract)
        contract.advance_to(b[Window.ANSWER][0])
        for j in sorted(late):
            parts[j].act_in_complaint_window(contract)
        for p in parts.values():
            p.act_in_answer_window(contract)
        contract.advance_to(b[Window.DISPUTE][0])
        for p in parts.values():
            p.act_in_dispute_window(contract)

    # qualification at the end of the dispute window
    contract.advance_to(b[Window.DISPUTE][1])
    qual = contract.close_qualification()
    honest = [i for i in parts if i not in byzantine]

    def finish(success: bool, reason: str, complete: List[int],
               audit: Optional[dict] = None) -> DKGResult:
        return DKGResult(
            success=success, reason=reason, qual=list(qual),
            complete_holders=complete,
            honest_complete_holders=[i for i in complete if i in honest],
            public_key=contract.public_key, v_matrix=contract.v_matrix(),
            complaints=dict(contract.complaints), disputes=list(contract.disputes),
            ledger=list(contract.ledger),
            gas_by_category=dict(contract.gas.by_category),
            tx_count=dict(contract.gas.tx_count), gas_total=contract.gas.total,
            treasury=contract.treasury, cpu_seconds=cpu, audit=audit)

    if len(qual) < params.min_qual:
        return finish(False, "qual-below-t+1", [])

    # key derivation
    with timed("key_derivation"):
        complete = [i for i, p in parts.items() if p.participates and p.finalize(qual)]
        key_shares = [ks for ks in (parts[i].key_share() for i in complete) if ks]
        derived: Dict[int, Optional[Point]] = {}
        for i in complete:
            derived[i] = parts[i].derive_public_key(key_shares)
        contract.advance_to(b[Window.KEY_DERIVATION][0])
        for i in complete:
            if derived[i] is not None:
                contract.submit_public_key(i, ser_point_evm(derived[i]))

    honest_complete = [i for i in complete if i in honest]
    if len(honest_complete) < params.recon_threshold:
        # completion would depend on Byzantine cooperation (D7 R2 hostage)
        return finish(False, "honest-holders-below-2t+1", complete)
    if contract.public_key is None:
        return finish(False, "public-key-not-finalized", complete)
    if {ser_point_evm(derived[i]) for i in honest_complete} != {contract.public_key}:
        return finish(False, "public-key-disagreement", complete)

    audit = None
    if audit_reconstruct:
        contribs = {i: parts[i].final_share for i in honest_complete}
        s, _ = reconstruct_secret(contribs, qual, parts[honest_complete[0]].commits_of,
                                  params)
        truth = sum(parts[d]._f.coeffs[0] for d in qual) % Q
        audit = {"secret_matches_dealer_sum": s == truth,
                 "public_key_is_g_to_s":
                     ser_point_evm(point_mul(s, G)) == contract.public_key}
        if not all(audit.values()):
            return finish(False, "audit-failed", complete, audit)
    return finish(True, "ok", complete, audit)


# ==========================================================================
# 9. Self-tests.
# ==========================================================================


def _selftest() -> None:  # pragma: no cover
    close = lambda a, b: math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
    ok = lambda s: print(f"  PASS  {s}")
    p1, p2 = SystemParams(t=1), SystemParams(t=2)
    S = Strategy

    def stake_of(r: DKGResult, i: int, base: float = 1000.0) -> float:
        return base - sum(e["amount"] for e in r.ledger if e["party"] == i) + \
            sum(e["reward"] for e in r.ledger if e["beneficiary"] == i)

    # (a) honest runs: full QUAL, g^s public key, audit, no complaints
    for p in (p1, p2):
        r = run_dkg(p, audit_reconstruct=True)
        assert r.success and r.reason == "ok", r.reason
        assert r.qual == list(range(1, p.n + 1)) and not r.complaints and not r.disputes
        assert r.audit == {"secret_matches_dealer_sum": True, "public_key_is_g_to_s": True}
        assert all(all(v == 1 for v in row) for row in r.v_matrix)
        assert "rejected" not in r.gas_by_category
    ok("(a) honest t=1,2: full QUAL, public key g^s finalized and audited (F3)")

    # (b) CORRUPT_ALL: excluded, slashed ONCE in full, reporter fee to the
    #     first disputer only, later disputes recorded without transfer (P2)
    r = run_dkg(p2, byzantine={7: (S.CORRUPT_ALL, ())}, audit_reconstruct=True)
    assert r.success and r.qual == [1, 2, 3, 4, 5, 6]
    up = [d for d in r.disputes if d["verdict"] == "upheld"]
    assert len(up) == 6 and all(d["dealer"] == 7 for d in up)
    assert sum(1 for d in up if d["slashed"] > 0) == 1
    assert close(stake_of(r, 7), 0.0) and close(stake_of(r, up[0]["accuser"]), 1100.0)
    assert close(r.treasury, 900.0)
    ok("(b) CORRUPT_ALL: excluded; slashed once (1000 -> 0); 10% fee to first disputer")

    # (c) SELECTIVE: now EXCLUDED (v1 kept it in QUAL; D7 R2), no hostage
    r = run_dkg(p2, byzantine={6: (S.SELECTIVE, (1, 2))}, audit_reconstruct=True)
    assert r.success and 6 not in r.qual and r.qual == [1, 2, 3, 4, 5, 7]
    assert sorted(r.honest_complete_holders) == [1, 2, 3, 4, 5, 7]
    ok("(c) SELECTIVE: upheld disputes exclude the dealer; all 6 honest hold shares (R2 fixed)")

    # (d) FALSE_ACCUSER: gamma slash on the accuser; honest dealer stays
    r = run_dkg(p1, byzantine={3: (S.FALSE_ACCUSER, (2,))})
    assert r.success and 2 in r.qual
    fa = [d for d in r.disputes if d["verdict"] == "false-accusation"]
    assert len(fa) == 1 and fa[0]["accuser"] == 3 and close(stake_of(r, 3), 900.0)
    assert close(stake_of(r, 2), 1000.0)
    ok("(d) false accusation: provable, accuser slashed gamma, dealer untouched")

    # (e) WITHHOLD: complaint unanswered -> excluded, NOT slashed (F14)
    r = run_dkg(p2, byzantine={5: (S.WITHHOLD, (1,))})
    assert r.success and 5 not in r.qual and close(stake_of(r, 5), 1000.0)
    assert r.complaints == {(1, 5): False} and not r.disputes
    ok("(e) omission: unanswered complaint excludes the dealer without slashing (F14)")

    # (f) WITHHOLD_THEN_ANSWER: answered on-chain -> victims obtain shares
    r = run_dkg(p2, byzantine={5: (S.WITHHOLD_THEN_ANSWER, (1, 2))}, audit_reconstruct=True)
    assert r.success and 5 in r.qual and r.complaints == {(1, 5): True, (2, 5): True}
    assert {1, 2} <= set(r.honest_complete_holders) and not r.ledger
    ok("(f) answered complaints: victims recover their shares from the chain")

    # (g) BAD_ANSWER: invalid answered share -> upheld dispute -> excluded + slashed
    r = run_dkg(p2, byzantine={5: (S.BAD_ANSWER, (1,))})
    assert r.success and 5 not in r.qual
    assert [d["verdict"] for d in r.disputes] == ["upheld"] and close(stake_of(r, 5), 0.0)
    ok("(g) invalid answer: provable, dealer excluded and slashed")

    # (h) FALSE_COMPLAINER: honest dealer answers; nobody slashed (unprovable)
    r = run_dkg(p2, byzantine={7: (S.FALSE_COMPLAINER, (3,))})
    assert r.success and 3 in r.qual and r.complaints == {(7, 3): True} and not r.ledger
    ok("(h) false non-delivery claim: dealer answers, stays in QUAL, no slashing")

    # (i) BAD_POK: public fault proof -> excluded + slashed once
    r = run_dkg(p2, byzantine={4: (S.BAD_POK, ())})
    assert r.success and 4 not in r.qual
    pk_rec = [d for d in r.disputes if d["kind"] == "pok"]
    assert pk_rec[0]["verdict"] == "upheld" and all(d["verdict"] == "duplicate" for d in pk_rec[1:])
    assert close(stake_of(r, 4), 0.0)
    ok("(i) invalid anchored PoK: public fraud proof, dealer excluded and slashed once")

    # (j) replayed dispute (v1 drained the dealer; D7 R1): 'duplicate', no transfer
    parts = {i: Participant(i, p2, *( (S.SELECTIVE, (1,)) if i == 6 else (S.HONEST, ())))
             for i in range(1, 8)}
    c = SmartContract(p2)
    for q in parts.values():
        c.register(*q.registration_tx())
    c.advance_to(c.bounds[Window.DEALING][0])
    mr, ph, msgs = parts[6].deal({i: parts[i].pk for i in parts})
    c.anchor(6, mr, ph)
    assert parts[1].receive_share(6, msgs[1], c) == "provable-invalid"
    c.advance_to(c.bounds[Window.COMPLAINT][0])
    pl = parts[1].provable_bad[6]
    ev = parts[1]._evidence(6, pl.commits, pl.c, pl.c_hat, pl.sig, c)
    verdicts = [c.dispute(ev) for _ in range(10)]
    assert verdicts == ["upheld"] + ["duplicate"] * 9
    assert close(c.registry[6].stake, 0.0) and close(c.registry[1].stake, 1100.0)
    ok("(j) replayed evidence: 1 upheld + 9 duplicates, single transfer (R1 fixed)")

    # (k) forged evidence: wrong DH key or tampered ciphertext -> dismissed
    c2 = SmartContract(p2)
    c2.__dict__.update({k: v for k, v in c.__dict__.items()
                        if k in ("registry", "anchors", "bounds")})
    c2.advance_to(c2.bounds[Window.COMPLAINT][0])
    bad_k = DisputeEvidence(**{**ev.__dict__, "K": point_add(ev.K, G)})
    assert c2.dispute(bad_k) == "dismissed"
    c3 = SmartContract(p2)
    c3.__dict__.update({k: v for k, v in c.__dict__.items()
                        if k in ("registry", "anchors", "bounds")})
    c3.advance_to(c3.bounds[Window.COMPLAINT][0])
    bad_c = DisputeEvidence(**{**ev.__dict__, "c": (ev.c + 1) % Q})
    assert c3.dispute(bad_c) == "dismissed"
    ok("(k) non-frameability: forged DH key / tampered ciphertext are dismissed")

    # (l) legacy rule ablation reproduces F1 / D7 R3 and the hostage case;
    #     the complaint rule is immune to the same schedule
    adv = {6: (S.VOTE_AGAINST, (5,)), 7: (S.VOTE_AGAINST, (5,))}
    legacy = SystemParams(t=2, qual_rule=QualRule.VOTES_LEGACY)
    r = run_dkg(legacy, byzantine=adv, late=frozenset({3, 4}))
    assert 5 not in r.qual
    r2 = run_dkg(p2, byzantine={6: (S.FALSE_COMPLAINER, (5,)),
                                7: (S.FALSE_COMPLAINER, (5,))}, late=frozenset({3, 4}))
    assert r2.success and 5 in r2.qual
    r3 = run_dkg(legacy, byzantine={6: (S.SELECTIVE, (1, 2))})
    assert 6 in r3.qual and not r3.success and r3.reason == "honest-holders-below-2t+1"
    ok("(l) legacy vote rule: honest dealer excluded (R3) and key hostage (R2); "
       "complaint rule unaffected")

    # (m) value conservation, stake lock, registration binding, windows
    r = run_dkg(p2, byzantine={5: (S.SILENT, ()), 6: (S.SELECTIVE, (1,))})
    assert r.success
    assert close(sum(stake_of(r, i) for i in range(1, 8)) + r.treasury, 7000.0)
    c = SmartContract(p1)
    q = Participant(1, p1)
    idx, pk, pok, addr, st = q.registration_tx()
    assert not c.register(2, pk, pok, addr, st)             # PoK bound to index
    assert c.register(idx, pk, pok, addr, st)
    assert not c.can_withdraw(1)
    c.advance_to(c.bounds[Window.DISPUTE][1] + p1.finality_margin)
    assert c.can_withdraw(1)
    assert not c.complain(1, 2)                              # outside window
    ok("(m) conservation, stake lock, index-bound PoK, window enforcement")

    # (n) forged key share is rejected; derived key unaffected
    r_parts = {i: Participant(i, p1) for i in range(1, 5)}
    cc = SmartContract(p1)
    for q in r_parts.values():
        cc.register(*q.registration_tx())
    cc.advance_to(cc.bounds[Window.DEALING][0])
    d = {i: r_parts[i].pk for i in r_parts}
    mail = {}
    for i, q in r_parts.items():
        mr, ph, msgs = q.deal(d)
        cc.anchor(i, mr, ph)
        mail[i] = msgs
    for j, q in r_parts.items():
        for i in r_parts:
            assert q.receive_share(i, mail[i][j], cc) == "valid"
        q.finalize([1, 2, 3, 4])
    kss = [q.key_share() for q in r_parts.values()]
    forged = KeyShare(kss[0].index, point_add(kss[0].A, G), kss[0].Bh,
                      kss[0].pok_a, kss[0].pok_b)
    assert not r_parts[2].verify_key_share(forged)
    pk_a = r_parts[2].derive_public_key([forged] + kss[1:])
    pk_b = r_parts[3].derive_public_key(kss)
    assert pk_a == pk_b is not None
    ok("(n) forged key share rejected; public key unchanged")

    # gas snapshot (estimates from the EVM schedule)
    r = run_dkg(p2)
    per = {k: r.gas_by_category[k] // r.tx_count[k] for k in r.gas_by_category}
    print(f"  info  honest t=2 gas {r.gas_total:,} total; per tx: {per}")
    print(f"  info  dispute tx {gas_dispute(p2):,} gas; answer tx {gas_answer(p2):,} "
          f"gas (t=2); reference CPU {sum(r.cpu_seconds.values()):.1f}s")
    print("protocol.py: all self-tests passed.")


if __name__ == "__main__":
    _selftest()
