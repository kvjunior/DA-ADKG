"""
network.py -- Discrete-event asynchronous driver for DA-ADKG, v2.

Reference: Gao et al., "DA-ADKG" (TDSC-2025-08-1249, major revision).  Change
IDs N1-N8 and findings F# refer to research/D7_code_review_and_change_plan.md
and research/D5_D6_*.md.

WHAT CHANGED AGAINST v1 (and why)
  N1  No oracle barriers.  v1 advanced a phase when every message it KNEW
      would arrive had arrived, which hid the QUAL flaw (F1) and hung forever
      when one share was withheld (D7, R4).  v2 is driven by the ledger: a
      simulated chain produces blocks at a fixed interval, every transaction
      is included in the first block after it reaches the chain, the
      contract (protocol.py) accepts it only inside its window, and nodes act
      when they OBSERVE that a window has closed on the FINALIZED chain (F12).
      Late messages are simply late; partitions can now change QUAL.
  N2  Disputes, complaints, answers, PoK fraud proofs and public-key votes are
      real transactions with latency, inclusion and finality.
  N3  Communication is measured from the actual serialised messages: bytes
      SENT and RECEIVED per node (the metric of Das'23 Table 3 and H2CT Table
      III), total network bytes, ledger WRITES (calldata) and ledger READS
      (what every full node must download), reported separately (E054).  The
      v1 "broadcast-once" convention, which undercounted the commitment
      vectors by a factor of n (D7, R5), is gone.
  N4  The chain is an explicit, declared profile (block time, finality
      depth).  Each window must outlast "finality + network delay + local
      computation", so completion time is governed by the window plan; on a
      chain with slow finality the protocol is slow, and the paper must say
      so (F12).
  N5  Computation is not assumed.  Every elliptic-curve operation the real
      code performs is COUNTED per node (OpCounter), and priced with unit
      costs MEASURED on the running machine (calibrate_unit_costs), or with
      unit costs the authors measure for an optimised library and pass in.
      The cost source is stored in every result.
  N6  The latency model (log-normal, mu = 4.2, sigma = 0.8 in log-ms) is
      kept as this artifact's own stated assumption; it is NOT attributed to
      any prior paper (F13).
  N7  The large-n estimator uses operation counts FITTED to instrumented runs
      at small n (exact quadratics in t) and is validated against the event
      simulator; it covers the honest/crash case, while adversarial
      behaviour is evaluated with the full simulator.
  N8  Partitions, outbound delays and crashes act on real scheduling, so
      they can exclude honest dealers when they outlast a window -- the
      partial-synchrony behaviour the revised model must state.

MODELLING ASSUMPTIONS (state them in the revised methods section)
  (M1) Point-to-point latency ~ LogNormal(mu, sigma) milliseconds (defaults
       4.2 / 0.8: median ~67 ms), plus transmission time through a per-node
       FIFO uplink of `bandwidth_bps`.  Chain-to-node and node-to-chain hops
       use the same latency law.
  (M2) Blocks every `block_time` seconds; a block is final `finality_blocks`
       later.  Transactions are included in the first block at or after
       their arrival (no congestion model: block gas limits are not
       binding at the sizes evaluated, cf. EthDKG's 15 M limit, E042).
  (M3) A node performs one action at a time; an action's duration is its
       counted operations times the unit costs.  Field arithmetic
       (polynomial evaluation, Lagrange coefficients) and serialisation
       other than point decompression are not counted.
  (M4) Payload bytes only (no TCP/TLS/framing overhead unless
       `msg_overhead_bytes` is set).
  (M5) The synchrony bound Delta used to size windows and deadlines is the
       `latency_quantile` (default 0.999) quantile of the latency law, times
       `window_safety`.  A log-normal law has no hard bound, so an honest
       message can still miss a window with small probability; the paper
       should state Delta as such a quantile (partial synchrony, F12).

Every number in the revised manuscript must come from fresh runs of this
module (and must report the unit-cost source), never from the previously
published tables.

Run `python3 network.py` for the self-test suite.
"""

from __future__ import annotations

import heapq
import itertools
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import crypto as _crypto
import protocol as _protocol
from crypto import G, Q, point_mul, ser_point_evm
from protocol import (
    KeyShare,
    Participant,
    SmartContract,
    Strategy,
    SystemParams,
    Window,
    WindowSchedule,
    calldata_bytes,
    gas_anchor,
    gas_register,
    gas_submit_key,
    reconstruct_secret,
)

CONTRACT = 0      # the chain's index in the latency model (participants: 1..n)


# ==========================================================================
# 1. Chain profiles and network timing.
# ==========================================================================


@dataclass(frozen=True)
class ChainProfile:
    """The ledger DA-ADKG runs on.  Values are deployment assumptions to be
    stated in the paper; they are not measurements of this artifact."""

    name: str
    block_time: float          # seconds between blocks
    finality_blocks: int       # blocks after which a block is final

    @property
    def finality_time(self) -> float:
        return self.block_time * self.finality_blocks


# A BFT-finality chain (e.g. a Tendermint-style consortium chain), 1 s blocks.
BFT_CHAIN = ChainProfile("bft-1s", block_time=1.0, finality_blocks=1)
# Ethereum L1 (proof of stake), conservative profile.  Sources, checked
# 12 Oct 2026: ethereum.org/developers/docs/consensus-mechanisms/pos (page
# updated 31 Aug 2026): "slots (12 seconds) and epochs (32 slots)";
# checkpoints are at the start of each epoch.  eth2book.info (Casper FFG):
# "it takes 12.8 minutes, two epochs, to finalise a checkpoint".  A block in
# epoch e becomes final with the checkpoint of epoch e+1, i.e. 64 to 96
# slots after inclusion depending on its position in the epoch [derived].
# The worst case under normal operation (96 slots) is used, so Ethereum
# completion times are upper bounds; missed slots are ignored.  EIP-7782
# (6 s slots) was a Draft with no scheduled fork at the time of checking.
ETHEREUM_L1 = ChainProfile("ethereum-l1", block_time=12.0, finality_blocks=96)


@dataclass(frozen=True)
class NetTiming:
    lat_mu: float = 4.2               # log-normal parameters, log-milliseconds
    lat_sigma: float = 0.8
    bandwidth_bps: float = 100e6      # per-node uplink
    msg_overhead_bytes: int = 0       # transport framing, if modelled
    latency_quantile: float = 0.999   # synchrony bound used to size windows
    window_safety: float = 2.0        # multiplier on every window estimate

    def sample_latency(self, rng: random.Random) -> float:
        return math.exp(rng.gauss(self.lat_mu, self.lat_sigma)) / 1000.0

    def latency_bound(self) -> float:
        """The `latency_quantile` quantile of the latency law (seconds)."""
        z = statistics.NormalDist().inv_cdf(self.latency_quantile)
        return math.exp(self.lat_mu + self.lat_sigma * z) / 1000.0


DEFAULT_NET = NetTiming()


# ==========================================================================
# 2. Operation counting and unit costs (N5).
# ==========================================================================

OP_KINDS = ("mul", "mul_small", "dmul", "add", "inv", "sqrt",
            "keccak_call", "keccak_byte")


@dataclass(frozen=True)
class UnitCosts:
    """Seconds per counted operation.  `source` must say where the numbers
    come from (e.g. "measured: pure-Python reference on <host>")."""

    mul: float          # full-length scalar multiplication (Jacobian)
    mul_small: float    # scalar multiplication by a scalar < 2^64
    dmul: float         # double-scalar multiplication a*P + b*Q
    add: float          # point addition (Jacobian)
    inv: float          # Jacobian -> affine conversion (one inversion)
    sqrt: float         # point decompression square root
    keccak_call: float  # per Keccak-256 call
    keccak_byte: float  # per hashed byte
    source: str

    def scaled(self, factor: float, source: str) -> "UnitCosts":
        """All costs divided by `factor` (e.g. a MEASURED speed-up of an
        optimised library over the reference); the caller states the source."""
        return UnitCosts(*(getattr(self, k) / factor for k in OP_KINDS), source=source)

    def seconds(self, counts: Counter) -> float:
        return sum(counts.get(k, 0) * getattr(self, k) for k in OP_KINDS)


def calibrate_unit_costs(reps: int = 60, seed: int = 7) -> UnitCosts:
    """Measure every unit cost on THIS machine with the pure-Python
    reference implementation (crypto.py).  Medians over `reps` runs."""
    rng = random.Random(seed)
    P_, Q_ = _crypto.P, _crypto.Q
    pts = [_crypto._to_j(point_mul(1 + rng.randrange(Q_ - 1), G)) for _ in range(8)]

    def med(fn) -> float:
        xs = []
        for i in range(reps):
            t0 = time.perf_counter()
            fn(i)
            xs.append(time.perf_counter() - t0)
        return statistics.median(xs)

    big = [1 + rng.randrange(Q_ - 1) for _ in range(reps)]
    small = [rng.randrange(2, 512) for _ in range(reps)]
    blob = bytes(range(256)) * 4
    return UnitCosts(
        mul=med(lambda i: _crypto._j_mul(big[i], pts[i % 8], P_, Q_)),
        mul_small=med(lambda i: _crypto._j_mul(small[i], pts[i % 8], P_, Q_)),
        dmul=med(lambda i: _crypto._j_double_mul(big[i], pts[i % 8], big[-i - 1],
                                                 pts[(i + 3) % 8], P_, Q_)),
        add=med(lambda i: _crypto._j_add(pts[i % 8], pts[(i + 1) % 8], P_)),
        inv=med(lambda i: _crypto._from_j(pts[i % 8], P_)),
        sqrt=med(lambda i: _crypto._sqrt_3mod4(big[i] % P_, P_)),
        keccak_call=med(lambda i: _crypto.keccak256(b"")),
        keccak_byte=med(lambda i: _crypto.keccak256(blob)) / len(blob),
        source="measured: pure-Python reference implementation (crypto.py) on this machine",
    )


class OpCounter:
    """Counts the elliptic-curve and hashing work the REAL code performs,
    attributed to the current actor.  Only top-level primitive calls are
    counted (a scalar multiplication's internal additions are part of its
    unit cost).  Installs wrappers on crypto's internals for the duration of
    a `with` block and restores them afterwards."""

    _TARGETS = ("_j_mul", "_j_double_mul", "_j_add", "_from_j", "_sqrt_3mod4",
                "keccak256")

    def __init__(self) -> None:
        self.counts: Dict[int, Counter] = defaultdict(Counter)
        self.actor: int = -1
        self._depth = 0
        self._saved: Dict[str, Callable] = {}
        self._saved_proto_keccak: Optional[Callable] = None

    def __enter__(self) -> "OpCounter":
        oc = self
        orig = {name: getattr(_crypto, name) for name in self._TARGETS}
        self._saved = orig

        def wrap(name: str, classify: Callable[..., Tuple[str, int]]):
            f = orig[name]

            def w(*a, **k):
                if oc._depth == 0:
                    kind, amount = classify(*a, **k)
                    c = oc.counts[oc.actor]
                    c[kind] += amount
                    if name == "keccak256":
                        c["keccak_byte"] += sum(len(x) for x in a)
                oc._depth += 1
                try:
                    return f(*a, **k)
                finally:
                    oc._depth -= 1
            return w

        def mul_kind(k, *_a, **_kw):
            n = _a[1] if len(_a) > 1 else Q
            return ("mul_small" if (k % n).bit_length() <= 64 else "mul", 1)

        setattr(_crypto, "_j_mul", wrap("_j_mul", mul_kind))
        setattr(_crypto, "_j_double_mul", wrap("_j_double_mul", lambda *a, **k: ("dmul", 1)))
        setattr(_crypto, "_j_add", wrap("_j_add", lambda *a, **k: ("add", 1)))
        setattr(_crypto, "_from_j", wrap("_from_j", lambda *a, **k: ("inv", 1)))
        setattr(_crypto, "_sqrt_3mod4", wrap("_sqrt_3mod4", lambda *a, **k: ("sqrt", 1)))
        setattr(_crypto, "keccak256", wrap("keccak256", lambda *a, **k: ("keccak_call", 1)))
        self._saved_proto_keccak = _protocol.keccak256
        _protocol.keccak256 = _crypto.keccak256
        return self

    def __exit__(self, *exc) -> None:
        for name, f in self._saved.items():
            setattr(_crypto, name, f)
        _protocol.keccak256 = self._saved_proto_keccak

    def snapshot(self, actor: int) -> Counter:
        return Counter(self.counts[actor])


# ==========================================================================
# 3. Network conditions.
# ==========================================================================


@dataclass(frozen=True)
class Partition:
    """`group` is cut off from everyone else -- including the chain -- during
    [start, end); crossing messages are held until the heal."""

    start: float
    end: float
    group: FrozenSet[int]

    def separates(self, a: int, b: int) -> bool:
        return (a in self.group) != (b in self.group)


@dataclass(frozen=True)
class AdversaryProfile:
    """Protocol behaviour (protocol.Strategy) plus network behaviour."""

    strategy: Strategy = Strategy.HONEST
    targets: Tuple[int, ...] = ()
    delay_factor: float = 1.0          # multiplies this node's outbound latency
    crash_time: Optional[float] = None  # node stops acting from this time on


HONEST = AdversaryProfile()


class LatencyModel:
    """Delivery times with per-node FIFO uplinks, adversarial delay and
    partitions.  Deterministic given its RNG."""

    def __init__(self, net: NetTiming, rng: random.Random,
                 profiles: Dict[int, AdversaryProfile],
                 partitions: Sequence[Partition]) -> None:
        self.net, self.rng = net, rng
        self.profiles = profiles
        self.partitions = list(partitions)
        self.uplink_free: Dict[int, float] = defaultdict(float)

    def _hold(self, src: int, dst: int, send: float, arr: float) -> float:
        for pt in self.partitions:
            if pt.separates(src, dst) and send < pt.end and arr > pt.start:
                arr = max(arr, pt.end + self.net.sample_latency(self.rng))
        return arr

    def send(self, src: int, dst: int, now: float, nbytes: int) -> float:
        """Arrival time of a message of `nbytes` sent by src at `now`."""
        size = nbytes + self.net.msg_overhead_bytes
        start = max(now, self.uplink_free[src])
        done = start + size * 8.0 / self.net.bandwidth_bps
        self.uplink_free[src] = done
        d = self.net.sample_latency(self.rng)
        prof = self.profiles.get(src)
        if prof is not None:
            d *= prof.delay_factor
        return self._hold(src, dst, start, done + d)

    def observe(self, dst: int, produced: float) -> float:
        """When node dst sees chain data produced at `produced`."""
        return self._hold(CONTRACT, dst, produced,
                          produced + self.net.sample_latency(self.rng))


# ==========================================================================
# 4. Event engine.
# ==========================================================================


@dataclass(order=True)
class _Event:
    time: float
    seq: int
    cb: Callable[[], None] = field(compare=False)


class Sim:
    """Time-ordered event queue; ties broken by insertion order."""

    def __init__(self) -> None:
        self._q: List[_Event] = []
        self._seq = itertools.count()
        self.now = 0.0

    def at(self, t: float, cb: Callable[[], None]) -> None:
        heapq.heappush(self._q, _Event(max(t, self.now), next(self._seq), cb))

    def run(self, until: Optional[Callable[[], bool]] = None) -> None:
        while self._q:
            ev = heapq.heappop(self._q)
            self.now = ev.time
            ev.cb()
            if until is not None and until():
                return


# ==========================================================================
# 5. Exact message sizes (validated against the real serialisers).
# ==========================================================================


def _enc_len(*part_lengths: int) -> int:
    return sum(4 + x for x in part_lengths)


def share_payload_bytes(p: SystemParams) -> int:
    """len(SharePayload.to_bytes()) -- deterministic for given t."""
    m = p.degree + 1
    commits = _enc_len(*([33] * m))
    pok = _enc_len(_enc_len(*([33] * m)), _enc_len(*([32] * m)), _enc_len(*([32] * m)))
    return _enc_len(4, 4, commits, pok, 32, 32, 65)


def key_share_bytes() -> int:
    """len(KeyShare.to_bytes())."""
    schnorr = _enc_len(33, 32)
    return _enc_len(4, 33, 33, schnorr, schnorr)


# ==========================================================================
# 6. Window planning (N4).
# ==========================================================================


@dataclass(frozen=True)
class PhaseCompute:
    """Per-node computation (seconds) for each action, used to size windows."""

    register: float
    deal: float
    verify: float
    complain: float
    answer: float
    dispute: float
    key_share: float
    key_verify: float


def recommended_windows(p: SystemParams, chain: ChainProfile, net: NetTiming,
                        compute: PhaseCompute) -> WindowSchedule:
    """Window lengths (blocks) that let every HONEST action complete within
    its window when the network is synchronous with delay bound
    net.latency_bound() (post-GST).  Every window includes the time to
    OBSERVE that the previous one closed on the finalized chain."""
    L = net.latency_bound()
    F = chain.finality_time
    bt = chain.block_time
    bw = net.bandwidth_bps
    share_tx = p.n * share_payload_bytes(p) * 8.0 / bw     # dealer uplink
    key_tx = p.n * key_share_bytes() * 8.0 / bw
    observe = F + L
    submit = L + bt
    secs = {
        "registration": compute.register + submit,
        "dealing": observe + compute.deal + submit,
        "complaint": observe + share_tx + L + compute.verify + compute.complain + submit,
        "answer": observe + compute.answer + submit,
        "dispute": observe + compute.dispute + submit,
        "key_derivation": observe + compute.key_share + key_tx + L
        + compute.key_verify + submit + bt,
    }
    blocks = {k: max(1, math.ceil(v * net.window_safety / bt)) for k, v in secs.items()}
    return WindowSchedule(**blocks)


# ==========================================================================
# 7. The asynchronous driver.
# ==========================================================================


@dataclass
class CommStats:
    sent: Dict[int, int]                 # payload bytes sent per node (P2P + tx)
    received: Dict[int, int]             # payload bytes received per node (P2P)
    p2p_total: int                       # all point-to-point bytes
    ledger_writes: int                   # calldata of all submitted transactions
    by_kind: Dict[str, int]

    def per_node_sent_mb(self) -> Tuple[float, float]:
        vals = list(self.sent.values()) or [0]
        return statistics.mean(vals) / 2**20, max(vals) / 2**20

    @property
    def ledger_reads_per_node(self) -> int:
        """Every full node downloads all calldata (E054: network READS)."""
        return self.ledger_writes


@dataclass
class AsyncResult:
    success: bool
    reason: str
    completion_time: Optional[float]     # PK finalized (simulated seconds)
    qual: List[int]
    complete_holders: List[int]
    honest_complete_holders: List[int]
    public_key: Optional[bytes]
    complaints: Dict[Tuple[int, int], bool]
    disputes: List[dict]
    gas_total: int
    gas_by_category: Dict[str, int]
    tx_accepted: Dict[str, int]
    tx_rejected: Dict[str, int]
    treasury: float
    comm: CommStats
    compute_seconds: Dict[int, float]    # per node, whole run
    ops: Dict[int, Counter]              # per node operation counts
    ops_by_action: Dict[str, Counter]    # summed over nodes, per action kind
    windows: WindowSchedule
    window_times: Dict[str, Tuple[float, float]]
    chain: ChainProfile
    unit_costs: UnitCosts
    seed: int
    events: List[dict] = field(default_factory=list)   # contract event log
    audit: Optional[dict] = None


_TX_METHODS = frozenset({"register", "anchor", "complain", "answer", "dispute",
                         "pok_fraud", "submit_vrow", "submit_public_key"})


class _ChainProxy:
    """What a participant sees as `contract`: reads go to the real contract
    state; transaction methods are queued for inclusion by the chain model
    (optimistic `True` return, the outcome is recorded at inclusion)."""

    def __init__(self, driver: "_Driver", actor: int) -> None:
        self._driver, self._actor = driver, actor

    def __getattr__(self, name: str):
        if name in _TX_METHODS:
            return lambda *a, **k: self._driver.submit_tx(self._actor, name, a, k)
        return getattr(self._driver.contract, name)


class _Driver:
    def __init__(self, params: SystemParams, profiles: Dict[int, AdversaryProfile],
                 partitions: Sequence[Partition], chain: ChainProfile,
                 net: NetTiming, unit: UnitCosts, seed: int,
                 counter: OpCounter, compute: Optional[PhaseCompute] = None) -> None:
        self.p, self.chain, self.net, self.unit = params, chain, net, unit
        self.compute = compute
        self.profiles = profiles
        self.rng = random.Random(seed)
        self.lat = LatencyModel(net, self.rng, profiles, partitions)
        self.sim = Sim()
        self.ops = counter
        self.contract = SmartContract(params)
        self.parts = {i: Participant(i, params, profiles.get(i, HONEST).strategy,
                                     profiles.get(i, HONEST).targets)
                      for i in range(1, params.n + 1)}
        self.proxy = {i: _ChainProxy(self, i) for i in self.parts}
        self.busy_until: Dict[int, float] = defaultdict(float)
        self.pending_tx: List[Tuple[float, int, int, str, tuple, dict]] = []
        self._txseq = itertools.count()
        self.tx_ok: Counter = Counter()
        self.tx_bad: Counter = Counter()
        self.sent: Dict[int, int] = defaultdict(int)
        self.recv: Dict[int, int] = defaultdict(int)
        self.by_kind: Counter = Counter()
        self.p2p_total = 0
        self.ledger_writes = 0
        self.ops_by_action: Dict[str, Counter] = defaultdict(Counter)
        self.compute_s: Dict[int, float] = defaultdict(float)
        self.mail: Dict[int, Dict[int, Optional[bytes]]] = defaultdict(dict)
        self.decided: set = set()
        self.verified: Dict[int, set] = defaultdict(set)
        self.seen_dealing_closed: set = set()
        self.pending_key_blobs: Dict[int, List[bytes]] = defaultdict(list)
        self.key_shares: Dict[int, Dict[int, KeyShare]] = defaultdict(dict)
        self.key_good: Dict[int, List[KeyShare]] = defaultdict(list)
        self.key_submitted: set = set()
        self.finalized_at: Optional[float] = None
        self.qual: Optional[List[int]] = None
        self.done = False
        self.end_reason: Optional[str] = None

    # -- time <-> height ----------------------------------------------------

    def block_time(self, h: int) -> float:
        return h * self.chain.block_time

    def height_for_arrival(self, t: float) -> int:
        return max(0, math.ceil(t / self.chain.block_time - 1e-12))

    def finalized_time(self, h: int) -> float:
        return self.block_time(h + self.chain.finality_blocks)

    def window_closed_seen(self, w: Window, node: int) -> float:
        end = self.contract.bounds[w][1]
        return self.lat.observe(node, self.finalized_time(end - 1))

    # -- participants ----------------------------------------------------------

    def alive(self, i: int, t: float) -> bool:
        prof = self.profiles.get(i, HONEST)
        return prof.crash_time is None or t < prof.crash_time

    def act(self, i: int, kind: str, fn: Callable[[], None], at: float) -> None:
        """Run a participant action when the node is free; its duration is its
        counted operations times the unit costs; messages and transactions it
        emits leave when the action ends."""
        def run() -> None:
            if not self.alive(i, self.sim.now):
                return
            start = max(self.sim.now, self.busy_until[i])
            if start > self.sim.now:
                self.sim.at(start, run)
                return
            before = self.ops.snapshot(i)
            self.ops.actor = i
            self._outbox: List[Tuple[str, tuple]] = []
            fn()
            self.ops.actor = -1
            delta = self.ops.snapshot(i) - before
            dur = self.unit.seconds(delta)
            self.compute_s[i] += dur
            self.ops_by_action[kind] += delta
            end = start + dur
            self.busy_until[i] = end
            for what, args in self._outbox:
                if what == "tx":
                    self._dispatch_tx(i, end, *args)
                else:
                    self._dispatch_msg(i, end, *args)
        self.sim.at(at, run)

    def submit_tx(self, actor: int, method: str, a: tuple, k: dict):
        self._outbox.append(("tx", (method, a, k)))
        return True if method != "dispute" and method != "pok_fraud" else "submitted"

    def send_msg(self, src: int, dst: int, kind: str, blob: Optional[bytes],
                 deliver: Callable[[Optional[bytes]], None]) -> None:
        self._outbox.append(("msg", (dst, kind, blob, deliver)))

    def _dispatch_msg(self, src: int, now: float, dst: int, kind: str,
                      blob: Optional[bytes], deliver) -> None:
        if blob is None:
            return
        if dst == src:                      # local hand-off, not network traffic
            self.sim.at(now, lambda: deliver(blob))
            return
        n = len(blob)
        self.sent[src] += n
        self.p2p_total += n
        self.by_kind[kind] += n
        arr = self.lat.send(src, dst, now, n)

        def on_arrival() -> None:
            self.recv[dst] += n
            deliver(blob)
        self.sim.at(arr, on_arrival)

    def _dispatch_tx(self, src: int, now: float, method: str, a: tuple, k: dict) -> None:
        nbytes = calldata_bytes(method, self.p)
        self.sent[src] += nbytes
        self.ledger_writes += nbytes
        self.by_kind["tx:" + method] += nbytes
        arr = self.lat.send(src, CONTRACT, now, nbytes)
        h = self.height_for_arrival(arr)

        def include() -> None:
            self.contract.advance_to(max(self.contract.height, h))
            n_rej = self.contract.gas.tx_count.get("rejected", 0)
            res = getattr(self.contract, method)(*a, **k)
            rejected = (self.contract.gas.tx_count.get("rejected", 0) > n_rej
                        or res is False or res == "rejected")
            (self.tx_bad if rejected else self.tx_ok)[method] += 1
            if method == "submit_public_key" and self.contract.public_key is not None \
                    and self.finalized_at is None:
                self.finalized_at = self.finalized_time(self.contract.height)
        self.sim.at(self.block_time(h), include)

    # -- protocol schedule ----------------------------------------------------

    def start(self) -> None:
        c = self.contract
        for i, part in self.parts.items():
            self.act(i, "register",
                     lambda part=part, i=i: self.proxy[i].register(*part.registration_tx()),
                     0.0)
        # each node acts when it SEES the relevant window closed
        for i in self.parts:
            self.sim.at(self.window_closed_seen(Window.REGISTRATION, i),
                        lambda i=i: self.act(i, "deal", lambda i=i: self._deal(i), self.sim.now))
            self.sim.at(self.window_closed_seen(Window.DEALING, i),
                        lambda i=i: self._dealing_closed(i))
            self.sim.at(self.window_closed_seen(Window.COMPLAINT, i),
                        lambda i=i: self.act(i, "answer",
                                             lambda i=i: self.parts[i].act_in_answer_window(self.proxy[i]),
                                             self.sim.now))
            self.sim.at(self.window_closed_seen(Window.ANSWER, i),
                        lambda i=i: self.act(i, "dispute",
                                             lambda i=i: self.parts[i].act_in_dispute_window(self.proxy[i]),
                                             self.sim.now))
        # the chain fixes QUAL at the first block after the DISPUTE window
        q_h = c.bounds[Window.DISPUTE][1]
        self.sim.at(self.block_time(q_h), self._close_qual)
        for i in self.parts:
            self.sim.at(self.lat.observe(i, self.finalized_time(q_h)),
                        lambda i=i: self.act(i, "key_share", lambda i=i: self._key_share(i),
                                             self.sim.now))
        # end of run: the KEY_DERIVATION window closes
        k_end = c.bounds[Window.KEY_DERIVATION][1]
        self.sim.at(self.finalized_time(k_end), self._stop)

    def _deal(self, i: int) -> None:
        part = self.parts[i]
        directory = {j: r.pk for j, r in self.contract.registry.items()}
        if i not in directory or len(directory) < self.p.n - self.p.t:
            # (protocol D9) the participant set is the registered set; a dealer
            # deals only if it registered and at least n - t parties did
            return
        out = part.deal(directory)
        if out is None:
            return
        mr, pok_hash, msgs = out
        self.submit_tx(i, "anchor", (i, mr, pok_hash), {})
        for j, blob in msgs.items():
            self.send_msg(i, j, "share", blob,
                          lambda b, i=i, j=j: self._share_arrived(j, i, b))

    def _share_arrived(self, j: int, i: int, blob: Optional[bytes]) -> None:
        self.mail[j][i] = blob
        if j in self.seen_dealing_closed:
            self._verify_one(j, i)

    def _verify_one(self, j: int, i: int) -> None:
        """Verify one dealer's share as soon as it can be checked (its anchor
        is final and the share has arrived); a node therefore has only the
        cheap complaint step left when its deadline comes."""
        part = self.parts[j]
        if i in self.verified[j] or i not in self.contract.anchors or i not in self.mail[j]:
            return
        self.verified[j].add(i)

        def run() -> None:
            if part.participates:
                part.receive_share(i, self.mail[j][i], self.proxy[j])
            self._maybe_decide(j)
        self.act(j, "verify", run, self.sim.now)

    def _dealing_closed(self, j: int) -> None:
        self.seen_dealing_closed.add(j)
        for i in sorted(self.mail[j]):
            self._verify_one(j, i)
        # deadline: leave time for the complaint action, submission and
        # inclusion before the COMPLAINT window closes, with the same safety
        # factor the window plan uses (latency is a quantile, not a hard bound)
        work = self.net.latency_bound()
        if self.compute is not None:
            work += self.compute.complain + self.compute.dispute
        margin = self.net.window_safety * work + self.chain.block_time
        deadline = self.block_time(self.contract.bounds[Window.COMPLAINT][1]) - margin
        self.sim.at(max(self.sim.now, deadline), lambda: self._decide(j))
        self._maybe_decide(j)

    def _maybe_decide(self, j: int) -> None:
        if j not in self.seen_dealing_closed:
            return
        anchored = set(self.contract.anchors)
        if anchored and anchored <= self.verified[j]:
            self._decide(j)

    def _decide(self, j: int) -> None:
        if j in self.decided:
            return
        self.decided.add(j)
        part = self.parts[j]

        def run() -> None:
            if not part.participates:
                return
            for d in sorted(self.contract.anchors):
                if d not in self.verified[j]:          # never arrived: unusable
                    part.receive_share(d, None, self.proxy[j])
            part.act_in_complaint_window(self.proxy[j])
        self.act(j, "complain", run, self.sim.now)

    def _close_qual(self) -> None:
        c = self.contract
        c.advance_to(max(c.height, c.bounds[Window.DISPUTE][1]))
        self.qual = c.close_qualification()

    def _key_share(self, i: int) -> None:
        part = self.parts[i]
        if not part.participates or self.qual is None or len(self.qual) < self.p.min_qual:
            return
        if not part.finalize(self.qual):
            return
        ks = part.key_share()
        blob = ks.to_bytes()
        for j in self.parts:
            self.send_msg(i, j, "key_share", blob,
                          lambda b, j=j: self._key_share_arrived(j, b))
        for early in self.pending_key_blobs.pop(i, []):   # arrived before we finalized
            self.sim.at(self.sim.now, lambda b=early, i=i: self._key_share_arrived(i, b))

    def _key_share_arrived(self, j: int, blob: bytes) -> None:
        part = self.parts[j]
        if not part.participates:
            return
        if part.final_share is None:
            if part.qual is None:                 # not finalized yet: buffer
                self.pending_key_blobs[j].append(blob)
            return

        def run() -> None:
            try:
                ks = KeyShare.from_bytes(blob)
            except ValueError:
                return
            if ks.index in self.key_shares[j]:
                return
            self.key_shares[j][ks.index] = ks
            if j in self.key_submitted or not part.verify_key_share(ks):
                return
            self.key_good[j].append(ks)
            if len(self.key_good[j]) >= self.p.recon_threshold:
                pk = part.public_key_from_verified(self.key_good[j])
                self.key_submitted.add(j)
                self.submit_tx(j, "submit_public_key", (j, ser_point_evm(pk)), {})
        self.act(j, "key_verify", run, self.sim.now)

    def _stop(self) -> None:
        self.done = True


def run_dkg_async(params: SystemParams,
                  adversaries: Optional[Dict[int, AdversaryProfile]] = None,
                  partitions: Sequence[Partition] = (),
                  chain: ChainProfile = BFT_CHAIN,
                  net: NetTiming = DEFAULT_NET,
                  unit_costs: Optional[UnitCosts] = None,
                  windows: Optional[WindowSchedule] = None,
                  cost_model: Optional["CostModel"] = None,
                  seed: int = 0,
                  audit_reconstruct: bool = False) -> AsyncResult:
    """Run DA-ADKG v2 end-to-end (real cryptography) under the chain/network
    model.  If `windows` is None they are planned by recommended_windows()
    from `cost_model` (fitted if not given)."""
    profiles = dict(adversaries or {})
    byz = {i for i, pr in profiles.items() if pr.strategy is not Strategy.HONEST}
    if len(byz) > params.t:
        raise ValueError("adversary exceeds Byzantine threshold t")
    unit = unit_costs or default_unit_costs()
    compute: Optional[PhaseCompute] = None
    if windows is None or cost_model is not None:
        compute = (cost_model or default_cost_model()).phase_compute(params, unit)
    if windows is None:
        windows = recommended_windows(params, chain, net, compute)
    params = replace(params, windows=windows)

    with OpCounter() as counter:
        d = _Driver(params, profiles, partitions, chain, net, unit, seed, counter, compute)
        d.start()
        d.sim.run(until=lambda: d.done)

    c = d.contract
    if d.qual is None:
        c.advance_to(max(c.height, c.bounds[Window.DISPUTE][1]))
        d.qual = c.close_qualification()
    honest = [i for i in d.parts if i not in byz]
    complete = sorted(i for i, pt in d.parts.items()
                      if pt.participates and pt.final_share is not None and d.alive(i, d.sim.now))
    honest_complete = [i for i in complete if i in honest]
    pk = c.public_key

    if len(c.registry) < params.n - params.t:
        ok, reason = False, "registration-below-n-t"
    elif len(d.qual) < params.min_qual:
        ok, reason = False, "qual-below-t+1"
    elif len(honest_complete) < params.recon_threshold:
        ok, reason = False, "honest-holders-below-2t+1"
    elif pk is None:
        ok, reason = False, "public-key-not-finalized"
    elif any(d.parts[i].public_key is not None and ser_point_evm(d.parts[i].public_key) != pk
             for i in honest_complete):
        ok, reason = False, "public-key-disagreement"
    else:
        ok, reason = True, "ok"

    audit = None
    if ok and audit_reconstruct:
        contribs = {i: d.parts[i].final_share for i in honest_complete}
        s, _ = reconstruct_secret(contribs, d.qual, d.parts[honest_complete[0]].commits_of,
                                  params)
        truth = sum(d.parts[x]._f.coeffs[0] for x in d.qual) % Q
        audit = {"secret_matches_dealer_sum": s == truth,
                 "public_key_is_g_to_s": ser_point_evm(point_mul(s, G)) == pk}
        if not all(audit.values()):
            ok, reason = False, "audit-failed"

    bt = chain.block_time
    wt = {w.name: (c.bounds[w][0] * bt, c.bounds[w][1] * bt) for w in c.bounds}
    return AsyncResult(
        success=ok, reason=reason,
        completion_time=d.finalized_at if ok else None,
        qual=list(d.qual), complete_holders=complete,
        honest_complete_holders=honest_complete, public_key=pk,
        complaints=dict(c.complaints), disputes=list(c.disputes),
        gas_total=c.gas.total, gas_by_category=dict(c.gas.by_category),
        tx_accepted=dict(d.tx_ok), tx_rejected=dict(d.tx_bad),
        treasury=c.treasury,
        comm=CommStats(sent={i: d.sent.get(i, 0) for i in d.parts},
                       received={i: d.recv.get(i, 0) for i in d.parts},
                       p2p_total=d.p2p_total, ledger_writes=d.ledger_writes,
                       by_kind=dict(d.by_kind)),
        compute_seconds={i: d.compute_s.get(i, 0.0) for i in d.parts},
        ops={i: Counter(counter.counts.get(i, Counter())) for i in d.parts},
        ops_by_action={k: Counter(v) for k, v in d.ops_by_action.items()},
        windows=windows, window_times=wt, chain=chain, unit_costs=unit,
        seed=seed, events=list(c.events), audit=audit)


# ==========================================================================
# 8. Cost model (fitted, validated) and the large-n estimator (N7).
# ==========================================================================

_ACTIONS = ("register", "deal", "verify", "complain", "answer", "dispute", "key_share",
            "key_verify")


@dataclass(frozen=True)
class CostModel:
    """Per-node operation counts per action in the HONEST run, as exact
    quadratics in t:  count(t) = a*t^2 + b*t + c, fitted on instrumented
    runs.  `fit_ts` records the t values used.  'key_verify' is the total
    over the 2t+1 key shares a node verifies (the critical path)."""

    coeffs: Dict[str, Dict[str, Tuple[float, float, float]]]
    fit_ts: Tuple[int, ...]

    def counts(self, action: str, t: int) -> Counter:
        out = Counter()
        for kind, (a, b, c) in self.coeffs.get(action, {}).items():
            v = a * t * t + b * t + c
            out[kind] = max(0, round(v))
        return out

    def phase_compute(self, p: SystemParams, unit: UnitCosts) -> PhaseCompute:
        sec = {a: unit.seconds(self.counts(a, p.t)) for a in _ACTIONS}
        return PhaseCompute(register=sec["register"], deal=sec["deal"],
                            verify=sec["verify"], complain=sec["complain"],
                            answer=sec["answer"],
                            dispute=sec["dispute"], key_share=sec["key_share"],
                            key_verify=sec["key_verify"])


def _per_node_action_counts(r: AsyncResult, n: int) -> Dict[str, Counter]:
    """Average per-node counts per action (honest run: all nodes alike, except
    key_verify, where the critical path is the first 2t+1 shares)."""
    out = {}
    for a in _ACTIONS:
        tot = r.ops_by_action.get(a, Counter())
        out[a] = Counter({k: v / n for k, v in tot.items()})
    return out


def fit_cost_model(ts: Sequence[int] = (1, 2, 3), seed: int = 11,
                   unit: Optional[UnitCosts] = None) -> CostModel:
    """Run instrumented honest simulations at the given t (with generous fixed
    windows) and fit exact quadratics per (action, op kind)."""
    import numpy as np
    unit = unit or default_unit_costs()
    data: Dict[int, Dict[str, Counter]] = {}
    for t in ts:
        p = SystemParams(t=t)
        r = run_dkg_async(p, chain=BFT_CHAIN, net=DEFAULT_NET, unit_costs=unit,
                          windows=WindowSchedule(5, 30, 30, 30, 30, 30), seed=seed)
        if not r.success:
            raise RuntimeError(f"calibration run failed at t={t}: {r.reason}")
        data[t] = _per_node_action_counts(r, p.n)
    coeffs: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    X = np.array([[t * t, t, 1.0] for t in ts])
    for a in _ACTIONS:
        kinds = set().union(*(data[t][a].keys() for t in ts))
        coeffs[a] = {}
        for k in kinds:
            y = np.array([data[t][a].get(k, 0.0) for t in ts])
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            coeffs[a][k] = tuple(float(v) for v in sol)
    return CostModel(coeffs=coeffs, fit_ts=tuple(ts))


_DEFAULT_UNIT: Optional[UnitCosts] = None
_DEFAULT_MODEL: Optional[CostModel] = None


def default_unit_costs() -> UnitCosts:
    global _DEFAULT_UNIT
    if _DEFAULT_UNIT is None:
        _DEFAULT_UNIT = calibrate_unit_costs()
    return _DEFAULT_UNIT


def default_cost_model() -> CostModel:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = fit_cost_model()
    return _DEFAULT_MODEL


@dataclass
class Estimate:
    n: int
    completion_time: float
    windows: WindowSchedule
    per_node_sent_bytes: int
    p2p_total_bytes: int
    ledger_writes_bytes: int
    gas_total: int
    compute_per_node: PhaseCompute
    chain: ChainProfile
    unit_source: str


def estimate_performance(params: SystemParams, chain: ChainProfile = BFT_CHAIN,
                         net: NetTiming = DEFAULT_NET,
                         unit_costs: Optional[UnitCosts] = None,
                         cost_model: Optional[CostModel] = None) -> Estimate:
    """Honest-case analytic estimate for large n (no cryptography executed):
    completion = start of the KEY_DERIVATION window + observation + key-share
    exchange + verification + vote inclusion + finality; communication and
    gas from the exact message/transaction sizes."""
    unit = unit_costs or default_unit_costs()
    model = cost_model or default_cost_model()
    pc = model.phase_compute(params, unit)
    w = recommended_windows(params, chain, net, pc)
    bt = chain.block_time
    start_key = sum((w.registration, w.dealing, w.complaint, w.answer, w.dispute)) * bt
    Lm = math.exp(net.lat_mu) / 1000.0          # median latency: expected path
    t_vote_arrives = (start_key + chain.finality_time + Lm + pc.key_share
                      + params.n * key_share_bytes() * 8.0 / net.bandwidth_bps + Lm
                      + pc.key_verify + Lm)
    h_incl = math.ceil(t_vote_arrives / bt)
    completion = (h_incl + chain.finality_blocks) * bt
    n = params.n
    share_b, key_b = share_payload_bytes(params), key_share_bytes()
    tx_b = (calldata_bytes("register", params) + calldata_bytes("anchor", params)
            + calldata_bytes("submit_public_key", params))
    per_node = (n - 1) * (share_b + key_b) + tx_b      # self-messages are local
    p2p_total = n * (n - 1) * (share_b + key_b)
    ledger = n * tx_b
    conf = params.key_confirmations
    gas = n * (gas_register() + gas_anchor()) + (conf - 1) * gas_submit_key(False) + \
        gas_submit_key(True) + (n - conf) * gas_submit_key(False)
    return Estimate(n=n, completion_time=completion, windows=w,
                    per_node_sent_bytes=per_node, p2p_total_bytes=p2p_total,
                    ledger_writes_bytes=ledger, gas_total=gas,
                    compute_per_node=pc, chain=chain, unit_source=unit.source)


# ==========================================================================
# 9. Self-tests.
# ==========================================================================


def _selftest() -> None:  # pragma: no cover
    from protocol import run_dkg
    ok = lambda s: print(f"  PASS  {s}")
    S = Strategy
    t0 = time.perf_counter()
    unit = calibrate_unit_costs()
    print(f"  info  unit costs ({unit.source}): mul {unit.mul*1e3:.2f} ms, "
          f"dmul {unit.dmul*1e3:.2f} ms, small mul {unit.mul_small*1e3:.3f} ms, "
          f"sqrt {unit.sqrt*1e3:.3f} ms")
    global _DEFAULT_UNIT
    _DEFAULT_UNIT = unit
    model = fit_cost_model(unit=unit)
    globals()["_DEFAULT_MODEL"] = model
    ok(f"cost model fitted on instrumented runs t={model.fit_ts} "
       f"({time.perf_counter() - t0:.0f}s)")

    p2 = SystemParams(t=2)

    # 1. honest run: success, audit, all windows met, no rejected tx
    r = run_dkg_async(p2, unit_costs=unit, seed=1, audit_reconstruct=True)
    assert r.success and r.qual == list(range(1, 8)), r.reason
    assert r.audit == {"secret_matches_dealer_sum": True, "public_key_is_g_to_s": True}
    assert not r.tx_rejected and not r.complaints
    ok(f"honest t=2 on {r.chain.name}: success, audit OK, completion "
       f"{r.completion_time:.1f}s, windows {r.windows}")

    # 2. determinism (fixed unit costs): identical outcome, timing, bytes, ops
    r2 = run_dkg_async(p2, unit_costs=unit, seed=1)
    assert (r2.completion_time, r2.comm.sent, r2.gas_total) == \
        (r.completion_time, r.comm.sent, r.gas_total)
    assert r2.ops == r.ops
    ok("deterministic given seed and unit costs (timing, bytes, gas, op counts)")

    # 3. exact byte accounting against the real serialisers
    mean_sent, _ = r.comm.per_node_sent_mb()
    est = estimate_performance(p2, unit_costs=unit, cost_model=model)
    assert all(v == est.per_node_sent_bytes for v in r.comm.sent.values()), \
        (r.comm.sent, est.per_node_sent_bytes)
    assert r.comm.p2p_total == est.p2p_total_bytes
    assert r.comm.ledger_writes == est.ledger_writes_bytes
    assert r.gas_total == est.gas_total
    ok(f"bytes sent per node = analytic {est.per_node_sent_bytes:,} B exactly; "
       f"ledger writes and gas match the estimator exactly")

    # 4. async agrees with the synchronous reference on protocol outcomes
    cases = {
        "corrupt-all": {7: (S.CORRUPT_ALL, ())},
        "selective": {6: (S.SELECTIVE, (1, 2))},
        "withhold": {5: (S.WITHHOLD, (1,))},
        "withhold+answer": {5: (S.WITHHOLD_THEN_ANSWER, (1, 2))},
        "bad-answer": {5: (S.BAD_ANSWER, (1,))},
        "false-accuser": {3: (S.FALSE_ACCUSER, (2,))},
        "false-complainer": {7: (S.FALSE_COMPLAINER, (3,))},
        "bad-pok": {4: (S.BAD_POK, ())},
        "silent": {5: (S.SILENT, ())},
    }
    for name, byz in cases.items():
        rs = run_dkg(p2, byzantine=byz)
        ra = run_dkg_async(p2, adversaries={i: AdversaryProfile(s, t) for i, (s, t)
                                            in byz.items()}, unit_costs=unit, seed=3)
        assert (rs.success, rs.qual, sorted(rs.honest_complete_holders)) == \
            (ra.success, ra.qual, sorted(ra.honest_complete_holders)), name
        assert sorted(rs.complaints.items()) == sorted(ra.complaints.items()), name
    ok(f"async == sync outcomes for {len(cases)} adversary configurations "
       "(incl. omission: terminates, D7 R4 fixed)")

    # 5. partition shorter than the windows: same QUAL, later completion
    base = r
    short = [Partition(0.5, 2.0, frozenset({1, 2}))]
    rp = run_dkg_async(p2, partitions=short, unit_costs=unit, seed=1)
    assert rp.success and rp.qual == base.qual
    ok("short partition (1.5 s): QUAL unchanged, success")

    # 6. partition outlasting the DEALING window cuts dealer 1 off the chain:
    #    its anchor misses the window -> honest dealer excluded (partial
    #    synchrony: no guarantee before GST); the run still succeeds
    w = r.window_times
    long_ = [Partition(w["DEALING"][0] - 0.01, w["COMPLAINT"][1] + 1.0, frozenset({1}))]
    rl = run_dkg_async(p2, partitions=long_, unit_costs=unit, seed=1)
    assert 1 not in rl.qual and rl.tx_rejected.get("anchor", 0) == 1
    ok(f"long partition isolating node 1: excluded from QUAL ({rl.qual}); "
       f"success={rl.success} ({rl.reason})")

    # 7. windows too short -> late transactions rejected and reported
    tiny = WindowSchedule(1, 1, 1, 1, 1, 1)
    rt = run_dkg_async(p2, unit_costs=unit, seed=1, windows=tiny)
    assert not rt.success and sum(rt.tx_rejected.values()) > 0
    ok(f"undersized windows: run fails ({rt.reason}), {sum(rt.tx_rejected.values())} "
       "late transactions rejected (not hidden)")

    # 8. outbound delay adversary: honest outcome, measured slowdown or failure
    rd = run_dkg_async(p2, adversaries={7: AdversaryProfile(S.HONEST, delay_factor=20.0)},
                       unit_costs=unit, seed=1)
    ok(f"delay adversary x20: success={rd.success}, QUAL={rd.qual}")

    # 9. estimator vs simulator, and scaling
    est2 = estimate_performance(p2, unit_costs=unit, cost_model=model)
    assert abs(est2.completion_time - r.completion_time) <= 2 * BFT_CHAIN.block_time, \
        (est2.completion_time, r.completion_time)
    ok(f"estimator completion {est2.completion_time:.1f}s vs simulated "
       f"{r.completion_time:.1f}s (within 2 blocks)")
    print("  info  honest estimates (bft-1s chain, reference unit costs):")
    for t in (1, 5, 21, 42, 85):
        e = estimate_performance(SystemParams(t=t), unit_costs=unit, cost_model=model)
        print(f"        n={e.n:4d}  completion={e.completion_time:9.1f}s  "
              f"sent/node={e.per_node_sent_bytes/2**20:8.2f} MiB  "
              f"gas={e.gas_total/1e6:7.2f} M")
    e_eth = estimate_performance(SystemParams(t=42), chain=ETHEREUM_L1,
                                 unit_costs=unit, cost_model=model)
    print(f"  info  n=127 on {ETHEREUM_L1.name}: completion {e_eth.completion_time/60:.1f} min "
          f"(finality-gated windows, F12)")
    print("network.py: all self-tests passed.")


if __name__ == "__main__":
    _selftest()
