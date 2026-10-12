"""
crypto.py -- Cryptographic primitives for the DA-ADKG reimplementation (v2).

Reference: Gao et al., "DA-ADKG: Dependable Asynchronous Distributed Key
Generation with Blockchain-Enforced Accountability" (TDSC-2025-08-1249,
major revision).  Section/algorithm references point to the manuscript;
change IDs (C1-C6) and findings (F#) point to the revision dossier
(research/D7_code_review_and_change_plan.md, research/D5_D6_*.md).

WHY v2 (summary of changes against v1; each is justified in D7 sec. 1)
  C1  The protocol group is BN254 G1 (y^2 = x^3 + 3), not secp256k1.  The
      paper's accountability mechanism re-executes the share-verification
      equation inside an Ethereum smart contract.  Ethereum offers cheap,
      precompiled elliptic-curve operations ONLY on BN254 (EIP-196:
      ECADD/ECMUL; EIP-197: pairing), which is exactly what EthDKG relies
      on (EthDKG p.7) and why Kemmoe et al. (TDSC'23, p.15) chose a BN
      curve.  secp256k1 point arithmetic has no precompile.
  C2  Dealer/attribution signatures are ECDSA over secp256k1, i.e. the
      signatures Ethereum's `ecrecover` precompile verifies (3,000 gas).
      Ed25519 (v1) has no EVM precompile.  Signatures are produced in
      Ethereum's (r, s, v) form with low-s and an RFC 6979-style
      deterministic nonce, and are cross-validated against OpenSSL.
  C3  Schnorr proofs of knowledge over an ARBITRARY base (needed for the
      Das'23 key-derivation step, which proves knowledge of both g^{s_i}
      and h^{s^_i}; finding F3).
  C4  Chaum-Pedersen DLEQ proofs plus EVM-decryptable masked share
      encryption, so an accuser can disclose a Diffie-Hellman key and the
      contract can open a posted ciphertext (EthDKG's dispute mechanism;
      needed for non-delivery complaints, finding F14).
  C5  Known-answer and cross-implementation tests (py_ecc for BN254,
      pycryptodome for Keccak-256, OpenSSL for ECDSA) when those packages
      are installed; the pure-Python core never depends on them.
  C6  Hashing that a contract must recompute (Fiat-Shamir challenges,
      Merkle roots, signed digests) uses Keccak-256 (the EVM's native
      KECCAK256 opcode) over the EVM point encoding (64-byte x||y), with a
      512-bit "wide" reduction for scalars so challenges are unbiased mod
      the 254-bit group order.  Off-chain-only primitives (ECIES transport
      encryption, HKDF) keep SHA-256/AES-256-GCM.
  Plus hardening: canonical (unique) encodings for every scalar and point
  that enters a hash or a stored digest; strict Merkle position checking;
  explicit point validation on every deserialisation.

SECURITY NOTES (must also be stated in the paper's implementation section)
  * BN254's security level is below 128 bits; it is commonly cited at
    roughly 100 bits after improved number-field-sieve attacks on its
    pairing target group [general literature; not one of the uploaded
    papers -- cite a primary source in the revision].  This is the price
    of EVM-verifiable disputes and is the same choice EthDKG made.
  * G1 of BN254 has prime order (cofactor 1): every point on the curve is
    in the group, so an on-curve check is a complete validity check.
  * Research artifact: NOT constant-time, NOT side-channel hardened.  Do
    not use in production.  All secret randomness comes from the OS CSPRNG
    (`secrets`); ECDSA nonces are deterministic (RFC 6979-style HMAC-DRBG).
  * Every number in the revised manuscript must come from fresh runs of
    the experiment harness, never from the previously published tables.

Run `python3 crypto.py` for the self-test suite.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "The 'cryptography' package is required (HKDF + AES-256-GCM for the "
        "off-chain ECIES transport). Install with: pip install cryptography"
    ) from exc

__all__ = [
    # constants / types
    "DOMAIN", "P", "Q", "B", "G", "H", "Point", "SECP_N",
    # encodings
    "i2b", "b2i", "enc_parts", "dec_parts", "scalar_from_bytes",
    "ser_point", "deser_point", "ser_point_evm", "deser_point_evm",
    # hashing
    "keccak256", "sha256", "hash_to_scalar",
    # group
    "is_on_curve", "require_point", "point_add", "point_neg", "point_mul",
    "double_mul", "msm", "rand_scalar", "keygen",
    # commitments and proofs
    "pedersen_commit", "SchnorrProof", "pok_prove", "pok_verify",
    "pok_prove_base", "pok_verify_base", "DLEQProof", "dleq_prove",
    "dleq_verify", "PolyPoK", "poly_pok_prove", "poly_pok_verify",
    # Merkle
    "merkle_root", "merkle_path", "merkle_verify", "commitment_root",
    # polynomials / Shamir
    "Polynomial", "lagrange_coeffs_at_zero", "lagrange_at_zero",
    "interpolate_at_zero_exp", "commit_poly_eval", "verify_share",
    # encryption
    "ecies_encrypt", "ecies_decrypt", "dh_key", "mask_scalars",
    "unmask_scalars",
    # ECDSA (Ethereum-compatible)
    "ecdsa_keygen", "ecdsa_pubkey", "ecdsa_sign", "ecdsa_verify",
    "ecdsa_recover", "eth_address", "signing_digest",
]

DOMAIN = b"DA-ADKG/v2"   # bumped from v1: hash function and group changed

# ==========================================================================
# 1. Encoding helpers.  Canonical and length-prefixed, so that no two
#    different transcripts, signed messages or stored digests can collide
#    by re-bracketing.
# ==========================================================================


def i2b(x: int, n: int = 32) -> bytes:
    """Big-endian fixed-width encoding; raises OverflowError if x does not
    fit (never silently truncates) and on negative input."""
    return int(x).to_bytes(n, "big")


def b2i(b: bytes) -> int:
    return int.from_bytes(b, "big")


def enc_parts(*parts: bytes) -> bytes:
    """Canonical encoding: 4-byte big-endian length prefix per part."""
    out = bytearray()
    for p in parts:
        out += len(p).to_bytes(4, "big")
        out += p
    return bytes(out)


def dec_parts(buf: bytes, expected: Optional[int] = None) -> List[bytes]:
    """Inverse of enc_parts.  Raises ValueError on malformed input or, if
    `expected` is given, on a wrong number of parts."""
    parts: List[bytes] = []
    i, n = 0, len(buf)
    while i < n:
        if i + 4 > n:
            raise ValueError("truncated length prefix")
        ln = int.from_bytes(buf[i : i + 4], "big")
        i += 4
        if i + ln > n:
            raise ValueError("truncated part")
        parts.append(buf[i : i + ln])
        i += ln
    if expected is not None and len(parts) != expected:
        raise ValueError(f"expected {expected} parts, got {len(parts)}")
    return parts


# ==========================================================================
# 2. Hash functions.
#    * keccak256: the EVM's native hash (pure-Python Keccak-f[1600], original
#      Keccak padding 0x01 -- NOT NIST SHA3-256, which pads with 0x06).
#    * sha256: off-chain only (HKDF salt-free extract input, RFC 6979 DRBG).
# ==========================================================================


def _keccak_constants() -> Tuple[List[int], List[List[int]]]:
    """Derive the round constants and rotation offsets from their FIPS 202
    definitions instead of transcribing tables (avoids transcription
    errors; validated by the known-answer test in the self-tests)."""
    def rc_bit(t: int) -> int:
        if t % 255 == 0:
            return 1
        r = [1, 0, 0, 0, 0, 0, 0, 0]
        for _ in range(t % 255):
            r = [0] + r
            r[0] ^= r[8]
            r[4] ^= r[8]
            r[5] ^= r[8]
            r[6] ^= r[8]
            r = r[:8]
        return r[0]

    rcs = []
    for ir in range(24):
        rc = 0
        for j in range(7):
            if rc_bit(j + 7 * ir):
                rc |= 1 << ((1 << j) - 1)
        rcs.append(rc)

    rot = [[0] * 5 for _ in range(5)]
    x, y = 1, 0
    for t in range(24):
        rot[x][y] = ((t + 1) * (t + 2) // 2) % 64
        x, y = y, (2 * x + 3 * y) % 5
    return rcs, rot


_KECCAK_RC, _KECCAK_ROT = _keccak_constants()
_MASK64 = (1 << 64) - 1


def _keccak_f(a: List[int]) -> None:
    """Keccak-f[1600] permutation in place; lane (x, y) is a[x + 5*y]."""
    rot = _KECCAK_ROT
    for rc in _KECCAK_RC:
        # theta
        c = [a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20] for x in range(5)]
        for x in range(5):
            d = c[(x - 1) % 5] ^ (((c[(x + 1) % 5] << 1) | (c[(x + 1) % 5] >> 63)) & _MASK64)
            for y in range(0, 25, 5):
                a[x + y] ^= d
        # rho + pi
        b = [0] * 25
        for x in range(5):
            for y in range(5):
                r = rot[x][y]
                v = a[x + 5 * y]
                b[y + 5 * ((2 * x + 3 * y) % 5)] = (
                    ((v << r) | (v >> (64 - r))) & _MASK64 if r else v)
        # chi
        for y in range(0, 25, 5):
            row = b[y : y + 5]
            for x in range(5):
                a[x + y] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5])
        # iota
        a[0] ^= rc


def _keccak256_pure(data: bytes) -> bytes:
    rate = 136
    msg = bytearray(data)
    msg.append(0x01)
    while len(msg) % rate:
        msg.append(0x00)
    msg[-1] |= 0x80
    a = [0] * 25
    for off in range(0, len(msg), rate):
        block = msg[off : off + rate]
        for i in range(rate // 8):
            a[i] ^= int.from_bytes(block[8 * i : 8 * i + 8], "little")
        _keccak_f(a)
    return b"".join(a[i].to_bytes(8, "little") for i in range(4))


try:  # optional C fast path; must agree with the pure implementation
    from Crypto.Hash import keccak as _pc_keccak  # type: ignore

    def keccak256(*parts: bytes) -> bytes:
        h = _pc_keccak.new(digest_bits=256)
        for p in parts:
            h.update(p)
        return h.digest()

    _KECCAK_BACKEND = "pycryptodome"
except ImportError:  # pragma: no cover
    def keccak256(*parts: bytes) -> bytes:
        return _keccak256_pure(b"".join(parts))

    _KECCAK_BACKEND = "pure-python"


def sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


# ==========================================================================
# 3. Short-Weierstrass a = 0 curve arithmetic in Jacobian coordinates,
#    shared by BN254 (protocol group) and secp256k1 (ECDSA only).  Affine
#    points are (x, y) tuples; None is the point at infinity; Jacobian
#    (X, Y, Z) with Z == 0 is infinity.  Formulas: dbl-2009-l, add-2007-bl
#    (Explicit-Formulas Database), both valid for any a = 0 curve.
# ==========================================================================

Point = Optional[Tuple[int, int]]
_JINF = (0, 0, 0)


def _to_j(pt: Point):
    return (pt[0], pt[1], 1) if pt is not None else _JINF


def _from_j(j, p: int) -> Point:
    if j[2] == 0:
        return None
    zi = pow(j[2], p - 2, p)
    zi2 = (zi * zi) % p
    return ((j[0] * zi2) % p, (j[1] * zi2 * zi) % p)


def _j_double(pt, p: int):
    X1, Y1, Z1 = pt
    if Z1 == 0 or Y1 == 0:
        return _JINF
    A = (X1 * X1) % p
    Bv = (Y1 * Y1) % p
    C = (Bv * Bv) % p
    D = (2 * ((X1 + Bv) * (X1 + Bv) - A - C)) % p
    E = (3 * A) % p
    F = (E * E) % p
    X3 = (F - 2 * D) % p
    Y3 = (E * (D - X3) - 8 * C) % p
    Z3 = (2 * Y1 * Z1) % p
    return (X3, Y3, Z3)


def _j_add(p1, p2, p: int):
    if p1[2] == 0:
        return p2
    if p2[2] == 0:
        return p1
    X1, Y1, Z1 = p1
    X2, Y2, Z2 = p2
    Z1Z1 = (Z1 * Z1) % p
    Z2Z2 = (Z2 * Z2) % p
    U1 = (X1 * Z2Z2) % p
    U2 = (X2 * Z1Z1) % p
    S1 = (Y1 * Z2 * Z2Z2) % p
    S2 = (Y2 * Z1 * Z1Z1) % p
    if U1 == U2:
        if S1 != S2:
            return _JINF          # P + (-P)
        return _j_double(p1, p)   # P + P
    Hh = (U2 - U1) % p
    I = (4 * Hh * Hh) % p
    Jv = (Hh * I) % p
    r = (2 * (S2 - S1)) % p
    V = (U1 * I) % p
    X3 = (r * r - Jv - 2 * V) % p
    Y3 = (r * (V - X3) - 2 * S1 * Jv) % p
    Z3 = (((Z1 + Z2) * (Z1 + Z2) - Z1Z1 - Z2Z2) * Hh) % p
    return (X3, Y3, Z3)


def _j_mul(k: int, j, p: int, n: int):
    """k*P with fixed 4-bit windows (NOT constant-time).  k is reduced mod
    the group order n, so negative scalars are handled correctly."""
    k %= n
    if k == 0 or j[2] == 0:
        return _JINF
    tbl = [_JINF] * 16
    tbl[1] = j
    tbl[2] = _j_double(j, p)
    for i in range(3, 16):
        tbl[i] = _j_add(tbl[i - 1], j, p)
    acc = _JINF
    for w in range((k.bit_length() + 3) // 4 - 1, -1, -1):
        if acc[2] != 0:
            acc = _j_double(_j_double(_j_double(_j_double(acc, p), p), p), p)
        nib = (k >> (4 * w)) & 0xF
        if nib:
            acc = _j_add(acc, tbl[nib], p)
    return acc


def _j_double_mul(a: int, Pj, b: int, Qj, p: int, n: int):
    """a*P + b*Q via Straus interleaving with joint 2-bit windows."""
    a %= n
    b %= n
    tP = [_JINF, Pj, _j_double(Pj, p)]
    tP.append(_j_add(tP[2], Pj, p))
    tQ = [_JINF, Qj, _j_double(Qj, p)]
    tQ.append(_j_add(tQ[2], Qj, p))
    tbl = [[_j_add(tP[i], tQ[k], p) for k in range(4)] for i in range(4)]
    acc = _JINF
    for w in range((max(a.bit_length(), b.bit_length(), 1) + 1) // 2 - 1, -1, -1):
        if acc[2] != 0:
            acc = _j_double(_j_double(acc, p), p)
        ia = (a >> (2 * w)) & 3
        ib = (b >> (2 * w)) & 3
        if ia or ib:
            acc = _j_add(acc, tbl[ia][ib], p)
    return acc


def _sqrt_3mod4(v: int, p: int) -> Optional[int]:
    """Square root mod p for p = 3 (mod 4); None if v is a non-residue."""
    y = pow(v, (p + 1) // 4, p)
    return y if (y * y) % p == v % p else None


# ==========================================================================
# 4. The protocol group: BN254 G1 (alt_bn128), as exposed by EIP-196/197.
#    y^2 = x^3 + 3 over F_P; prime order Q; generator G = (1, 2).
# ==========================================================================

P = 21888242871839275222246405745257275088696311157297823662689037894645226208583
Q = 21888242871839275222246405745257275088548364400416034343698204186575808495617
B = 3
G: Point = (1, 2)


def is_on_curve(pt: Point) -> bool:
    """Complete validity check for G1 (cofactor 1).  Infinity is valid."""
    if pt is None:
        return True
    if not (isinstance(pt, tuple) and len(pt) == 2):
        return False
    x, y = pt
    if not (isinstance(x, int) and isinstance(y, int)):
        return False
    if not (0 <= x < P and 0 <= y < P):
        return False
    return (y * y - (x * x * x + B)) % P == 0


def require_point(pt: Point, allow_infinity: bool = False) -> Point:
    """Raise ValueError unless pt is a valid G1 element (and, by default,
    not the identity).  Call on every point received from the network."""
    if pt is None and not allow_infinity:
        raise ValueError("point at infinity not allowed here")
    if not is_on_curve(pt):
        raise ValueError("point not on BN254 G1")
    return pt


def point_add(a: Point, b: Point) -> Point:
    return _from_j(_j_add(_to_j(a), _to_j(b), P), P)


def point_neg(a: Point) -> Point:
    return None if a is None else (a[0], (-a[1]) % P)


def point_mul(k: int, pt: Point) -> Point:
    return _from_j(_j_mul(k, _to_j(pt), P, Q), P)


def double_mul(a: int, p1: Point, b: int, p2: Point) -> Point:
    """a*p1 + b*p2 (the shape of every Pedersen commitment and opening)."""
    return _from_j(_j_double_mul(a, _to_j(p1), b, _to_j(p2), P, Q), P)


def msm(scalars: Sequence[int], points: Sequence[Point]) -> Point:
    """Multi-scalar multiplication sum_i k_i * P_i (pairs processed with
    Straus; correctness over speed)."""
    if len(scalars) != len(points):
        raise ValueError("length mismatch")
    acc = _JINF
    i = 0
    while i + 1 < len(points):
        acc = _j_add(acc, _j_double_mul(scalars[i], _to_j(points[i]),
                                        scalars[i + 1], _to_j(points[i + 1]),
                                        P, Q), P)
        i += 2
    if i < len(points):
        acc = _j_add(acc, _j_mul(scalars[i], _to_j(points[i]), P, Q), P)
    return _from_j(acc, P)


# --- encodings ------------------------------------------------------------
# Off-chain wire format: compressed, 33 bytes (0x02/0x03 || x); infinity is
# 33 zero bytes.  EVM format (EIP-196): uncompressed 64 bytes x || y, with
# infinity encoded as (0, 0).  All hashing that a contract recomputes uses
# the EVM format.


def ser_point(pt: Point) -> bytes:
    if pt is None:
        return b"\x00" * 33
    return bytes([2 + (pt[1] & 1)]) + i2b(pt[0], 32)


def deser_point(b: bytes) -> Point:
    if len(b) != 33:
        raise ValueError("compressed point must be 33 bytes")
    if b[0] == 0:
        if any(b):
            raise ValueError("non-canonical infinity encoding")
        return None
    if b[0] not in (2, 3):
        raise ValueError("bad point prefix")
    x = b2i(b[1:])
    if x >= P:
        raise ValueError("x out of range (non-canonical)")
    y = _sqrt_3mod4((pow(x, 3, P) + B) % P, P)
    if y is None:
        raise ValueError("not a curve point")
    if (y & 1) != (b[0] & 1):
        y = P - y
    return (x, y)


def ser_point_evm(pt: Point) -> bytes:
    if pt is None:
        return b"\x00" * 64
    return i2b(pt[0], 32) + i2b(pt[1], 32)


def deser_point_evm(b: bytes) -> Point:
    if len(b) != 64:
        raise ValueError("EVM point must be 64 bytes")
    x, y = b2i(b[:32]), b2i(b[32:])
    if x == 0 and y == 0:
        return None
    pt = (x, y)
    if not is_on_curve(pt):
        raise ValueError("not a BN254 G1 point (or non-canonical)")
    return pt


# --- scalars and hashing to scalars ---------------------------------------


def scalar_from_bytes(b: bytes) -> int:
    """Canonical 32-byte scalar decoding: rejects values >= Q so that each
    scalar has exactly one encoding (proof bytes are hashed and stored)."""
    if len(b) != 32:
        raise ValueError("scalar must be 32 bytes")
    v = b2i(b)
    if v >= Q:
        raise ValueError("non-canonical scalar (>= group order)")
    return v


def hash_to_scalar(tag: bytes, *chunks: bytes) -> int:
    """Fiat-Shamir challenge / hash-to-field with WIDE reduction:
        pre = DOMAIN || "/" || tag || "/" || enc_parts(chunks)
        c   = (keccak256(0x00 || pre) || keccak256(0x01 || pre)) mod Q.
    The 512-bit input makes the result statistically uniform mod the
    254-bit Q (bias < 2^-250); a single 256-bit hash mod Q would be biased
    (2^256 / Q ~ 5.3).  Solidity recipe: hi = keccak256(0x00||pre),
    lo = keccak256(0x01||pre);
    c = addmod(mulmod(hi, 2^256 mod Q, Q), lo % Q, Q)."""
    pre = DOMAIN + b"/" + tag + b"/" + enc_parts(*chunks)
    return b2i(keccak256(b"\x00", pre) + keccak256(b"\x01", pre)) % Q


def rand_scalar() -> int:
    """Uniform nonzero scalar in Z_Q^* from the OS CSPRNG."""
    return 1 + secrets.randbelow(Q - 1)


def keygen() -> Tuple[int, Point]:
    """(sk, pk = sk*G): protocol identity / encryption key (Sec. III-A)."""
    sk = rand_scalar()
    return sk, point_mul(sk, G)


# --- the second Pedersen generator H --------------------------------------


def _hash_to_point(tag: bytes) -> Point:
    """Try-and-increment: x = keccak256(DOMAIN, "/hash2point/", tag, ctr)
    mod P until x^3 + 3 is a square; take the even root.  The discrete log
    of the result w.r.t. G is unknown to everyone (Sec. III-E).  Public
    input, so the variable running time leaks nothing."""
    ctr = 0
    while True:
        x = b2i(keccak256(DOMAIN, b"/hash2point/", tag, i2b(ctr, 4))) % P
        y = _sqrt_3mod4((pow(x, 3, P) + B) % P, P)
        if y is not None:
            return (x, y if y % 2 == 0 else P - y)
        ctr += 1


H: Point = _hash_to_point(b"pedersen-generator-H")
if not (is_on_curve(H) and H is not None and H != G):  # import-time sanity
    raise RuntimeError("generator H derivation failed")


# ==========================================================================
# 5. Pedersen commitments (Sec. III-E): C = v*G + r*H.  Perfectly hiding,
#    computationally binding under DL in G1.
# ==========================================================================


def pedersen_commit(v: int, r: int) -> Point:
    return double_mul(v, G, r, H)


# ==========================================================================
# 6. Schnorr proofs of knowledge (Alg. 1 l.5-9, 11; C3 arbitrary base).
#    Statement: X = x*base.  Proof (R, s): R = r*base,
#    c = H(tag, context, base, X, R), s = r + c*x.
#    Verify: s*base == R + c*X.  On-chain: 2 ECMUL + 1 ECADD.
# ==========================================================================

_POK_TAG = b"pok-dlog"


@dataclass(frozen=True)
class SchnorrProof:
    R: Point
    s: int

    def to_bytes(self) -> bytes:
        return enc_parts(ser_point(self.R), i2b(self.s))

    @classmethod
    def from_bytes(cls, b: bytes) -> "SchnorrProof":
        r, s = dec_parts(b, expected=2)
        return cls(deser_point(r), scalar_from_bytes(s))


def _pok_challenge(base: Point, X: Point, R: Point, context: bytes) -> int:
    return hash_to_scalar(_POK_TAG, context, ser_point_evm(base),
                          ser_point_evm(X), ser_point_evm(R))


def pok_prove_base(x: int, base: Point, X: Point, context: bytes = b"") -> SchnorrProof:
    require_point(base)
    r = rand_scalar()
    R = point_mul(r, base)
    c = _pok_challenge(base, X, R, context)
    return SchnorrProof(R, (r + c * x) % Q)


def pok_verify_base(X: Point, proof: SchnorrProof, base: Point,
                    context: bytes = b"") -> bool:
    if X is None or proof.R is None or base is None:
        return False
    if not (is_on_curve(X) and is_on_curve(proof.R) and is_on_curve(base)):
        return False
    if not (0 <= proof.s < Q):
        return False
    c = _pok_challenge(base, X, proof.R, context)
    return point_mul(proof.s, base) == point_add(proof.R, point_mul(c, X))


def pok_prove(sk: int, pk: Point, context: bytes = b"") -> SchnorrProof:
    """Knowledge of sk with pk = sk*G (registration, Alg. 1)."""
    return pok_prove_base(sk, G, pk, context)


def pok_verify(pk: Point, proof: SchnorrProof, context: bytes = b"") -> bool:
    return pok_verify_base(pk, proof, G, context)


# ==========================================================================
# 7. Chaum-Pedersen DLEQ proof (C4): log_{B1} X1 == log_{B2} X2.
#    Compact (c, z) form, 64 bytes of calldata: T1 = z*B1 - c*X1,
#    T2 = z*B2 - c*X2, accept iff c == H(tag, ctx, B1, X1, B2, X2, T1, T2).
#    On-chain: 4 ECMUL + 2 ECADD + 2 KECCAK.  Used for EthDKG-style key
#    disclosure: an accuser proves K = sk*E (resp. sk*pk_dealer) for its
#    registered pk = sk*G, letting the contract open a posted ciphertext.
# ==========================================================================

_DLEQ_TAG = b"dleq"


@dataclass(frozen=True)
class DLEQProof:
    c: int
    z: int

    def to_bytes(self) -> bytes:
        return i2b(self.c) + i2b(self.z)

    @classmethod
    def from_bytes(cls, b: bytes) -> "DLEQProof":
        if len(b) != 64:
            raise ValueError("DLEQ proof must be 64 bytes")
        return cls(scalar_from_bytes(b[:32]), scalar_from_bytes(b[32:]))


def _dleq_challenge(B1, X1, B2, X2, T1, T2, context: bytes) -> int:
    return hash_to_scalar(_DLEQ_TAG, context, *[ser_point_evm(p) for p in
                                                (B1, X1, B2, X2, T1, T2)])


def dleq_prove(x: int, B1: Point, B2: Point, context: bytes = b"") -> DLEQProof:
    require_point(B1)
    require_point(B2)
    X1, X2 = point_mul(x, B1), point_mul(x, B2)
    w = rand_scalar()
    T1, T2 = point_mul(w, B1), point_mul(w, B2)
    c = _dleq_challenge(B1, X1, B2, X2, T1, T2, context)
    return DLEQProof(c, (w + c * x) % Q)


def dleq_verify(B1: Point, X1: Point, B2: Point, X2: Point, proof: DLEQProof,
                context: bytes = b"") -> bool:
    for pt in (B1, X1, B2, X2):
        if pt is None or not is_on_curve(pt):
            return False
    if not (0 <= proof.c < Q and 0 <= proof.z < Q):
        return False
    T1 = double_mul(proof.z, B1, (-proof.c) % Q, X1)
    T2 = double_mul(proof.z, B2, (-proof.c) % Q, X2)
    return proof.c == _dleq_challenge(B1, X1, B2, X2, T1, T2, context)


# ==========================================================================
# 8. Aggregated PoK of the committed polynomial coefficients (Alg. 2 l.12,
#    "ZKProve({a_ik, a^_ik}, {C_ik})"): one challenge over all commitments;
#    for each k the prover shows knowledge of an opening (a_k, a^_k) of
#    C_k = a_k*G + a^_k*H.  The caller MUST bind the dealer identity into
#    `context` (sid || dealer index) so proofs cannot be replayed by others.
# ==========================================================================

_PPOK_TAG = b"pok-poly"


@dataclass(frozen=True)
class PolyPoK:
    T: Tuple[Point, ...]
    z: Tuple[int, ...]
    zh: Tuple[int, ...]

    def to_bytes(self) -> bytes:
        return enc_parts(
            enc_parts(*[ser_point(t) for t in self.T]),
            enc_parts(*[i2b(x) for x in self.z]),
            enc_parts(*[i2b(x) for x in self.zh]),
        )

    @classmethod
    def from_bytes(cls, b: bytes) -> "PolyPoK":
        tb, zb, zhb = dec_parts(b, expected=3)
        T = tuple(deser_point(x) for x in dec_parts(tb))
        z = tuple(scalar_from_bytes(x) for x in dec_parts(zb))
        zh = tuple(scalar_from_bytes(x) for x in dec_parts(zhb))
        return cls(T, z, zh)


def _ppok_challenge(C: Sequence[Point], T: Sequence[Point], context: bytes) -> int:
    return hash_to_scalar(_PPOK_TAG, context,
                          *[ser_point_evm(x) for x in C],
                          *[ser_point_evm(t) for t in T])


def poly_pok_prove(a: Sequence[int], ah: Sequence[int], C: Sequence[Point],
                   context: bytes = b"") -> PolyPoK:
    if not (len(a) == len(ah) == len(C)) or not C:
        raise ValueError("coefficient/commitment length mismatch")
    u = [rand_scalar() for _ in a]
    uh = [rand_scalar() for _ in a]
    T = tuple(double_mul(u[k], G, uh[k], H) for k in range(len(a)))
    c = _ppok_challenge(C, T, context)
    z = tuple((u[k] + c * a[k]) % Q for k in range(len(a)))
    zh = tuple((uh[k] + c * ah[k]) % Q for k in range(len(a)))
    return PolyPoK(T, z, zh)


def poly_pok_verify(C: Sequence[Point], proof: PolyPoK, context: bytes = b"") -> bool:
    if not C or not (len(C) == len(proof.T) == len(proof.z) == len(proof.zh)):
        return False
    if not all(is_on_curve(x) for x in list(C) + list(proof.T)):
        return False
    if not all(0 <= v < Q for v in proof.z + proof.zh):
        return False
    c = _ppok_challenge(C, proof.T, context)
    for k in range(len(C)):
        if double_mul(proof.z[k], G, proof.zh[k], H) != point_add(
                proof.T[k], point_mul(c, C[k])):
            return False
    return True


# ==========================================================================
# 9. Merkle tree (Alg. 2 l.10-11; Alg. 3 l.4).  Keccak-256 with leaf/node
#    domain separation (0x00 / 0x01); an odd last node is promoted
#    unchanged.  merkle_verify is STRICT: it recomputes the expected path
#    shape from (index, n_leaves), so a proof attests the leaf's POSITION,
#    not merely its membership.
# ==========================================================================


def _leaf_hash(b: bytes) -> bytes:
    return keccak256(b"\x00", b)


def _node_hash(left: bytes, right: bytes) -> bytes:
    return keccak256(b"\x01", left, right)


def _next_layer(layer: List[bytes]) -> List[bytes]:
    nxt = [_node_hash(layer[i], layer[i + 1]) for i in range(0, len(layer) - 1, 2)]
    if len(layer) & 1:
        nxt.append(layer[-1])
    return nxt


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    if not leaves:
        raise ValueError("empty leaf set")
    layer = [_leaf_hash(x) for x in leaves]
    while len(layer) > 1:
        layer = _next_layer(layer)
    return layer[0]


def merkle_path(leaves: Sequence[bytes], index: int) -> List[Optional[bytes]]:
    """Sibling hashes from leaf to root; None where the node was promoted.
    Left/right placement is implied by (index, n_leaves)."""
    if not (0 <= index < len(leaves)):
        raise IndexError("leaf index out of range")
    layer = [_leaf_hash(x) for x in leaves]
    path: List[Optional[bytes]] = []
    idx = index
    while len(layer) > 1:
        sib = idx ^ 1
        path.append(layer[sib] if sib < len(layer) else None)
        layer = _next_layer(layer)
        idx >>= 1
    return path


def merkle_verify(root: bytes, leaf: bytes, index: int, n_leaves: int,
                  path: Sequence[Optional[bytes]]) -> bool:
    if not (0 <= index < n_leaves):
        return False
    h = _leaf_hash(leaf)
    idx, width, depth = index, n_leaves, 0
    while width > 1:
        if depth >= len(path):
            return False
        sib = idx ^ 1
        entry = path[depth]
        if sib < width:
            if entry is None:
                return False
            h = _node_hash(entry, h) if idx & 1 else _node_hash(h, entry)
        elif entry is not None:
            return False
        idx >>= 1
        width = (width + 1) // 2
        depth += 1
    return depth == len(path) and h == root


def commitment_root(C: Sequence[Point]) -> bytes:
    """Canonical on-chain anchor MR_i of a commitment vector (D3): Merkle
    root over the EVM encodings, so the contract can recompute it from the
    same calldata it feeds to ECMUL."""
    return merkle_root([ser_point_evm(c) for c in C])


# ==========================================================================
# 10. Polynomials over Z_Q and Shamir machinery (Sec. III-E, IV-B).
#     Participant indices are 1..n; x = 0 encodes the secret f(0).
# ==========================================================================


@dataclass(frozen=True)
class Polynomial:
    coeffs: Tuple[int, ...]  # (a_0, ..., a_d)

    @classmethod
    def random(cls, degree: int, secret: Optional[int] = None) -> "Polynomial":
        if degree < 0:
            raise ValueError("degree must be >= 0")
        cs = [secrets.randbelow(Q) for _ in range(degree + 1)]
        if secret is not None:
            cs[0] = secret % Q
        return cls(tuple(cs))

    @property
    def degree(self) -> int:
        """Formal degree (length - 1); the leading coefficient may be 0 with
        probability 1/Q, which does not affect any protocol property."""
        return len(self.coeffs) - 1

    def eval(self, x: int) -> int:
        acc = 0
        for c in reversed(self.coeffs):
            acc = (acc * x + c) % Q
        return acc


def lagrange_coeffs_at_zero(xs: Sequence[int]) -> List[int]:
    """lambda_i = prod_{j != i} x_j / (x_j - x_i)  (mod Q), so that
    f(0) = sum_i lambda_i f(x_i) for deg f < len(xs)."""
    xs = [x % Q for x in xs]
    if len(set(xs)) != len(xs):
        raise ValueError("duplicate evaluation points")
    if any(x == 0 for x in xs):
        raise ValueError("x = 0 is reserved for the secret")
    out = []
    for i, xi in enumerate(xs):
        num, den = 1, 1
        for j, xj in enumerate(xs):
            if i != j:
                num = (num * xj) % Q
                den = (den * (xj - xi)) % Q
        out.append((num * pow(den, Q - 2, Q)) % Q)
    return out


def lagrange_at_zero(shares: Sequence[Tuple[int, int]]) -> int:
    """Interpolate f(0) from (x_i, y_i) pairs (Sec. IV-B)."""
    lam = lagrange_coeffs_at_zero([x for x, _ in shares])
    return sum(l * (y % Q) for l, (_, y) in zip(lam, shares)) % Q


def interpolate_at_zero_exp(points: Sequence[Tuple[int, Point]]) -> Point:
    """Interpolation in the exponent: from (x_i, f(x_i)*G) return f(0)*G.
    Used by key derivation (F3: public key g^s from 2t+1 g^{s_i})."""
    for _, pt in points:
        require_point(pt, allow_infinity=True)
    lam = lagrange_coeffs_at_zero([x for x, _ in points])
    return msm(lam, [pt for _, pt in points])


def commit_poly_eval(C: Sequence[Point], x: int) -> Point:
    """sum_k x^k * C_k via Horner on group elements; each step multiplies by
    the SMALL scalar x (a participant index), which keeps per-share
    verification cheap (Alg. 3 l.7).  On-chain: d ECMUL + d ECADD."""
    if not C:
        raise ValueError("empty commitment vector")
    acc = _JINF
    for Ck in reversed(C):
        if acc[2] != 0:
            acc = _j_mul(x, acc, P, Q)
        acc = _j_add(acc, _to_j(Ck), P)
    return _from_j(acc, P)


def verify_share(x: int, s: int, s_hat: int, C: Sequence[Point]) -> bool:
    """Alg. 3 l.6-8:  s*G + s_hat*H == sum_k x^k * C_k."""
    if not C or not all(is_on_curve(c) for c in C):
        return False
    return double_mul(s, G, s_hat, H) == commit_poly_eval(C, x)


# ==========================================================================
# 11. Encryption.
#   (a) ECIES transport (off-chain AuthEncrypt, Alg. 2 l.19): ephemeral
#       BN254 ECDH -> HKDF-SHA256 -> AES-256-GCM, caller-supplied AAD.
#   (b) EVM-openable masked shares (C4): s' = s + keccak-derived mask(K)
#       mod Q, where K is a Diffie-Hellman point.  Confidentiality from CDH
#       in the random-oracle model; integrity comes from the dealer's
#       signature over the PLAINTEXT share.  If the recipient discloses K
#       with a DLEQ proof, anyone (the contract) can unmask and re-check.
# ==========================================================================

_ECIES_INFO = DOMAIN + b"/ecies"
_MASK_TAG = b"share-mask"
_ECIES_MIN = 33 + 12 + 16


def _kdf(shared: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=_ECIES_INFO).derive(shared)


def ecies_encrypt(pk: Point, plaintext: bytes, aad: bytes = b"") -> bytes:
    """Output: ser_point(E) (33) || nonce (12) || AES-GCM(ct || tag)."""
    require_point(pk)
    e = rand_scalar()
    E = point_mul(e, G)
    key = _kdf(ser_point(E) + ser_point(point_mul(e, pk)))
    nonce = secrets.token_bytes(12)
    return ser_point(E) + nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


def ecies_decrypt(sk: int, blob: bytes, aad: bytes = b"") -> bytes:
    """Raises ValueError on malformed input and cryptography's InvalidTag on
    any authenticity failure (AuthDecrypt, Alg. 3 l.3)."""
    if len(blob) < _ECIES_MIN:
        raise ValueError("ciphertext too short")
    E = deser_point(blob[:33])
    if E is None:
        raise ValueError("bad ephemeral key")
    key = _kdf(ser_point(E) + ser_point(point_mul(sk, E)))
    return AESGCM(key).decrypt(blob[33:45], blob[45:], aad)


def dh_key(sk: int, other: Point) -> Point:
    """Diffie-Hellman point K = sk*other (other validated)."""
    return point_mul(sk, require_point(other))


def _masks(K: Point, context: bytes, count: int) -> List[int]:
    require_point(K)
    kb = ser_point_evm(K)
    return [hash_to_scalar(_MASK_TAG, context, kb, i2b(i, 4)) for i in range(count)]


def mask_scalars(K: Point, context: bytes, values: Sequence[int]) -> List[int]:
    """Additive masking mod Q under K.  `context` must bind the session and
    both endpoints (e.g. enc_parts(sid, dealer, recipient)), and each K must
    mask only one message per context."""
    return [(v + m) % Q for v, m in zip(values, _masks(K, context, len(values)))]


def unmask_scalars(K: Point, context: bytes, masked: Sequence[int]) -> List[int]:
    return [(v - m) % Q for v, m in zip(masked, _masks(K, context, len(masked)))]


# ==========================================================================
# 12. Ethereum-compatible ECDSA over secp256k1 (C2): dealer attribution.
#     Signatures are 65 bytes r || s || v with v in {27, 28} and LOW s
#     (s <= n/2), the form `ecrecover` + OpenZeppelin's ECDSA.recover accept.
#     The signed digest is keccak256 of a domain-separated message.
# ==========================================================================

SECP_P = 2**256 - 2**32 - 977
SECP_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_SECP_G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
           0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
_SECP_B = 7
_SECP_HALF_N = SECP_N // 2


def _secp_on_curve(pt: Point) -> bool:
    if pt is None:
        return False
    x, y = pt
    return 0 <= x < SECP_P and 0 <= y < SECP_P and (y * y - x ** 3 - _SECP_B) % SECP_P == 0


def _secp_mul(k: int, pt) -> Point:
    return _from_j(_j_mul(k, _to_j(pt), SECP_P, SECP_N), SECP_P)


def _secp_double_mul(a: int, p1, b: int, p2) -> Point:
    return _from_j(_j_double_mul(a, _to_j(p1), b, _to_j(p2), SECP_P, SECP_N), SECP_P)


def signing_digest(message: bytes) -> bytes:
    """32-byte digest that dealers sign and the contract recomputes."""
    return keccak256(DOMAIN, b"/sig/", message)


def ecdsa_keygen() -> Tuple[int, bytes]:
    """(sk, address): address = last 20 bytes of keccak256(x || y)."""
    sk = 1 + secrets.randbelow(SECP_N - 1)
    return sk, eth_address(ecdsa_pubkey(sk))


def ecdsa_pubkey(sk: int) -> Point:
    return _secp_mul(sk, _SECP_G)


def eth_address(pub: Point) -> bytes:
    if not _secp_on_curve(pub):
        raise ValueError("invalid secp256k1 public key")
    return keccak256(i2b(pub[0]) + i2b(pub[1]))[12:]


def _rfc6979_nonces(sk: int, digest: bytes):
    """HMAC-SHA256 DRBG of RFC 6979 sec. 3.2 (qlen = hlen = 256), yielding
    successive candidate nonces in [1, n-1]."""
    x = i2b(sk)
    h1 = i2b(b2i(digest) % SECP_N)
    K, V = b"\x00" * 32, b"\x01" * 32
    K = hmac.new(K, V + b"\x00" + x + h1, hashlib.sha256).digest()
    V = hmac.new(K, V, hashlib.sha256).digest()
    K = hmac.new(K, V + b"\x01" + x + h1, hashlib.sha256).digest()
    V = hmac.new(K, V, hashlib.sha256).digest()
    while True:
        V = hmac.new(K, V, hashlib.sha256).digest()
        k = b2i(V)
        if 1 <= k < SECP_N:
            yield k
        K = hmac.new(K, V + b"\x00", hashlib.sha256).digest()
        V = hmac.new(K, V, hashlib.sha256).digest()


def ecdsa_sign(sk: int, message: bytes) -> bytes:
    """Sign signing_digest(message); returns r(32) || s(32) || v(1)."""
    if not (1 <= sk < SECP_N):
        raise ValueError("invalid ECDSA secret key")
    digest = signing_digest(message)
    z = b2i(digest) % SECP_N
    for k in _rfc6979_nonces(sk, digest):
        R = _secp_mul(k, _SECP_G)
        if R[0] >= SECP_N:      # recovery id would be >= 2: not ecrecover-able
            continue            # (probability ~ 2^-128); take the next nonce
        r = R[0]
        s = (pow(k, SECP_N - 2, SECP_N) * (z + r * sk)) % SECP_N
        if r == 0 or s == 0:
            continue
        recid = R[1] & 1
        if s > _SECP_HALF_N:
            s, recid = SECP_N - s, recid ^ 1
        return i2b(r) + i2b(s) + bytes([27 + recid])
    raise AssertionError("unreachable")  # pragma: no cover


def _parse_sig(sig: bytes) -> Optional[Tuple[int, int, int]]:
    if len(sig) != 65:
        return None
    r, s, v = b2i(sig[:32]), b2i(sig[32:64]), sig[64]
    if not (1 <= r < SECP_N and 1 <= s <= _SECP_HALF_N and v in (27, 28)):
        return None              # strict: low-s only, v in {27, 28}
    return r, s, v


def ecdsa_recover(message: bytes, sig: bytes) -> Optional[Point]:
    """Mirror of `ecrecover` (+ low-s rule): the signer's public key, or
    None if the signature is malformed."""
    parsed = _parse_sig(sig)
    if parsed is None:
        return None
    r, s, v = parsed
    y = _sqrt_3mod4((pow(r, 3, SECP_P) + _SECP_B) % SECP_P, SECP_P)
    if y is None:
        return None
    if (y & 1) != (v - 27):
        y = SECP_P - y
    z = b2i(signing_digest(message)) % SECP_N
    rinv = pow(r, SECP_N - 2, SECP_N)
    pub = _secp_double_mul((-z * rinv) % SECP_N, _SECP_G, (s * rinv) % SECP_N, (r, y))
    return pub if _secp_on_curve(pub) else None


def ecdsa_verify(address: bytes, message: bytes, sig: bytes) -> bool:
    """True iff `sig` is a valid strict-form signature on `message` by the
    key whose Ethereum address is `address` (exactly what a contract checks
    with ecrecover)."""
    pub = ecdsa_recover(message, sig)
    return pub is not None and eth_address(pub) == address


# ==========================================================================
# 13. Self-tests.
# ==========================================================================


def _is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _raises(fn, *args) -> bool:
    try:
        fn(*args)
    except Exception:
        return True
    return False


def _selftest() -> None:  # pragma: no cover
    import time

    ok = lambda name: print(f"  PASS  {name}")
    skip = lambda name: print(f"  SKIP  {name}")

    # --- 1. curve parameters ------------------------------------------------
    assert _is_probable_prime(P) and _is_probable_prime(Q)
    assert P % 4 == 3 and B == 3 and is_on_curve(G)
    assert point_mul(Q, G) is None and point_mul(Q - 1, G) == point_neg(G)
    G2 = (1368015179489954701390400359078579693043519447331113978918064868415326638035,
          9918110051302171585080402603319702774565515993150576347155970296011118125764)
    assert point_add(G, G) == G2 and point_mul(2, G) == G2   # KAT (checked vs py_ecc)
    assert _is_probable_prime(SECP_P) and _is_probable_prime(SECP_N)
    assert _secp_on_curve(_SECP_G) and _secp_mul(SECP_N, _SECP_G) is None
    ok("BN254 G1 and secp256k1 parameters (primality, p=3 mod 4, order, 2G KAT)")

    try:
        from py_ecc.bn128 import G1 as _pG1, multiply as _pmul, add as _padd  # type: ignore
        for _ in range(5):
            k1, k2 = rand_scalar(), rand_scalar()
            a1 = _pmul(_pG1, k1)
            assert point_mul(k1, G) == (a1[0].n, a1[1].n)
            a2 = _padd(a1, _pmul(_pG1, k2))
            assert point_add(point_mul(k1, G), point_mul(k2, G)) == (a2[0].n, a2[1].n)
        ok("BN254 arithmetic matches py_ecc (independent implementation)")
    except ImportError:
        skip("py_ecc not installed: BN254 cross-implementation check")

    # --- 2. group law, encodings ---------------------------------------------
    a, b = rand_scalar(), rand_scalar()
    A, Bp = point_mul(a, G), point_mul(b, G)
    assert point_add(A, Bp) == point_mul(a + b, G)
    assert double_mul(a, G, b, H) == point_add(point_mul(a, G), point_mul(b, H))
    assert msm([a, b, 7], [G, H, A]) == point_add(double_mul(a, G, b, H), point_mul(7, A))
    assert point_add(G, point_neg(G)) is None and point_add(None, G) == G
    assert point_mul(-a, G) == point_neg(A) and point_mul(0, G) is None
    for pt in (A, H, G, None):
        assert deser_point(ser_point(pt)) == pt
        assert deser_point_evm(ser_point_evm(pt)) == pt
    assert _raises(deser_point, b"\x04" + i2b(1))            # bad prefix
    assert _raises(deser_point, b"\x02" + i2b(P))            # x >= P
    assert _raises(deser_point, b"\x00" * 32 + b"\x01")      # bad infinity
    assert _raises(deser_point_evm, i2b(1) + i2b(3))         # off curve
    assert _raises(deser_point_evm, i2b(A[0] + P) + i2b(A[1]))  # non-canonical
    assert not is_on_curve((1, 3)) and _raises(require_point, None)
    ok("group law, MSM, compressed + EVM encodings, malformed-input rejection")

    assert is_on_curve(H) and H != G and H == _hash_to_point(b"pedersen-generator-H")
    ok(f"generator H derived by hash-to-curve (x = 0x{H[0]:064x})")

    # --- 3. hashing ------------------------------------------------------------
    assert _keccak256_pure(b"").hex() == (
        "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")
    for ln in (0, 1, 135, 136, 137, 300):
        m = secrets.token_bytes(ln)
        assert _keccak256_pure(m) == keccak256(m)
    assert keccak256(b"") != hashlib.sha3_256(b"").digest()   # not NIST SHA3
    ok(f"Keccak-256 KAT + multi-block agreement (backend: {_KECCAK_BACKEND})")
    c1 = hash_to_scalar(b"t", b"x")
    assert 0 <= c1 < Q and c1 == hash_to_scalar(b"t", b"x")
    assert c1 != hash_to_scalar(b"u", b"x") and c1 != hash_to_scalar(b"t", b"x", b"")
    hi, lo = keccak256(b"\x00", DOMAIN + b"/t/" + enc_parts(b"x")), \
        keccak256(b"\x01", DOMAIN + b"/t/" + enc_parts(b"x"))
    assert c1 == ((b2i(hi) * pow(2, 256, Q)) % Q + b2i(lo) % Q) % Q  # Solidity recipe
    ok("hash_to_scalar: wide reduction, domain separation, Solidity recipe")

    # --- 4. Pedersen -------------------------------------------------------------
    v1, r1, v2, r2 = (secrets.randbelow(Q) for _ in range(4))
    assert point_add(pedersen_commit(v1, r1), pedersen_commit(v2, r2)) == \
        pedersen_commit(v1 + v2, r1 + r2)
    ok("Pedersen commitment homomorphism")

    # --- 5. Schnorr (base G and base H) -----------------------------------------
    sk, pk = keygen()
    prf = pok_prove(sk, pk, context=b"sid")
    assert pok_verify(pk, prf, b"sid") and not pok_verify(pk, prf, b"other")
    assert not pok_verify(pk, SchnorrProof(prf.R, (prf.s + 1) % Q), b"sid")
    assert not pok_verify(pk, SchnorrProof(prf.R, prf.s + Q), b"sid")  # non-canonical
    assert SchnorrProof.from_bytes(prf.to_bytes()) == prf
    assert _raises(SchnorrProof.from_bytes, enc_parts(ser_point(prf.R), i2b(prf.s + Q)))
    xh = rand_scalar()
    Xh = point_mul(xh, H)
    ph = pok_prove_base(xh, H, Xh, b"sid")
    assert pok_verify_base(Xh, ph, H, b"sid") and not pok_verify_base(Xh, ph, G, b"sid")
    ok("Schnorr PoK on base G and base H (+ tamper, context, canonicity)")

    # --- 6. DLEQ and EthDKG-style key disclosure -------------------------------
    dsk, dpk = keygen()          # dealer
    rsk, rpk = keygen()          # recipient
    K_dealer = dh_key(dsk, rpk)
    K_recip = dh_key(rsk, dpk)
    assert K_dealer == K_recip
    ctx = enc_parts(b"sid", i2b(1, 4), i2b(2, 4))
    s_plain = [rand_scalar(), rand_scalar()]
    masked = mask_scalars(K_dealer, ctx, s_plain)
    proof = dleq_prove(rsk, G, dpk, context=b"disclose")       # rpk = rsk*G, K = rsk*dpk
    assert dleq_verify(G, rpk, dpk, K_recip, proof, context=b"disclose")
    assert unmask_scalars(K_recip, ctx, masked) == s_plain      # third party opens
    assert not dleq_verify(G, rpk, dpk, point_add(K_recip, G), proof, b"disclose")
    assert not dleq_verify(G, rpk, dpk, K_recip, proof, b"other")
    assert DLEQProof.from_bytes(proof.to_bytes()) == proof
    assert unmask_scalars(K_recip, ctx + b"x", masked) != s_plain
    ok("DLEQ proof + masked-share disclosure (EthDKG-style opening)")

    # --- 7. polynomial PoK --------------------------------------------------------
    d = 4
    f, fh = Polynomial.random(d), Polynomial.random(d)
    C = [pedersen_commit(f.coeffs[k], fh.coeffs[k]) for k in range(d + 1)]
    pp = poly_pok_prove(f.coeffs, fh.coeffs, C, context=b"sid|dealer-3")
    assert poly_pok_verify(C, pp, b"sid|dealer-3")
    assert not poly_pok_verify(C, pp, b"sid|dealer-4")              # replay by another
    bad = PolyPoK(pp.T, ((pp.z[0] + 1) % Q,) + pp.z[1:], pp.zh)
    assert not poly_pok_verify(C, bad, b"sid|dealer-3")
    assert PolyPoK.from_bytes(pp.to_bytes()) == pp
    assert not poly_pok_verify([], PolyPoK((), (), ()), b"")
    ok("polynomial-coefficient PoK (+ tamper, dealer binding, empty input)")

    # --- 8. Merkle (strict position) -------------------------------------------
    for nleaves in range(1, 10):
        leaves = [secrets.token_bytes(64) for _ in range(nleaves)]
        root = merkle_root(leaves)
        for idx in range(nleaves):
            path = merkle_path(leaves, idx)
            assert merkle_verify(root, leaves[idx], idx, nleaves, path)
            flipped = bytes([leaves[idx][0] ^ 1]) + leaves[idx][1:]   # always differs
            assert not merkle_verify(root, flipped, idx, nleaves, path)
            if nleaves > 1:
                assert not merkle_verify(root, leaves[idx], (idx + 1) % nleaves,
                                         nleaves, path)
    assert commitment_root(C) == merkle_root([ser_point_evm(c) for c in C])
    ok("Merkle tree, strict position verification, sizes 1..9")

    # --- 9. Shamir, interpolation in the exponent, share check --------------------
    secret = rand_scalar()
    f = Polynomial.random(4, secret=secret)
    pts = [(x, f.eval(x)) for x in range(1, 6)]
    assert lagrange_at_zero(pts) == secret
    assert lagrange_at_zero([(x, f.eval(x)) for x in (2, 5, 7, 9, 11)]) == secret
    assert interpolate_at_zero_exp([(x, point_mul(y, G)) for x, y in pts]) == \
        point_mul(secret, G)
    assert _raises(lagrange_coeffs_at_zero, [1, 1]) and _raises(lagrange_coeffs_at_zero, [0, 2])
    fh = Polynomial.random(4)
    C = [pedersen_commit(f.coeffs[k], fh.coeffs[k]) for k in range(5)]
    x = 7
    assert verify_share(x, f.eval(x), fh.eval(x), C)
    assert not verify_share(x, (f.eval(x) + 1) % Q, fh.eval(x), C)
    assert not verify_share(x, f.eval(x), fh.eval(x), [])
    ok("Shamir / Lagrange (any subset), interpolation in exponent, share check")

    # --- 10. ECIES ---------------------------------------------------------------
    msg = b"share payload " + secrets.token_bytes(64)
    ct = ecies_encrypt(rpk, msg, aad=b"sid")
    assert ecies_decrypt(rsk, ct, aad=b"sid") == msg
    assert _raises(ecies_decrypt, dsk, ct, b"sid")
    assert _raises(ecies_decrypt, rsk, ct, b"other")
    assert _raises(ecies_decrypt, rsk, ct[:-1] + bytes([ct[-1] ^ 1]), b"sid")
    assert _raises(ecies_decrypt, rsk, ct[:40], b"sid")
    assert _raises(ecies_encrypt, (1, 3), msg)
    ok("ECIES over BN254 (+ wrong key / AAD / tamper / truncation / bad pk)")

    # --- 11. ECDSA ------------------------------------------------------------------
    esk, addr = ecdsa_keygen()
    m = b"share-v2|..."
    sig = ecdsa_sign(esk, m)
    assert len(sig) == 65 and sig[64] in (27, 28) and b2i(sig[32:64]) <= _SECP_HALF_N
    assert ecdsa_sign(esk, m) == sig                                 # deterministic
    assert ecdsa_verify(addr, m, sig) and ecdsa_recover(m, sig) == ecdsa_pubkey(esk)
    assert not ecdsa_verify(addr, m + b"!", sig)
    high = sig[:32] + i2b(SECP_N - b2i(sig[32:64])) + bytes([sig[64] ^ 1])
    assert ecdsa_recover(m, high) is None                            # high-s rejected
    assert ecdsa_recover(m, sig[:64] + b"\x1d") is None              # bad v
    _, other = ecdsa_keygen()
    assert not ecdsa_verify(other, m, sig)
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import (
            Prehashed, encode_dss_signature)
        pub = ecdsa_pubkey(esk)
        ossl = ec.EllipticCurvePublicNumbers(pub[0], pub[1], ec.SECP256K1()).public_key()
        ossl.verify(encode_dss_signature(b2i(sig[:32]), b2i(sig[32:64])),
                    signing_digest(m), ec.ECDSA(Prehashed(hashes.SHA256())))
        priv = ec.derive_private_key(esk, ec.SECP256K1())
        assert priv.public_key().public_numbers().x == pub[0]
        ok("ECDSA (r,s,v), low-s, RFC 6979 determinism, ecrecover mirror; "
           "verified by OpenSSL")
    except Exception as exc:  # cross-check must not silently pass
        raise AssertionError(f"OpenSSL ECDSA cross-check failed: {exc!r}")

    # --- 12. micro-benchmark ------------------------------------------------------
    t0 = time.perf_counter()
    for _ in range(50):
        point_mul(rand_scalar(), G)
    dt = (time.perf_counter() - t0) / 50 * 1000
    print(f"  info  BN254 G1 256-bit scalar mult: {dt:.2f} ms (pure Python, "
          "reference only; network.py must use a measured optimised-library cost)")
    print("crypto.py: all self-tests passed.")


if __name__ == "__main__":
    _selftest()
