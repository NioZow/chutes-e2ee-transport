"""
ML-KEM-768 + HKDF-SHA256 + ChaCha20-Poly1305 helpers for Chutes E2EE encryption.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pqcrypto.kem.ml_kem_768 import decrypt as mlkem_decrypt
from pqcrypto.kem.ml_kem_768 import encrypt as mlkem_encrypt
from pqcrypto.kem.ml_kem_768 import generate_keypair as mlkem_generate_keypair

MLKEM_CT_SIZE = 1088
TAG_SIZE = 16

INFO_REQ = b"e2e-req-v1"
INFO_RESP = b"e2e-resp-v1"
INFO_STREAM = b"e2e-stream-v1"


def derive_key(shared_secret: bytes, mlkem_ct: bytes, info: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=mlkem_ct[:16],
        info=info,
    ).derive(shared_secret)


def chacha_encrypt(key: bytes, nonce: bytes, plaintext: bytes) -> tuple[bytes, bytes]:
    ct_tag = ChaCha20Poly1305(key).encrypt(nonce, plaintext, None)
    return ct_tag[:-TAG_SIZE], ct_tag[-TAG_SIZE:]


def chacha_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, tag: bytes) -> bytes:
    return ChaCha20Poly1305(key).decrypt(nonce, ciphertext + tag, None)


@dataclass
class E2EERequestResult:
    blob: bytes
    response_sk: bytes


def build_e2ee_request(e2e_pubkey_b64: str, payload: dict) -> E2EERequestResult:
    """Build an encrypted E2EE request blob.

    Returns the binary blob to send as the request body plus the response
    secret key needed to decrypt the response.
    """
    response_pk, response_sk = mlkem_generate_keypair()

    e2e_pubkey = base64.b64decode(e2e_pubkey_b64)
    mlkem_ct, shared_secret = mlkem_encrypt(e2e_pubkey)
    sym_key = derive_key(shared_secret, mlkem_ct, INFO_REQ)

    payload_with_pk = {**payload, "e2e_response_pk": base64.b64encode(response_pk).decode()}
    compressed = gzip.compress(json.dumps(payload_with_pk).encode())

    nonce = os.urandom(12)
    ciphertext, tag = chacha_encrypt(sym_key, nonce, compressed)
    blob = mlkem_ct + nonce + ciphertext + tag

    return E2EERequestResult(blob=blob, response_sk=response_sk)


def decrypt_response(response_blob: bytes, response_sk: bytes) -> dict:
    """Decrypt a non-streaming E2EE response blob."""
    mlkem_ct = response_blob[:MLKEM_CT_SIZE]
    nonce = response_blob[MLKEM_CT_SIZE : MLKEM_CT_SIZE + 12]
    ciphertext = response_blob[MLKEM_CT_SIZE + 12 : -TAG_SIZE]
    tag = response_blob[-TAG_SIZE:]

    shared_secret = mlkem_decrypt(response_sk, mlkem_ct)
    sym_key = derive_key(shared_secret, mlkem_ct, INFO_RESP)
    plaintext = gzip.decompress(chacha_decrypt(sym_key, nonce, ciphertext, tag))
    return json.loads(plaintext)


def decrypt_stream_init(response_sk: bytes, mlkem_ct_b64: str) -> bytes:
    """Decrypt the e2e_init event to derive the stream symmetric key."""
    mlkem_ct = base64.b64decode(mlkem_ct_b64)
    shared_secret = mlkem_decrypt(response_sk, mlkem_ct)
    return derive_key(shared_secret, mlkem_ct, INFO_STREAM)


def decrypt_stream_chunk(enc_chunk_b64: str, stream_key: bytes) -> str:
    """Decrypt a single E2EE streaming chunk."""
    raw = base64.b64decode(enc_chunk_b64)
    nonce = raw[:12]
    ciphertext = raw[12:-TAG_SIZE]
    tag = raw[-TAG_SIZE:]
    return chacha_decrypt(stream_key, nonce, ciphertext, tag).decode()
