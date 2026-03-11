import base64
import hashlib
import hmac
import os
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class PolyL2Creds:
    address: str
    api_key: str
    api_passphrase: str
    api_secret: str  # base64-url encoded


def load_l2_creds_from_env(prefix: str = "POLY_") -> PolyL2Creds:
    """
    Reads L2 CLOB credentials from environment variables:
      - POLY_ADDRESS
      - POLY_API_KEY
      - POLY_PASSPHRASE
      - POLY_API_SECRET
    """
    address = os.environ.get(f"{prefix}ADDRESS", "").strip()
    api_key = os.environ.get(f"{prefix}API_KEY", "").strip()
    api_passphrase = os.environ.get(f"{prefix}PASSPHRASE", "").strip()
    api_secret = os.environ.get(f"{prefix}API_SECRET", "").strip()

    missing = [
        name
        for name, val in [
            ("ADDRESS", address),
            ("API_KEY", api_key),
            ("PASSPHRASE", api_passphrase),
            ("API_SECRET", api_secret),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(prefix + m for m in missing)}")

    return PolyL2Creds(
        address=address,
        api_key=api_key,
        api_passphrase=api_passphrase,
        api_secret=api_secret,
    )


def build_hmac_signature(secret_b64url: str, timestamp: int, method: str, request_path: str, body: str | None = None) -> str:
    """
    Mirrors py-clob-client's HMAC signing:
      message = f\"{timestamp}{method}{request_path}\" + body (if provided)
      secret is base64-url decoded
      signature is base64-url encoded HMAC-SHA256 digest
    """
    base64_secret = base64.urlsafe_b64decode(secret_b64url)
    message = f"{timestamp}{method}{request_path}"
    if body:
        message += str(body).replace("'", '"')
    digest = hmac.new(base64_secret, message.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8")


def make_l2_headers(creds: PolyL2Creds, method: str, request_path: str, body: str | None = None, timestamp: int | None = None) -> dict[str, str]:
    """
    Returns the 5 L2 headers documented by Polymarket:
      POLY_ADDRESS, POLY_SIGNATURE, POLY_TIMESTAMP, POLY_API_KEY, POLY_PASSPHRASE
    """
    ts = int(time.time()) if timestamp is None else int(timestamp)
    sig = build_hmac_signature(creds.api_secret, ts, method, request_path, body)
    return {
        "POLY_ADDRESS": creds.address,
        "POLY_SIGNATURE": sig,
        "POLY_TIMESTAMP": str(ts),
        "POLY_API_KEY": creds.api_key,
        "POLY_PASSPHRASE": creds.api_passphrase,
    }

