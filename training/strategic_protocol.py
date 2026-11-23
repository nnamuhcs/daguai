"""Versioned, additive strategic feature protocol; v2 positions stay unchanged."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifests" / "strategic-v3.json"
SPEC = json.loads(MANIFEST.read_text())
SIZES = {key: SPEC[key] for key in ("encodingVersion", "observationSize", "actionSize", "historySize")}


def fingerprint():
    return hashlib.sha256(MANIFEST.read_bytes() + (ROOT / "strategic-env.js").read_bytes()
                          + (ROOT.parent / "strategic-policy.js").read_bytes()).hexdigest()
