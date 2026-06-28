from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class VerificationArtifact:
    fingerprints_path: Path
    scores_path: Path
    meta_path: Path


def build_reference_payload(artifact: VerificationArtifact) -> Dict:
    # Placeholder for future online integrity verification stage.
    return {
        "status": "placeholder",
        "message": "Reference payload interface reserved for online verification.",
        "fingerprints": str(artifact.fingerprints_path),
        "scores": str(artifact.scores_path),
        "meta": str(artifact.meta_path),
    }
