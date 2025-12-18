from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class CheckResult:
    check_id: str
    description: str
    severity: str  # P0_BLOCKER, P1_HIGH, P2_MEDIUM, P3_INFO
    passed: bool
    details: dict[str, Any]
    sample_path: Optional[str] = None
