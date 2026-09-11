"""
incident_log.py

Maintains a machine-readable incident log (JSON on disk + a live
Pandas DataFrame in memory), matching the structure specified in the
project brief.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


@dataclass
class Incident:
    incident_id: int
    car_number: Optional[str]
    timestamp: str
    frame: int
    turn: str
    status: str
    confidence: float
    evidence: str
    replay: Optional[str]


class IncidentLog:
    def __init__(self, log_path: str = "output/logs/incident_log.json"):
        self.log_path = log_path
        self._incidents: List[Incident] = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def add_incident(
        self,
        car_number: Optional[str],
        timestamp_sec: float,
        frame_number: int,
        turn: str,
        status: str,
        confidence: float,
        evidence_path: str,
        replay_path: Optional[str],
    ) -> Incident:
        incident = Incident(
            incident_id=len(self._incidents) + 1,
            car_number=car_number,
            timestamp=_format_timestamp(timestamp_sec),
            frame=frame_number,
            turn=turn,
            status=status,
            confidence=round(float(confidence), 3),
            evidence=evidence_path,
            replay=replay_path,
        )
        self._incidents.append(incident)
        self._flush()
        return incident

    def to_dataframe(self) -> pd.DataFrame:
        if not self._incidents:
            return pd.DataFrame(
                columns=[
                    "incident_id", "car_number", "timestamp", "frame",
                    "turn", "status", "confidence", "evidence", "replay",
                ]
            )
        return pd.DataFrame([asdict(i) for i in self._incidents])

    def _flush(self) -> None:
        with open(self.log_path, "w") as f:
            json.dump([asdict(i) for i in self._incidents], f, indent=2)

    def __len__(self) -> int:
        return len(self._incidents)

    @property
    def incidents(self) -> List[Incident]:
        return list(self._incidents)
