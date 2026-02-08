from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class Severity(str, Enum):
    RESUS = "resus"
    URGENT = "urgent"
    NON_URGENT = "non_urgent"


@dataclass
class AudioFeatures:
    cry_weak: float = 0.0
    cough_severity: float = 0.0
    stridor_like_sound: float = 0.0


@dataclass
class VideoFeatures:
    respiratory_rate: int = 0
    chest_retraction: float = 0.0
    nasal_flaring: float = 0.0


@dataclass
class Vitals:
    heart_rate: int = 0
    spo2: int = 100
    temperature_c: float = 36.5
    capillary_refill_sec: float = 1.5


@dataclass
class PatientInput:
    age_months: int
    audio: AudioFeatures
    video: VideoFeatures
    vitals: Vitals
    diarrhea: bool = False
    fever: bool = False
    respiratory_complaint: bool = False


@dataclass
class TriageResult:
    severity: Severity
    dehydration_score: float
    respiratory_score: float
    reasons: List[str]


class SmartTriageModel:
    """Rule-based baseline model for pediatric ED smart triage.

    This is a deterministic baseline scaffold that can later be replaced by a
    learned AI model without changing input/output interfaces.
    """

    def score_dehydration(self, patient: PatientInput) -> tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        if patient.diarrhea:
            score += 1.0
            reasons.append("Diare")

        if patient.vitals.capillary_refill_sec >= 3:
            score += 2.0
            reasons.append("CRT memanjang")

        if patient.audio.cry_weak >= 0.6:
            score += 1.0
            reasons.append("Tangis lemah")

        if patient.vitals.heart_rate > 160:
            score += 1.0
            reasons.append("Takikardi")

        if patient.vitals.temperature_c >= 38.5:
            score += 0.5
            reasons.append("Demam tinggi")

        return score, reasons

    def score_respiratory_failure(self, patient: PatientInput) -> tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        rr = patient.video.respiratory_rate
        if rr >= 60:
            score += 2.0
            reasons.append("Takipnea berat")
        elif rr >= 45:
            score += 1.0
            reasons.append("Takipnea")

        if patient.video.chest_retraction >= 0.6:
            score += 2.0
            reasons.append("Retraksi dada")

        if patient.video.nasal_flaring >= 0.6:
            score += 1.0
            reasons.append("Cuping hidung")

        if patient.audio.cough_severity >= 0.6:
            score += 0.5
            reasons.append("Batuk berat")

        if patient.audio.stridor_like_sound >= 0.5:
            score += 1.5
            reasons.append("Curiga stridor")

        if patient.vitals.spo2 < 90:
            score += 3.0
            reasons.append("Hipoksemia berat")
        elif patient.vitals.spo2 < 94:
            score += 1.5
            reasons.append("Desaturasi")

        return score, reasons

    def predict(self, patient: PatientInput) -> TriageResult:
        dehydration_score, d_reasons = self.score_dehydration(patient)
        respiratory_score, r_reasons = self.score_respiratory_failure(patient)

        reasons = d_reasons + r_reasons

        if respiratory_score >= 5 or dehydration_score >= 4:
            severity = Severity.RESUS
        elif respiratory_score >= 3 or dehydration_score >= 2:
            severity = Severity.URGENT
        else:
            severity = Severity.NON_URGENT

        return TriageResult(
            severity=severity,
            dehydration_score=dehydration_score,
            respiratory_score=respiratory_score,
            reasons=reasons,
        )


def explain_result(result: TriageResult) -> Dict[str, object]:
    return {
        "severity": result.severity.value,
        "dehydration_score": round(result.dehydration_score, 2),
        "respiratory_score": round(result.respiratory_score, 2),
        "reasons": result.reasons,
    }
