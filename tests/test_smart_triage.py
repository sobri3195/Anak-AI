import unittest

from smart_triage import (
    AudioFeatures,
    PatientInput,
    Severity,
    SmartTriageModel,
    VideoFeatures,
    Vitals,
)


class SmartTriageModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SmartTriageModel()

    def test_resus_when_severe_hypoxemia(self) -> None:
        patient = PatientInput(
            age_months=18,
            audio=AudioFeatures(cough_severity=0.7),
            video=VideoFeatures(respiratory_rate=65, chest_retraction=0.8),
            vitals=Vitals(heart_rate=150, spo2=88, temperature_c=38.0),
            respiratory_complaint=True,
        )
        result = self.model.predict(patient)
        self.assertEqual(result.severity, Severity.RESUS)
        self.assertGreaterEqual(result.respiratory_score, 5)

    def test_urgent_when_moderate_dehydration_risk(self) -> None:
        patient = PatientInput(
            age_months=10,
            audio=AudioFeatures(cry_weak=0.7),
            video=VideoFeatures(respiratory_rate=30),
            vitals=Vitals(heart_rate=165, spo2=98, temperature_c=38.7, capillary_refill_sec=2.0),
            diarrhea=True,
            fever=True,
        )
        result = self.model.predict(patient)
        self.assertEqual(result.severity, Severity.URGENT)
        self.assertGreaterEqual(result.dehydration_score, 2)

    def test_non_urgent_when_low_risk(self) -> None:
        patient = PatientInput(
            age_months=36,
            audio=AudioFeatures(),
            video=VideoFeatures(respiratory_rate=28),
            vitals=Vitals(heart_rate=105, spo2=98, temperature_c=37.0, capillary_refill_sec=1.5),
        )
        result = self.model.predict(patient)
        self.assertEqual(result.severity, Severity.NON_URGENT)


if __name__ == "__main__":
    unittest.main()
