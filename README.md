# Anak AI — Smart Triage Baseline

Prototype kode untuk triase balita (<5 tahun) di IGD berbasis input multimodal:
- audio (tangis/batuk),
- video pola napas,
- tanda vital.

Saat ini implementasi masih **rule-based baseline**, sebagai kerangka awal sebelum diganti model AI terlatih.

## Jalankan test

```bash
python -m unittest discover -s tests -v
```

## Contoh penggunaan cepat

```python
from smart_triage import AudioFeatures, VideoFeatures, Vitals, PatientInput, SmartTriageModel, explain_result

model = SmartTriageModel()
patient = PatientInput(
    age_months=14,
    audio=AudioFeatures(cough_severity=0.7, cry_weak=0.2),
    video=VideoFeatures(respiratory_rate=58, chest_retraction=0.7, nasal_flaring=0.6),
    vitals=Vitals(heart_rate=150, spo2=91, temperature_c=38.2, capillary_refill_sec=2.0),
    respiratory_complaint=True,
)
result = model.predict(patient)
print(explain_result(result))
```
