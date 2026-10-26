"""
Heterogeneous Big Data Generator for Healthcare IoT Patient Monitoring
Generates realistic multi-modal telemetry streams:
- High-frequency continuous vitals (HR, SBP, DBP, SpO2, RR, Temp, ECG HRV)
- Static/slowly changing clinical demographics (Age Bracket, Gender, Comorbidities)
- Transmission & IoT metadata (Signal Quality Index SQI, Battery, Latency)
- Ground-truth clinical risk categories for validation and benchmarking.
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_healthcare_iot_data(n_samples=50000, random_state=42, output_path=None):
    """
    Generate heterogeneous patient vitals with realistic clinical interdependencies.

    Clinical Distributions:
    - 65% Stable Baseline
    - 10% Sepsis / Septic Shock Risk (Tachycardia, Hypotension, Hyperthermia, Tachypnea)
    - 10% Respiratory Distress / ARDS (Severe Hypoxemia, Tachypnea)
    - 10% Hypertensive Crisis / Cardiac Strain (Severe Hypertension, elevated HRV strain)
    - 5%  Sensor Artifact / Telemetry Disconnects (Erratic drops, low SQI)
    """
    np.random.seed(random_state)

    # Cohort assignments based on target clinical distributions
    cohort_probs = [0.65, 0.10, 0.10, 0.10, 0.05]
    cohort_assignments = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=cohort_probs)

    # Initialize physiological arrays
    heart_rate = np.zeros(n_samples)
    systolic_bp = np.zeros(n_samples)
    diastolic_bp = np.zeros(n_samples)
    spo2 = np.zeros(n_samples)
    resp_rate = np.zeros(n_samples)
    temperature = np.zeros(n_samples)
    ecg_hrv = np.zeros(n_samples)  # SDNN in ms
    sqi = np.zeros(n_samples)      # Signal Quality Index [0, 1]
    latency_ms = np.zeros(n_samples)
    battery_pct = np.random.uniform(15.0, 100.0, size=n_samples)

    for i in range(n_samples):
        cohort = cohort_assignments[i]

        if cohort == 0:
            # Stable Baseline
            heart_rate[i] = np.random.normal(72, 6)
            systolic_bp[i] = np.random.normal(118, 7)
            diastolic_bp[i] = np.random.normal(76, 5)
            spo2[i] = np.clip(np.random.normal(98.2, 0.9), 94.0, 100.0)
            resp_rate[i] = np.random.normal(15, 2)
            temperature[i] = np.random.normal(36.8, 0.3)
            ecg_hrv[i] = np.random.normal(65, 12)
            sqi[i] = np.clip(np.random.normal(0.95, 0.04), 0.70, 1.0)
            latency_ms[i] = np.random.exponential(15) + 10

        elif cohort == 1:
            # Sepsis / Septic Shock Risk (Compensatory Tachycardia, Hypotension, Fever/Hypothermia)
            heart_rate[i] = np.random.normal(122, 12)
            systolic_bp[i] = np.random.normal(82, 7)    # Hypotension
            diastolic_bp[i] = np.random.normal(52, 6)
            spo2[i] = np.clip(np.random.normal(93.0, 2.5), 85.0, 98.0)
            resp_rate[i] = np.random.normal(27, 4)      # Tachypnea
            temperature[i] = np.random.choice([np.random.normal(39.1, 0.6), np.random.normal(35.4, 0.4)])
            ecg_hrv[i] = np.random.normal(28, 8)
            sqi[i] = np.clip(np.random.normal(0.88, 0.08), 0.60, 1.0)
            latency_ms[i] = np.random.exponential(25) + 15

        elif cohort == 2:
            # Acute Respiratory Distress (Hypoxemia + Elevated RR)
            heart_rate[i] = np.random.normal(112, 10)
            systolic_bp[i] = np.random.normal(130, 12)
            diastolic_bp[i] = np.random.normal(82, 8)
            spo2[i] = np.clip(np.random.normal(84.0, 3.5), 65.0, 89.9)  # Dangerous hypoxemia
            resp_rate[i] = np.random.normal(32, 5)                       # Severe tachypnea
            temperature[i] = np.random.normal(37.4, 0.5)
            ecg_hrv[i] = np.random.normal(35, 9)
            sqi[i] = np.clip(np.random.normal(0.91, 0.06), 0.65, 1.0)
            latency_ms[i] = np.random.exponential(20) + 12

        elif cohort == 3:
            # Hypertensive Crisis / Severe Cardiac Strain
            heart_rate[i] = np.random.normal(108, 11)
            systolic_bp[i] = np.random.normal(195, 15)  # Severe SBP
            diastolic_bp[i] = np.random.normal(118, 10) # Severe DBP
            spo2[i] = np.clip(np.random.normal(95.5, 1.8), 90.0, 99.0)
            resp_rate[i] = np.random.normal(21, 3)
            temperature[i] = np.random.normal(37.0, 0.4)
            ecg_hrv[i] = np.random.normal(22, 6)        # Suppressed autonomic tone
            sqi[i] = np.clip(np.random.normal(0.93, 0.05), 0.70, 1.0)
            latency_ms[i] = np.random.exponential(18) + 10

        elif cohort == 4:
            # Sensor Lead Detach / Transmission Noise Artifact
            heart_rate[i] = np.random.choice([0.0, np.random.uniform(30, 220)])
            systolic_bp[i] = np.random.uniform(40, 240)
            diastolic_bp[i] = np.random.uniform(20, 140)
            spo2[i] = np.random.choice([0.0, np.random.uniform(40, 100)])
            resp_rate[i] = np.random.choice([0.0, np.random.uniform(4, 50)])
            temperature[i] = np.random.uniform(32.0, 42.0)
            ecg_hrv[i] = np.random.uniform(5, 120)
            sqi[i] = np.clip(np.random.normal(0.25, 0.12), 0.0, 0.45)   # Degraded SQI
            latency_ms[i] = np.random.exponential(120) + 80             # High transmission latency

    # Demographics and Medical History (Categorical / Static Context)
    patient_ids = [f"PAT_{i:07d}" for i in range(1, n_samples + 1)]
    device_ids = [f"DEV_IOT_{np.random.randint(100, 999)}" for _ in range(n_samples)]
    age_groups = np.random.choice(["18-35", "36-50", "51-65", "66-80", "80+"], size=n_samples, p=[0.15, 0.20, 0.30, 0.25, 0.10])
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.51, 0.49])
    
    # Comorbidities (0 or 1)
    has_hypertension = np.random.choice([0, 1], size=n_samples, p=[0.60, 0.40])
    has_diabetes = np.random.choice([0, 1], size=n_samples, p=[0.72, 0.28])
    has_copd = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    has_cardiovascular = np.random.choice([0, 1], size=n_samples, p=[0.78, 0.22])

    # Assemble into DataFrame
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "device_id": device_ids,
        "age_group": age_groups,
        "gender": gender,
        "heart_rate": np.round(heart_rate, 1),
        "systolic_bp": np.round(systolic_bp, 1),
        "diastolic_bp": np.round(diastolic_bp, 1),
        "spo2": np.round(spo2, 1),
        "resp_rate": np.round(resp_rate, 1),
        "temperature": np.round(temperature, 2),
        "ecg_hrv": np.round(ecg_hrv, 1),
        "has_hypertension": has_hypertension,
        "has_diabetes": has_diabetes,
        "has_copd": has_copd,
        "has_cardiovascular": has_cardiovascular,
        "sqi": np.round(sqi, 3),
        "battery_pct": np.round(battery_pct, 1),
        "latency_ms": np.round(latency_ms, 1),
        "ground_truth_cohort": cohort_assignments
    })

    # Save to disk if requested
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if output_path.endswith(".parquet"):
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        print(f"[Data Generator] Successfully generated {n_samples:,} records to: {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Healthcare IoT Big Data Generator")
    parser.add_argument("--samples", type=int, default=50000, help="Number of telemetry events to generate")
    parser.add_argument("--out", type=str, default="data/healthcare_iot_telemetry.csv", help="Output file path")
    args = parser.parse_args()

    generate_healthcare_iot_data(n_samples=args.samples, output_path=args.out)
