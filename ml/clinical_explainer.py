# -*- coding: utf-8 -*-
"""Clinical Explainability Engine for SynCura.

Transforms raw mathematical SHAP attributions into physician-friendly,
actionable clinical insights with directional signs, reference ranges,
and recommended bedside actions.
"""
import numpy as np

# Clinical translation dictionary with reference ranges and organ system
CLINICAL_FEATURE_META = {
    'HR': {
        'label': 'Heart Rate',
        'unit': 'bpm',
        'normal': (60, 100),
        'system': 'Cardiovascular',
        'abnormal_high': 'Severe Tachycardia',
        'abnormal_low': 'Bradycardia'
    },
    'RespRate': {
        'label': 'Respiratory Rate',
        'unit': 'breaths/min',
        'normal': (12, 20),
        'system': 'Pulmonary',
        'abnormal_high': 'Tachypnea (Respiratory Distress)',
        'abnormal_low': 'Bradypnea / Hypoventilation'
    },
    'Temp': {
        'label': 'Body Temperature',
        'unit': '°C',
        'normal': (36.1, 37.8),
        'system': 'Systemic / Infection',
        'abnormal_high': 'Hyperthermia / Fever',
        'abnormal_low': 'Hypothermia'
    },
    'NISysABP': {
        'label': 'Systolic Blood Pressure',
        'unit': 'mmHg',
        'normal': (90, 140),
        'system': 'Hemodynamic',
        'abnormal_high': 'Hypertensive Urgency',
        'abnormal_low': 'Hypotension (Perfusion Failure)'
    },
    'NIDiasABP': {
        'label': 'Diastolic Blood Pressure',
        'unit': 'mmHg',
        'normal': (60, 90),
        'system': 'Hemodynamic',
        'abnormal_high': 'Elevated Diastolic Pressure',
        'abnormal_low': 'Loss of Vasomotor Tone'
    },
    'SpO2': {
        'label': 'Oxygen Saturation (SpO2)',
        'unit': '%',
        'normal': (94, 100),
        'system': 'Pulmonary',
        'abnormal_high': 'Hyperoxia',
        'abnormal_low': 'Hypoxemia'
    },
    'GCS': {
        'label': 'Glasgow Coma Scale (GCS)',
        'unit': 'pts',
        'normal': (14, 15),
        'system': 'Neurological',
        'abnormal_high': 'Normal Alertness',
        'abnormal_low': 'Neurological Deterioration / Lethargy'
    },
    'BUN': {
        'label': 'Blood Urea Nitrogen',
        'unit': 'mg/dL',
        'normal': (7, 20),
        'system': 'Renal / Metabolic',
        'abnormal_high': 'Azotemia / Renal Impairment',
        'abnormal_low': 'Low Urea Nitrogen'
    },
    'Creatinine': {
        'label': 'Serum Creatinine',
        'unit': 'mg/dL',
        'normal': (0.6, 1.2),
        'system': 'Renal / Metabolic',
        'abnormal_high': 'Acute Kidney Injury (AKI)',
        'abnormal_low': 'Low Muscle Mass'
    },
    'WBC': {
        'label': 'White Blood Cells',
        'unit': 'x10^9/L',
        'normal': (4.0, 11.0),
        'system': 'Hematology / Sepsis',
        'abnormal_high': 'Leukocytosis (Infection / Inflammation)',
        'abnormal_low': 'Leukopenia'
    },
    'Platelets': {
        'label': 'Platelet Count',
        'unit': 'x10^9/L',
        'normal': (150, 400),
        'system': 'Hematology / Coagulation',
        'abnormal_high': 'Thrombocytosis',
        'abnormal_low': 'Thrombocytopenia (Bleeding/DIC Risk)'
    },
    'Glucose': {
        'label': 'Blood Glucose',
        'unit': 'mg/dL',
        'normal': (70, 140),
        'system': 'Endocrine / Metabolic',
        'abnormal_high': 'Severe Hyperglycemia',
        'abnormal_low': 'Hypoglycemia'
    },
    'Age': {
        'label': 'Patient Age',
        'unit': 'years',
        'normal': (18, 65),
        'system': 'Demographics',
        'abnormal_high': 'Advanced Age Frailty',
        'abnormal_low': 'Young Adult'
    }
}


def build_physician_shap_report(feature_importance_dict, current_vitals_dict=None, risk_score=None):
    """Generate structured, actionable clinical explanations for doctors.

    Args:
        feature_importance_dict: dict of {feature_name: signed_shap_value}
        current_vitals_dict: dict of {feature_name: raw_observed_value}
        risk_score: optional calibrated patient risk percentage (0-100)

    Returns:
        dict containing:
          - clinical_summary: 1-2 sentence doctor summary
          - top_drivers: list of top risk-increasing drivers
          - stabilizing_factors: list of protective features lowering risk
          - recommended_actions: list of targeted clinical interventions
    """
    if current_vitals_dict is None:
        current_vitals_dict = {}

    drivers = []
    protective = []

    # Sort items by magnitude
    sorted_items = sorted(
        feature_importance_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    for feat_name, shap_val in sorted_items:
        # Extract base vital name
        base_name = feat_name.split('_')[0] if '_' in feat_name else feat_name
        meta = CLINICAL_FEATURE_META.get(base_name, {
            'label': base_name,
            'unit': '',
            'normal': (None, None),
            'system': 'Physiological',
            'abnormal_high': f'Elevated {base_name}',
            'abnormal_low': f'Low {base_name}'
        })

        raw_val = current_vitals_dict.get(base_name, current_vitals_dict.get(feat_name, None))
        val_str = f"{raw_val:.1f} {meta['unit']}".strip() if raw_val is not None and not np.isnan(raw_val) else "Measured in window"

        # Check abnormality
        abnormal_desc = ""
        if raw_val is not None and not np.isnan(raw_val):
            low, high = meta['normal']
            if high is not None and raw_val > high:
                abnormal_desc = meta['abnormal_high']
            elif low is not None and raw_val < low:
                abnormal_desc = meta['abnormal_low']

        entry = {
            'feature': meta['label'],
            'system': meta['system'],
            'current_value': val_str,
            'shap_magnitude': float(abs(shap_val)),
            'direction': 'RISK_INCREASING' if shap_val >= 0 else 'PROTECTIVE',
            'clinical_condition': abnormal_desc or ("Deviating from normal" if shap_val >= 0 else "Within stable range")
        }

        if shap_val >= 0:
            drivers.append(entry)
        else:
            protective.append(entry)

    top_drivers = drivers[:4]
    top_protective = protective[:2]

    # Generate clinical narrative
    primary_causes = [f"{d['feature']} ({d['clinical_condition']})" for d in top_drivers if d['clinical_condition']]
    if not primary_causes:
        primary_causes = [d['feature'] for d in top_drivers]

    summary_text = f"Elevated deterioration risk driven predominantly by: {', '.join(primary_causes[:2])}."
    if top_protective:
        summary_text += f" Stabilizing factor: {top_protective[0]['feature']} is currently buffering risk."

    # Generate recommended clinical bedside actions
    actions = []
    systems_involved = {d['system'] for d in top_drivers}
    if 'Neurological' in systems_involved:
        actions.append("Perform urgent bedside GCS evaluation & assess pupil reactivity")
    if 'Hemodynamic' in systems_involved:
        actions.append("Verify blood pressure cuff placement & evaluate fluid bolus / vasopressor titration")
    if 'Pulmonary' in systems_involved:
        actions.append("Auscultate lung bases, verify FiO2 delivery & obtain stat Arterial Blood Gas (ABG)")
    if 'Renal / Metabolic' in systems_involved or 'Hematology / Sepsis' in systems_involved:
        actions.append("Send urgent repeat serum lactate, basic metabolic panel (BMP), and CBC")

    if not actions:
        actions.append("Perform full primary nursing assessment and review current medication infusions")

    return {
        'clinical_summary': summary_text,
        'top_risk_drivers': top_drivers,
        'protective_factors': top_protective,
        'recommended_bedside_actions': actions
    }
