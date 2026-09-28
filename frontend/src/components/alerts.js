// Shared alert builders: dashboard list and bed profile cards use the same
// clinical terms, causes, and next steps. One alert per bed — the
// highest-severity active condition wins, so escalation replaces the old card.

export function scoreContributions(vitals) {
  return [
    { name: 'Heart rate', value: (vitals.HR - 85) * 0.24 },
    { name: 'SpO2', value: (92 - vitals.SpO2) * 1.7 },
    { name: 'Respiratory rate', value: (vitals.Resp - 18) * 0.6 },
    { name: 'Temperature', value: (vitals.Temp - 37) * 4.5 },
  ].map((m) => ({ ...m, points: Number((m.value * 0.05).toFixed(2)) }))
}

export function clinicalTermForDriver(name, vitals) {
  switch (name) {
    case 'Heart rate':
      if (vitals.HR >= 100) return 'Tachycardia'
      if (vitals.HR <= 60) return 'Bradycardia'
      return 'Abnormal heart rate'
    case 'SpO2':
      return 'Hypoxemia'
    case 'Respiratory rate':
      return vitals.Resp >= 20 ? 'Tachypnea' : 'Abnormal breathing'
    case 'Temperature':
      return vitals.Temp >= 38 ? 'Fever' : 'Abnormal temperature'
    default:
      return name
  }
}

export function topClinicalPattern(vitals, count = 2) {
  const terms = scoreContributions(vitals)
    .filter((m) => Number.isFinite(m.value))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, count)
    .map((m) => clinicalTermForDriver(m.name, vitals))
  const unique = [...new Set(terms)]
  return unique.length ? unique.join(' + ') : 'insufficient data'
}

export function calculateNews2({ vitals: { HR, Resp, Temp, SpO2 } }) {
  let score = 0
  if (Resp <= 8 || Resp >= 25) score += 3
  else if (Resp >= 21) score += 2
  else if (Resp >= 9 && Resp <= 11) score += 1
  if (SpO2 <= 91) score += 3
  else if (SpO2 <= 93) score += 2
  else if (SpO2 <= 95) score += 1
  if (Temp <= 35) score += 3
  else if (Temp >= 39.1) score += 2
  else if (Temp >= 38.1) score += 1
  if (HR <= 40 || HR >= 131) score += 3
  else if (HR >= 111) score += 2
  else if (HR >= 91 || HR <= 50) score += 1
  return score
}

export function buildAlerts(patients) {
  // One alert per bed: the highest-severity active condition wins. When a bed
  // escalates (e.g. warning -> critical), the new alert replaces the old one
  // instead of stacking, so the list never shows stale repeats.
  const alerts = []
  patients.forEach((p) => {
    const base = { key: `${p.patient_id}-alert`, bed: p.bed }
    if (p.risk >= 90) {
      alerts.push({
        ...base,
        level: 'critical',
        title: 'Very high risk',
        condition: `Pattern: ${topClinicalPattern(p.vitals).toLowerCase()}`,
        signal: `Risk ${p.risk}% (alerts at 90%).`,
        cause: 'The concerning vitals behind this score are shown above. Sensors can also misread — confirm the probe and repeat key readings before acting on numbers alone.',
        action: 'See the patient now and follow your unit\u2019s escalation protocol.',
      })
      return
    }
    if (p.vitals.SpO2 <= 88) {
      alerts.push({
        ...base,
        level: 'warning',
        title: 'Hypoxemia',
        signal: `Low oxygen: SpO2 ${p.vitals.SpO2}% — alert level is 88% or below.`,
        cause: 'The probe may have slipped or be giving a weak signal. Check placement, then recheck the reading.',
        action: 'Reassess the patient; if it stays low, escalate per your unit\u2019s protocol.',
      })
      return
    }
    if (p.vitals.Resp >= 30) {
      alerts.push({
        ...base,
        level: 'warning',
        title: 'Tachypnea',
        signal: `Fast breathing: ${p.vitals.Resp} breaths a minute — alert level is 30 or more.`,
        cause: 'Monitors can miscount when the patient moves. Count breaths yourself over a full minute to confirm.',
        action: 'Assess the patient; if confirmed, escalate per your unit\u2019s protocol.',
      })
      return
    }
    if (p.vitals.Temp >= 39) {
      alerts.push({
        ...base,
        level: 'info',
        title: 'Fever',
        signal: `High temperature: ${p.vitals.Temp.toFixed(1)} °C — alert level is 39 °C or above.`,
        cause: 'Thermometers and measurement sites vary. Repeat the measurement to confirm.',
        action: 'Assess the patient; if confirmed, escalate per your unit\u2019s protocol.',
      })
    }
  })
  return alerts.slice(0, 6)
}
