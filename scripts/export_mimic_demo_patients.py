import argparse
import json
import os
import zipfile

import numpy as np
import pandas as pd


def default_paths():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.environ.get("SYNCURA_DATA_ROOT", os.path.join(root, "data"))
    return (
        os.path.join(data_root, "mimic-iv-clinical-database-demo-2.2.zip"),
        os.path.join(root, "frontend", "src", "mimicDemoPatients.json"),
    )


DEFAULT_ZIP_PATH, DEFAULT_OUT_PATH = default_paths()
ZIP_PATH = DEFAULT_ZIP_PATH
OUT_PATH = DEFAULT_OUT_PATH


def risk_from_vitals(hr: float | None, rr: float | None, temp: float | None, spo2: float | None) -> int:
    score = 30
    if hr is not None and (hr > 110 or hr < 50):
        score += 15
    if rr is not None and (rr > 30 or rr < 12):
        score += 15
    if temp is not None and (temp > 39 or temp < 35):
        score += 10
    if spo2 is not None and spo2 < 92:
        score += 15
    return int(min(99, max(8, score)))


def main():
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open("mimic-iv-clinical-database-demo-2.2/icu/d_items.csv.gz") as f:
            d_items = pd.read_csv(f, compression="gzip")

        def pick_itemid(predicate) -> int | None:
            labels = d_items["label"].fillna("")
            hits = d_items[predicate(labels)][["itemid", "label", "unitname"]]
            if hits.empty:
                return None
            return int(hits.iloc[0]["itemid"])

        hr_id = pick_itemid(lambda s: s.str.fullmatch("Heart Rate", case=False)) or pick_itemid(
            lambda s: s.str.contains("Heart Rate", case=False, na=False)
        )
        rr_id = pick_itemid(lambda s: s.str.fullmatch("Respiratory Rate", case=False)) or pick_itemid(
            lambda s: s.str.contains("Respiratory Rate", case=False, na=False)
        )
        spo2_id = pick_itemid(lambda s: s.str.contains("o2 saturation", case=False, na=False)) or pick_itemid(
            lambda s: s.str.contains("SpO2", case=False, na=False)
        )
        temp_id = pick_itemid(lambda s: s.str.fullmatch("Temperature Celsius", case=False)) or pick_itemid(
            lambda s: s.str.contains("Temperature", case=False, na=False)
        )

        if not all([hr_id, rr_id, spo2_id, temp_id]):
            raise RuntimeError(f"Missing itemids hr={hr_id} rr={rr_id} spo2={spo2_id} temp={temp_id}")

        with z.open("mimic-iv-clinical-database-demo-2.2/icu/icustays.csv.gz") as f:
            icu = pd.read_csv(f, compression="gzip")

        cols = ["stay_id", "charttime", "itemid", "valuenum"]
        with z.open("mimic-iv-clinical-database-demo-2.2/icu/chartevents.csv.gz") as f:
            ce = pd.read_csv(f, compression="gzip", usecols=cols)

        ce = ce.dropna(subset=["stay_id", "itemid", "valuenum"])
        ce["stay_id"] = ce["stay_id"].astype(int)
        ce["itemid"] = ce["itemid"].astype(int)
        ce["charttime"] = pd.to_datetime(ce["charttime"], errors="coerce")

        use_itemids = {hr_id, rr_id, spo2_id, temp_id}
        ce = ce[ce["itemid"].isin(use_itemids)]

        counts = ce.groupby(["stay_id", "itemid"]).size().unstack(fill_value=0)
        good = counts[
            (counts.get(hr_id, 0) > 0)
            & (counts.get(rr_id, 0) > 0)
            & (counts.get(temp_id, 0) > 0)
            & (counts.get(spo2_id, 0) > 0)
        ].index.tolist()

        if len(good) < 12:
            good = ce.groupby("stay_id").size().sort_values(ascending=False).head(12).index.tolist()
        else:
            good = good[:12]

        out = []
        for idx, stay_id in enumerate(good, start=1):
            stay_row = icu[icu["stay_id"] == stay_id].iloc[0]
            start = pd.to_datetime(stay_row.get("intime"), errors="coerce")

            df = ce[ce["stay_id"] == stay_id].copy()
            if start is not None and not pd.isna(start):
                df = df[df["charttime"] >= start]
                df = df[df["charttime"] <= start + pd.Timedelta(hours=2)]

            def vital(itemid):
                vals = df[df["itemid"] == itemid]["valuenum"]
                if vals.empty:
                    return None
                return float(np.nanmedian(vals.values))

            hr = vital(hr_id)
            rr = vital(rr_id)
            spo2 = vital(spo2_id)
            temp = vital(temp_id)

            if hr is not None:
                hr = int(round(np.clip(hr, 45, 170)))
            if rr is not None:
                rr = int(round(np.clip(rr, 10, 42)))
            if spo2 is not None:
                spo2 = int(round(np.clip(spo2, 75, 100)))
            if temp is not None:
                temp = float(np.clip(temp, 34.5, 41))

            risk = risk_from_vitals(hr, rr, temp, spo2)

            out.append(
                {
                    "patient_id": str(stay_id),
                    "bed": f"ICU-{idx:02d}",
                    "status": "Watch",
                    "risk": risk,
                    "trend": "+0",
                    "lead": "MIMIC baseline",
                    "vitals": {
                        "HR": hr if hr is not None else 80,
                        "SpO2": spo2 if spo2 is not None else 96,
                        "Resp": rr if rr is not None else 18,
                        "Temp": round(temp, 1) if temp is not None else 36.8,
                    },
                    "waveform": [max(8, min(99, risk + d)) for d in (-6, -4, -3, -2, -1, 0)],
                }
            )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {OUT_PATH} ({len(out)} patients)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export MIMIC-IV demo patients to frontend JSON.")
    ap.add_argument("--zip-path", default=DEFAULT_ZIP_PATH)
    ap.add_argument("--out-path", default=DEFAULT_OUT_PATH)
    args = ap.parse_args()
    ZIP_PATH, OUT_PATH = args.zip_path, args.out_path
    main()

