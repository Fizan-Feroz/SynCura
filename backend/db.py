import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = str(Path(__file__).parent / "data" / "vitals.db")


LAB_COLUMNS = ["gcs", "bun", "creatinine", "wbc", "platelets", "glucose"]


def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    with closing(conn):
        cur = conn.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS vitals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            timestamp REAL,
            hr REAL,
            spo2 REAL,
            rr REAL,
            systolic REAL,
            diastolic REAL,
            temp REAL,
            etco2 REAL,
            risk_score REAL
        )
        ''')
        # Migrate: persist the lab/neuro features the API already accepts.
        existing = {row[1] for row in cur.execute("PRAGMA table_info(vitals)").fetchall()}
        for col in LAB_COLUMNS:
            if col not in existing:
                cur.execute(f"ALTER TABLE vitals ADD COLUMN {col} REAL")
        # Every dashboard query filters by patient and orders by time.
        cur.execute("CREATE INDEX IF NOT EXISTS idx_vitals_patient_ts ON vitals (patient_id, timestamp)")
        # WAL lets the dashboard read while ingest writes (default mode blocks readers).
        cur.execute("PRAGMA journal_mode=WAL")
        conn.commit()


def _get(record, *keys):
    """Return the first present value from record using the given keys, preserving 0.0."""
    for k in keys:
        v = record.get(k)
        if v is not None:
            return v
    return None


def insert_vital(record):
    conn = sqlite3.connect(DB_PATH)
    with closing(conn):
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO vitals (patient_id,timestamp,hr,spo2,rr,systolic,diastolic,temp,etco2,risk_score,"
            "gcs,bun,creatinine,wbc,platelets,glucose) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.get("patient_id"),
                record.get("timestamp"),
                _get(record, "hr", "HR"),
                _get(record, "spo2", "SpO2"),
                _get(record, "rr", "RespRate"),
                _get(record, "systolic", "NISysABP"),
                _get(record, "diastolic", "NIDiasABP"),
                _get(record, "temp", "Temp"),
                _get(record, "etco2", "EtCO2"),
                record.get("risk_score", 0),
                _get(record, "gcs", "GCS"),
                _get(record, "bun", "BUN"),
                _get(record, "creatinine", "Creatinine"),
                _get(record, "wbc", "WBC"),
                _get(record, "platelets", "Platelets"),
                _get(record, "glucose", "Glucose"),
            ),
        )
        conn.commit()


def get_latest_vitals(patient_id, limit=10):
    """Get the latest vital readings for a patient."""
    conn = sqlite3.connect(DB_PATH)
    with closing(conn):
        cur = conn.cursor()
        cur.execute(
            "SELECT timestamp,hr,spo2,rr,systolic,diastolic,temp,etco2,risk_score,"
            "gcs,bun,creatinine,wbc,platelets,glucose FROM vitals WHERE patient_id=? ORDER BY timestamp DESC LIMIT ?",
            (patient_id, limit)
        )
        rows = cur.fetchall()
    return rows


def get_top_patients(limit=6):
    """Get top N patients by latest risk score (exactly one row per patient).

    Ties on timestamp (e.g. a device sending a constant timestamp) resolve to
    the most recently inserted row instead of returning the patient twice.
    """
    conn = sqlite3.connect(DB_PATH)
    with closing(conn):
        cur = conn.cursor()
        cur.execute("""
            SELECT patient_id, risk_score, timestamp FROM (
                SELECT patient_id, risk_score, timestamp,
                       ROW_NUMBER() OVER (PARTITION BY patient_id
                                          ORDER BY timestamp DESC, id DESC) AS rn
                FROM vitals
            )
            WHERE rn = 1
            ORDER BY risk_score DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
    return rows
