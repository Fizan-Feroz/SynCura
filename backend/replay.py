"""
CSV/PhysioNet replay script for streaming live vitals via HTTP or MQTT.

Usage (HTTP mode - pushes JSON to backend API):
  python backend/replay.py \
    --mode http \
    --url http://localhost:8000/ingest \
    --physionet "C:/Users/fizan/Downloads/Techfusion/.../set-a" \
    --outcomes "C:/Users/fizan/Downloads/Techfusion/.../Outcomes-a.txt" \
    --speed 10 \
    --max-patients 5

Usage (MQTT mode - publishes to broker):
  python backend/replay.py \
    --mode mqtt \
    --broker localhost \
    --topic vitals \
    --physionet "C:/Users/fizan/Downloads/Techfusion/.../set-a" \
    --speed 10
"""
import argparse
import time
import json
import glob
import os
import requests
import paho.mqtt.client as mqtt
import pandas as pd

try:
    from backend.inference import canonicalize_vital
except ImportError:
    try:
        from inference import canonicalize_vital
    except ImportError:
        # Standalone use without the serving package: alias map only.
        def canonicalize_vital(vital_dict):
            return {({'SaO2': 'SpO2'}.get(k, k)): v for k, v in vital_dict.items()}


def load_physionet_patient(file_path):
    """Load a single PhysioNet patient file and return events as dict list."""
    df = pd.read_csv(file_path)
    patient_id = os.path.basename(file_path).replace('.txt', '')

    # Pivot: rows=Time, columns=Parameter, values=Value. Duplicate
    # (Time, Parameter) samples are averaged (explicit aggfunc).
    df_pivot = df[df['Parameter'] != 'RecordID'].pivot_table(
        index='Time', columns='Parameter', values='Value', aggfunc='mean')

    def time_to_minutes(t):
        try:
            h, m = str(t).split(':')
            return int(h) * 60 + int(m)
        except (ValueError, AttributeError):
            return None

    df_pivot.index = df_pivot.index.map(time_to_minutes)
    df_pivot = df_pivot[df_pivot.index.notna()]
    df_pivot.index = df_pivot.index.astype(int)
    df_pivot = df_pivot.sort_index()

    for col in df_pivot.columns:
        df_pivot[col] = pd.to_numeric(df_pivot[col], errors='coerce')

    # Route events through the serving alias map so renamed PhysioNet columns
    # are not silently ignored (e.g. SaO2 -> SpO2).
    events = []
    for idx, row in df_pivot.iterrows():
        event = {'patient_id': patient_id, 'timestamp': float(idx) * 60}  # convert to seconds
        for col in df_pivot.columns:
            val = row[col]
            if pd.notna(val):
                event[col] = float(val)
        if len(event) > 2:  # has at least one vital
            events.append(canonicalize_vital(event))

    return events


def replay_http(url, patient_files, speed=1):
    """Stream patient vitals to HTTP endpoint."""
    for pf in patient_files:
        print(f'Replaying {pf}...')
        events = load_physionet_patient(pf)
        
        last_time = None
        for evt in events:
            # Sleep proportional to real time elapsed
            if last_time is not None:
                elapsed = (evt['timestamp'] - last_time) / speed
                if elapsed > 0:
                    time.sleep(min(elapsed, 1.0))  # cap at 1s per event
            
            last_time = evt['timestamp']
            
            # POST to backend
            try:
                resp = requests.post(url, json=evt, timeout=2)
                print(f"[HTTP] {evt['patient_id']} @ {evt['timestamp']:.0f}s -> {resp.status_code}")
            except Exception as e:
                print(f"[HTTP ERROR] {e}")


def replay_mqtt(broker, topic, patient_files, speed=1):
    """Stream patient vitals to MQTT broker."""
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except (TypeError, AttributeError):
        client = mqtt.Client()
    client.connect(broker, 1883, 60)
    client.loop_start()
    
    for pf in patient_files:
        print(f'Replaying {pf}...')
        events = load_physionet_patient(pf)
        
        last_time = None
        for evt in events:
            if last_time is not None:
                elapsed = (evt['timestamp'] - last_time) / speed
                if elapsed > 0:
                    time.sleep(min(elapsed, 1.0))
            
            last_time = evt['timestamp']
            patient_id = evt['patient_id']
            payload = json.dumps(evt)
            client.publish(f"{topic}/{patient_id}", payload)
            print(f"[MQTT] {patient_id} @ {evt['timestamp']:.0f}s")
    
    client.loop_stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['http', 'mqtt'], default='http')
    parser.add_argument('--url', default='http://localhost:8000/ingest', help='HTTP endpoint')
    parser.add_argument('--broker', default='localhost', help='MQTT broker')
    parser.add_argument('--topic', default='vitals', help='MQTT topic')
    parser.add_argument('--physionet', required=True, help='PhysioNet set-a/b/c directory')
    parser.add_argument('--speed', type=float, default=10, help='Speedup factor')
    parser.add_argument('--max-patients', type=int, default=None)
    args = parser.parse_args()
    
    # Find patient files
    patient_files = sorted(glob.glob(os.path.join(args.physionet, '*.txt')))
    if args.max_patients:
        patient_files = patient_files[:args.max_patients]
    
    print(f'Found {len(patient_files)} patient files')
    
    if args.mode == 'http':
        replay_http(args.url, patient_files, args.speed)
    else:
        replay_mqtt(args.broker, args.topic, patient_files, args.speed)


if __name__ == '__main__':
    main()
