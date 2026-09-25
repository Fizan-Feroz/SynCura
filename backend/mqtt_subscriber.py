import os
import json
import paho.mqtt.client as mqtt

try:
    from backend.db import init_db, insert_vital
    from backend.inference import get_engine, canonicalize_vital
except ImportError:
    from db import init_db, insert_vital
    from inference import get_engine, canonicalize_vital

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
TOPIC = os.getenv("MQTT_TOPIC", "vitals/#")

REQUIRED_FIELDS = {"patient_id", "timestamp"}

init_db()
inference_engine = get_engine()

try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except (TypeError, AttributeError):
    client = mqtt.Client()


def on_connect(client, userdata, flags, rc, properties=None):
    print("Connected to MQTT", rc)
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        if not isinstance(data, dict) or not REQUIRED_FIELDS.issubset(data.keys()):
            print("Skipping invalid MQTT payload: missing required fields")
            return
        # Devices may send lowercase keys ("hr", "spo2"); map to model names.
        data = canonicalize_vital(data)
        data["patient_id"] = str(data["patient_id"])
        risk_score = inference_engine.add_vital(data["patient_id"], data)
        if risk_score is None:
            print(f"Model unavailable; storing {data['patient_id']} reading without a score")
        data["risk_score"] = risk_score
        insert_vital(data)
    except json.JSONDecodeError:
        print("Failed to decode MQTT message: not valid JSON")
    except Exception as e:
        print("Failed to handle message:", e)


client.on_connect = on_connect
client.on_message = on_message

if __name__ == "__main__":
    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD or None)
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()
