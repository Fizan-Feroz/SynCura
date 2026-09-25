/*
  Placeholder ESP32 + MAX30105 publisher sketch.
  Requires: MAX30105 library, PubSubClient, WiFi credentials set below.

  Payload contract (must match backend/inference.py FEATURES):
    {"patient_id":"P1","timestamp":<unix seconds>,"HR":72,"SpO2":98}
  Keys are case-sensitive model feature names. The timestamp must be real
  wall-clock time: the backend bins readings into 1-minute slots, so a
  constant timestamp collapses every reading into a single minute.
*/

#include <WiFi.h>
#include <PubSubClient.h>
#include <time.h>

const char* ssid = "YOUR_SSID";
const char* password = "YOUR_PASS";
const char* mqtt_server = "192.168.1.10"; // change to your broker
const char* patient_id = "P1";

WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
  delay(10);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
}

void setup_time() {
  // UTC via NTP; wait until the clock is past 2020-01-01.
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");
  while (time(nullptr) < 1577836800) {
    delay(500);
  }
}

void reconnect() {
  while (!client.connected()) {
    if (client.connect("esp32-client")) {
      // connected
    } else {
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  setup_wifi();
  setup_time();
  client.setServer(mqtt_server, 1883);
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();

  // Read sensor here; placeholders below.
  float hr = 72;
  float spo2 = 98;

  char payload[160];
  snprintf(payload, sizeof(payload),
           "{\"patient_id\":\"%s\",\"timestamp\":%lu,\"HR\":%.1f,\"SpO2\":%.1f}",
           patient_id, (unsigned long)time(nullptr), hr, spo2);
  char topic[48];
  snprintf(topic, sizeof(topic), "vitals/%s", patient_id);
  client.publish(topic, payload);
  delay(1000);
}
