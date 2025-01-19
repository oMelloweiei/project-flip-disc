#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <esp32cam.h>

const char* WIFI_SSID = "GalaxyBall";
const char* WIFI_PASS = "Ballrock123";
//
const char* SERVER_HOST = "192.168.225.136"; // Replace with your C server's IP
const int SERVER_PORT = 8080; // C server's port

const char* server_fingerprint = "BF:25:D3:25:7F:BB:4E:A9:07:3A:DA:DE:88:DD:BA:1D:4B:D0:C3:FC";
// Your server's SSL certificate fingerprint (SHA-1)

WiFiClientSecure client;

void setup() {
  Serial.begin(115200);

  // Connect to Wi-Fi
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected!");

  // Configure ESP32-CAM
  using namespace esp32cam;
  Config cfg;
  cfg.setPins(pins::AiThinker);
  cfg.setResolution(Resolution::find(800, 600));
  cfg.setBufferCount(2);
  cfg.setJpeg(80);

  if (!Camera.begin(cfg)) {
    Serial.println("Camera initialization failed!");
    return;
  }
}

void sendFrameToServer() {
  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("Failed to capture frame.");
    return;
  }

  Serial.printf("Connecting to server %s:%d\n", SERVER_HOST, SERVER_PORT);
  client.setInsecure(); // Skip SSL certificate validation

  if (!client.connect(SERVER_HOST, SERVER_PORT)) {
    Serial.println("Connection to server failed!");
    return;
  }

  String request = String("POST /webcam HTTP/1.1\r\n") +
                   "Host: " + SERVER_HOST + "\r\n" +
                   "Content-Type: image/jpeg\r\n" +
                   "Content-Length: " + String(frame->size()) + "\r\n" +
                   "Connection: close\r\n\r\n";

  client.print(request);
  frame->writeTo(client);

  while (client.connected() || client.available()) {
    String line = client.readStringUntil('\n');
    Serial.println(line);
  }

  client.stop();
}

void loop() {
  sendFrameToServer();
  delay(5000); // Wait 5 seconds before sending the next frame
}
