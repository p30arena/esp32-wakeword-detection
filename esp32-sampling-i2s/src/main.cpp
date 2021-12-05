#include <Arduino.h>
#include <WiFi.h>
#include "ADCSampler.h"
#include "creds.h"

#define SAMPLE_SIZE 16000

boolean wifiConnected = false;
WiFiServer Server(8840);
WiFiClient RemoteClient;

ADCSampler *adcSampler = NULL;

unsigned long clientAcceptedAt = 0UL;

i2s_config_t adcI2SConfig = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S_LSB,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 1024,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0};

void connectToWiFi(const char *ssid, const char *pwd);
void WiFiEvent(WiFiEvent_t event);
void CheckForConnections();

void adcWriterTask(void *param)
{
  I2SSampler *sampler = (I2SSampler *)param;
  const TickType_t xMaxBlockTime = pdMS_TO_TICKS(100);
  int16_t *samples = (int16_t *)malloc(sizeof(uint16_t) * SAMPLE_SIZE);

  while (true)
  {

    if (wifiConnected)
    {
      CheckForConnections();
    }

    if (wifiConnected && RemoteClient.connected() && (millis() - clientAcceptedAt) > 200)
    {
      int samples_read = sampler->read(samples, SAMPLE_SIZE);
      if (samples_read == SAMPLE_SIZE)
      {
        // Send a packet
        RemoteClient.write((uint8_t *)samples, sizeof(uint16_t) * SAMPLE_SIZE);
      }
    }
  }
}

void setup()
{
  // setCpuFrequencyMhz(240);
  Serial.begin(115200);
  adcSampler = new ADCSampler(ADC_UNIT_1, ADC1_CHANNEL_5, adcI2SConfig);
  TaskHandle_t adcWriterTaskHandle;
  xTaskCreatePinnedToCore(adcWriterTask, "ADC Writer Task", 4096, adcSampler, 1, &adcWriterTaskHandle, 1);
  adcSampler->start();

  connectToWiFi(WIFI_SSID, WIFI_PWD);
}

void loop()
{
}

void connectToWiFi(const char *ssid, const char *pwd)
{
  Serial.println("Connecting to WiFi network: " + String(ssid));

  // delete old config
  WiFi.disconnect(true);
  //register event handler
  WiFi.onEvent(WiFiEvent);

  //Initiate connection
  WiFi.begin(ssid, pwd);

  Serial.println("Waiting for WIFI connection...");
}

//wifi event handler
void WiFiEvent(WiFiEvent_t event)
{
  switch (event)
  {
  case SYSTEM_EVENT_STA_GOT_IP:
    // When connected set
    Serial.print("WiFi connected! IP address: ");
    Serial.println(WiFi.localIP());
    wifiConnected = true;
    Server.begin();
    break;
  case SYSTEM_EVENT_STA_DISCONNECTED:
    Serial.println("WiFi lost connection");
    wifiConnected = false;
    RemoteClient.stop();
    Server.end();
    ESP.deepSleep(1000000U);
    ESP.restart();
    break;
  }
}

void CheckForConnections()
{
  if (Server.hasClient())
  {
    // If we are already connected to another computer,
    // then reject the new connection. Otherwise accept
    // the connection.
    if (RemoteClient.connected())
    {
      Serial.println("Connection rejected");
      Server.available().stop();
    }
    else
    {
      Serial.println("Connection accepted");
      RemoteClient = Server.available();
      clientAcceptedAt = millis();
    }
  }
}