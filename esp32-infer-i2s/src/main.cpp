#include <Arduino.h>
#include "ADCSampler.h"
#include "creds.h"

#include <math.h>
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"

ADCSampler *adcSampler = NULL;

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

void adcWriterTask(void *param)
{
  I2SSampler *sampler = (I2SSampler *)param;
  const TickType_t xMaxBlockTime = pdMS_TO_TICKS(100);
  while (true)
  {
    // wait for some samples to save
    uint32_t ulNotificationValue = ulTaskNotifyTake(pdTRUE, xMaxBlockTime);
    if (ulNotificationValue > 0)
    {
      if (wifiConnected)
      {
        CheckForConnections();
      }

      if (wifiConnected && RemoteClient.connected())
      {
        // Send a packet
        RemoteClient.write((uint8_t *)sampler->getCapturedAudioBuffer(), sampler->getBufferSizeInBytes());
      }
    }
  }
}

void setup()
{
  // setCpuFrequencyMhz(240);
  Serial.begin(115200);
  adcSampler = new ADCSampler(ADC_UNIT_1, ADC1_CHANNEL_5);
  TaskHandle_t adcWriterTaskHandle;
  xTaskCreatePinnedToCore(adcWriterTask, "ADC Writer Task", 4096, adcSampler, 1, &adcWriterTaskHandle, 1);
  adcSampler->start(I2S_NUM_0, adcI2SConfig, 16000, adcWriterTaskHandle);
}

void loop()
{
}
