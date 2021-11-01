#include <Arduino.h>
#include <driver/adc.h>
#include "soc/rtc_wdt.h"

#define samples 3200U

volatile unsigned int idx = 0;
int ptr[samples];

TaskHandle_t adcReaderTaskHandle;
TaskHandle_t adcWriterTaskHandle;

void adcReaderTask(void *param)
{
  const TickType_t xMaxBlockTime = pdMS_TO_TICKS(100);
  while (true)
  {
    if (idx == samples)
    {
      xTaskNotify(adcWriterTaskHandle, 1, eIncrement);
      while (ulTaskNotifyTake(pdTRUE, xMaxBlockTime) == 0)
        vTaskDelay(pdMS_TO_TICKS(100));
      ;
    }
    else
    {
      adc2_get_raw(ADC2_CHANNEL_5, ADC_WIDTH_BIT_12, &ptr[idx++]);
    }
  }
}

void adcWriterTask(void *param)
{
  const TickType_t xMaxBlockTime = pdMS_TO_TICKS(100);
  while (true)
  {
    // wait for some samples to save
    uint32_t ulNotificationValue = ulTaskNotifyTake(pdTRUE, xMaxBlockTime);
    if (ulNotificationValue > 0)
    {
      for (int i = 0; i < samples; i++)
      {
        // ESP32 is LE so we need first 2 bytes of uint32_t
        Serial.write((uint8_t *)&ptr[i], 2);
        Serial.write('\n');
      }

      idx = 0;
      xTaskNotify(adcReaderTaskHandle, 1, eIncrement);
    }
    else
    {
      vTaskDelay(pdMS_TO_TICKS(100));
    }
  }
}

void setup()
{
  Serial.begin(115200);

  setCpuFrequencyMhz(240);
  // rtc_wdt_protect_off();
  // rtc_wdt_disable();
  adc2_config_channel_atten(ADC2_CHANNEL_5, ADC_ATTEN_11db);

  xTaskCreatePinnedToCore(adcReaderTask, "ADC Reader Task", 4096, NULL, 1, &adcReaderTaskHandle, 0);
  xTaskCreatePinnedToCore(adcWriterTask, "ADC Writer Task", 4096, NULL, 1, &adcWriterTaskHandle, 1);
}

void loop()
{
}