#include <Arduino.h>
#include "ADCSampler.h"
#include "creds.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "model.h"

ADCSampler *adcSampler = NULL;
// Create a memory pool for the nodes in the network
constexpr int tensor_pool_size = 20 * 1024;
uint8_t tensor_pool[tensor_pool_size];

// Define the model to be used
const tflite::Model *wake_model;

// Define the interpreter
tflite::MicroInterpreter *interpreter;

// Input/Output nodes for the network
TfLiteTensor *input;
TfLiteTensor *output;

int8_t *model_input_buffer = nullptr;

void setup_tflite();

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
  int cnt = 0;
  while (true)
  {
    // wait for some samples to save
    uint32_t ulNotificationValue = ulTaskNotifyTake(pdTRUE, xMaxBlockTime);
    if (ulNotificationValue > 0)
    {
      // int8_t *data = (int8_t *)sampler->getCapturedAudioBuffer();
      // cnt++;
      // int offset = cnt == 1 ? 0 : 8000;
      // for (int i = 0; i < sampler->getBufferSizeInBytes(); i++)
      // {
      // }

      // if (cnt == 2)
      // {
      //   model_input_buffer[i + offset] = data[i];
      //   cnt = 0;
      // }
    }
  }
}

void setup()
{
  // setCpuFrequencyMhz(240);
  Serial.begin(115200);

  setup_tflite();

  adcSampler = new ADCSampler(ADC_UNIT_1, ADC1_CHANNEL_5);
  TaskHandle_t adcWriterTaskHandle;
  xTaskCreatePinnedToCore(adcWriterTask, "ADC Writer Task", 4096, adcSampler, 1, &adcWriterTaskHandle, 1);
  adcSampler->start(I2S_NUM_0, adcI2SConfig, 16000, adcWriterTaskHandle);
}

void loop()
{
}

void setup_tflite()
{
  // Load the sample sine model
  Serial.println("Loading Tensorflow model....");
  wake_model = tflite::GetModel(model_data);
  Serial.println("Model loaded!");

  static tflite::ErrorReporter *error_reporter;
  static tflite::MicroErrorReporter micro_error;
  error_reporter = &micro_error;

  // Define ops resolver and error reporting
  static tflite::MicroMutableOpResolver<4> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddConv2D() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk)
  {
    return;
  }

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      wake_model, micro_op_resolver, tensor_pool, tensor_pool_size, error_reporter);

  interpreter = &static_interpreter;

  // Allocate the the model's tensors in the memory pool that was created.
  Serial.println("Allocating tensors to memory pool");
  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    Serial.println("There was an error allocating the memory...ooof");
    return;
  }

  // Define input and output nodes
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("Starting inferences... Input a number! ");

  model_input_buffer = input->data.int8;
}