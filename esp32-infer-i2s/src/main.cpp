#include "main.h"

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

static tflite::ErrorReporter *error_reporter;
static tflite::MicroErrorReporter micro_error;

// Define ops resolver and error reporting
static tflite::MicroMutableOpResolver<4> micro_op_resolver(error_reporter);

int8_t *model_input_buffer = nullptr;

int16_t data[FREQ] = {0};
int16_t last_half_data[FREQ_HALF] = {0};
double x[4000] = {0};
double *y[4];

void setup_tflite();

i2s_config_t adcI2SConfig = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
    .sample_rate = FREQ,
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
  int8_t *spectrogram;
  bool first_time = true;

  y[0] = new double[4000];
  y[1] = new double[4000];
  y[2] = new double[4000];
  y[3] = x;

  while (true)
  {
    // wait for some samples to save
    uint32_t ulNotificationValue = ulTaskNotifyTake(pdTRUE, xMaxBlockTime);
    if (ulNotificationValue > 0)
    {
      if (cnt == 2)
      {
        Serial.println(ESP.getFreeHeap());
        Serial.println(heap_caps_get_free_size(MALLOC_CAP_8BIT));
        Serial.println(heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
        // double **x = new double *[1];
        // x[0] = (double *)heap_caps_malloc(8 * 16000, MALLOC_CAP_8BIT);
        // // data[0] = (double *)multi_heap_malloc(, 8 * 16000);
        // if (x[0] == NULL)
        // {
        //   Serial.println("FUCK ESP32!");
        // }
        // Serial.println(ESP.getFreeHeap());
        // Serial.println(heap_caps_get_free_size(MALLOC_CAP_8BIT));
        // Serial.println(heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
        // delete[] x[0];
        // delete[] x;
        // Serial.println(ESP.getFreeHeap());
        // Serial.println(heap_caps_get_free_size(MALLOC_CAP_8BIT));
        // Serial.println(heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
        // double **data1 = new double *[1];
        // double **data2 = new double *[1];
        // data1[0] = new double[8000];
        // Serial.println(ESP.getFreeHeap());
        // Serial.println(heap_caps_get_free_size(MALLOC_CAP_8BIT));
        // Serial.println(heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
        // data2[0] = new double[8000];
        // Serial.println(ESP.getFreeHeap());
        // Serial.println(heap_caps_get_free_size(MALLOC_CAP_8BIT));
        // Serial.println(heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
        // delete[] data1[0];
        // delete[] data2[0];
        // delete[] data1;
        // delete[] data2;
        // Serial.println(ESP.getFreeHeap());
        // Serial.println(heap_caps_get_free_size(MALLOC_CAP_8BIT));
        // Serial.println(heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

        Serial.println("1");
        getSpectrogram(data, y);
        Serial.println("2");
        // freeSpectrogram(out);
        // Serial.println("3");
        // if (first_time)
        // {
        //   // TfLiteStatus init_status = InitializeMicroFeatures(error_reporter);
        //   // if (init_status != kTfLiteOk)
        //   // {
        //   //   return;
        //   // }
        // }

        // size_t num_samples_read;
        // if (!first_time)
        // {
        // }
        // else
        // {
        //   // TfLiteStatus generate_status = GenerateMicroFeatures(
        //   //     error_reporter, data, FREQ, kFeatureSliceSize,
        //   //     spectrogram, &num_samples_read);

        //   first_time = false;
        // }

        cnt = 0;
      }
      else
      {
        if (cnt == 1)
        {
          // memcpy(last_half_data, sampler->getCapturedAudioBuffer(), FREQ);
        }
        else
        {
          // memcpy(&data[cnt == 0 ? 0 : FREQ_HALF], sampler->getCapturedAudioBuffer(), FREQ);
        }

        cnt++;
      }
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
  adcSampler->start(I2S_NUM_0, adcI2SConfig, FREQ, adcWriterTaskHandle);
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

  error_reporter = &micro_error;

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
  Serial.println("Starting inferences...! ");

  model_input_buffer = input->data.int8;
}
