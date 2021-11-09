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

void dump_d(double *b, int len)
{
  for (int i = 0; i < len; i++)
  {
    Serial.print(b[i], 4);
    Serial.print(" ");
  }
  Serial.println();
}

bool predict()
{
  zeroSPGBuffer();
  getSpectrogram(data);

  for (int i = 0; i < SPG_IMG_SIZE; i++)
  {
    int8_t value = spg_img_buffer[i] * 255 - 128;

    if (value > 127)
    {
      value = 127;
    }

    if (value < -128)
    {
      value = -128;
    }

    spg_img_buffer[i] = value;
  }

  for (int i = 0; i < SPG_IMG_SIZE; i++)
  {
    model_input_buffer[i] = (int8_t)spg_img_buffer[i];
  }

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return false;
  }

  double a = output->data.int8[0];
  double b = output->data.int8[1];
  Serial.print(a);
  Serial.print(" ");
  Serial.print(b);
  Serial.println();
  softmax2(&a, &b);

  Serial.print("CMD PROB: ");
  Serial.println(a);
  Serial.print("OTHER PROB: ");
  Serial.println(b);
  Serial.println("\n\n");

  if (a > b)
  {
    Serial.println("I'm at your service!");
    Serial.println("abreman.ir");
    Serial.println('\n');
  }

  if (a > b)
  {
    return true;
  }
  else
  {
    return false;
  }
}

void adcWriterTask(void *param)
{
  I2SSampler *sampler = (I2SSampler *)param;
  const TickType_t xMaxBlockTime = pdMS_TO_TICKS(100);
  int cnt = 0;
  int8_t *spectrogram;
  bool first_time = true;

  initSPGBuffer();

  while (true)
  {
    // wait for some samples to save
    uint32_t ulNotificationValue = ulTaskNotifyTake(pdTRUE, xMaxBlockTime);
    if (ulNotificationValue > 0)
    {
      if (cnt == 2)
      {
        predict();

        cnt = 0;
      }
      else
      {
        memcpy(&data[cnt == 0 ? 0 : FREQ_HALF], sampler->getCapturedAudioBuffer(), FREQ);

        cnt++;
      }
    }
  }
}

void setup()
{
  setCpuFrequencyMhz(240);
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

  Serial.print("input nbytes: ");
  Serial.println(input->bytes);
  Serial.print("output nbytes: ");
  Serial.println(output->bytes);

  model_input_buffer = input->data.int8;

  Serial.println("Starting inferences...! ");
}
