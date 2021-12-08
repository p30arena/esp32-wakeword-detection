#include "commons.h"
#include "STFT.h"
#include "spg.h"

#define STFT_FRAME_SIZE 512
#define STFT_SHIFT 512
#define STFT_OUT_W 128
#define STFT_OUT_SIZE (STFT_OUT_W * STFT_OUT_W)

#define max_spec_val 340.0

void normalizeEX(double *b, int len)
{
  for (int i = 0; i < len; i++)
  {
    b[i] = abs(b[i]);
    b[i] = b[i] >= max_spec_val ? 1 : b[i] / max_spec_val;
  }
}

void getSpectrogram(int16_t *buf_in, int16_t *buf_2_in, bool use_second_buffer = false)
{
  double **data;
  double **buf_out = spg_buffer;
  double *buf_img = spg_img_buffer;

  STFT process(1, STFT_FRAME_SIZE, STFT_SHIFT);

  data = new double *[1];
  data[0] = new double[STFT_FRAME_SIZE + 2];
  memset(data[0], 0, sizeof(double) * (STFT_FRAME_SIZE + 2));

  int offset = 0;
  int block_offset = 0; // to 32
  int length = 0;
  const int step = STFT_SHIFT * 1;

  while (offset < FREQ)
  {
    length = min(FREQ - offset, step);
    process.stft((offset < FREQ_HALF && use_second_buffer) ? &buf_2_in[offset] : &buf_in[use_second_buffer ? offset - FREQ_HALF : offset], length, data);
    // memcpy throws error!
    // memcpy(&buf_out[block_offset], data[0], sizeof(double) * length);
    for (int i = 0; i < length; i++)
    {
      buf_out[block_offset][i] = data[0][i];
    }
    offset += step;
    block_offset++;
  }

  image_32_512_t srcImg = {buf_out, STFT_OUT_W, STFT_OUT_W};
  image_t dstImg = {buf_img, SPG_IMG_W, SPG_IMG_W};
  resize(&srcImg, &dstImg, SPG_IMG_W, SPG_IMG_W);
  normalizeEX(buf_img, SPG_IMG_SIZE);

  delete[] data[0];
  delete[] data;
}

void softmax2(double *a, double *b)
{
  double m, sum, constant;

  m = max(*a, *b);

  sum = 0.0;
  sum += exp(*a - m);
  sum += exp(*b - m);

  constant = m + log(sum);
  *a = exp(*a - constant);
  *b = exp(*b - constant);
}

double sigmoid(double x)
{
  double result;
  result = 1 / (1 + exp(-x));
  return result;
}