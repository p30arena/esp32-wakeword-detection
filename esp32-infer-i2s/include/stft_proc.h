#include "commons.h"
#include "STFT.h"

#define STFT_FRAME_SIZE 512
#define STFT_SHIFT 512
#define STFT_OUT_SIZE 128 * 128

void abs_vector(double *b, int len)
{
  for (int i = 0; i < len; i++)
  {
    b[i] = abs(b[i]);
  }
}

void normalize(double *b, int len)
{
  double m = -INFINITY;
  for (int i = 0; i < len; i++)
  {
    if (m < b[i])
    {
      m = b[i];
    }
  }
  for (int i = 0; i < len; i++)
  {
    b[i] = b[i] / m;
  }
}

void getSpectrogram(int16_t *buf_in, double **buf_out)
{
  const int rate = FREQ;
  double **data;

  STFT process(1, STFT_FRAME_SIZE, STFT_SHIFT);

  data = new double *[1];
  data[0] = new double[STFT_FRAME_SIZE + 2];
  memset(data[0], 0, sizeof(double) * (STFT_FRAME_SIZE + 2));
  double buf_img[32 * 32] = {0};

  int offset = 0;
  int block_offset = 0; // to 32
  int length = 0;
  const int step = STFT_SHIFT * 1;

  while (offset < FREQ)
  {
    length = min(FREQ - offset, step);
    process.stft(&buf_in[offset], length, data);
    memcpy(&buf_out[block_offset], data[0], sizeof(double) * length);
    offset += step;
    block_offset++;
  }

  image_32_512_t srcImg = {buf_out, 128, 128};
  image_t dstImg = {buf_img, 32, 32};
  resize(&srcImg, &dstImg, 32, 32);
  abs_vector(buf_img, 32 * 32);
  normalize(buf_img, 32 * 32);

  delete[] data[0];
  delete[] data;
}

void softmax2(double *a, double *b)
{
  int i;
  double m, sum, constant;

  m = max(*a, *b);

  sum = 0.0;
  sum += exp(*a - m);
  sum += exp(*b - m);

  constant = m + log(sum);
  *a = exp(*a - constant);
  *b = exp(*b - constant);
}
