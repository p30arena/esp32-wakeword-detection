#include "commons.h"
#include "STFT.h"

#define STFT_FRAME_SIZE 512
#define STFT_SHIFT 512
#define STFT_OUT_SIZE 128 * 128

double enclosingPowerOfTwo(int value)
{
  // Return 2**N for integer N such that 2**N >= value.
  return floor(pow(2, ceil(log(value) / log(2.0))));
}

double *cosineWindow(
    int windowLength, int a, int b)
{
  const int even = 1 - windowLength % 2;
  double *newValues = new double[windowLength];
  for (int i = 0; i < windowLength; ++i)
  {
    const double cosArg = (2.0 * PI * i) / (windowLength + even - 1);
    newValues[i] = a - b * cos(cosArg);
  }
  return newValues;
}

double *hannWindow_(int windowLength)
{
  return cosineWindow(windowLength, 0.5, 0.5);
}

void getSpectrogram(int16_t *buf_in, double **out)
{
  const int rate = FREQ;
  double **data;

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
    process.stft(&buf_in[offset], length, data);
    memcpy(&out[block_offset], data[0], sizeof(double) * length);
    offset += step;
    block_offset++;
  }

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
