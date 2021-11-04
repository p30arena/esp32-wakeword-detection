#include "commons.h"
#include "STFT.h"

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

double **getSpectrogram(int16_t *buf_in, const int ch = 1, const int frame = 255, const int shift = 128)
{
  const int fftLength = enclosingPowerOfTwo(frame);
  // hannWindow_(frame);
  STFT process(ch, frame, shift);

  // short buf_in[ch * shift];
  double **data;

  data = new double *[ch];
  for (int i = 0; i < ch; i++)
  {
    data[i] = new double[fftLength];
    memset(data[i], 0, sizeof(double) * (fftLength));
  }

  process.stft(buf_in, FREQ, data);

  return data;
}

void freeSpectrogram(double **data, const int ch = 1)
{
  for (int i = 0; i < ch; i++)
    delete[] data[i];
  delete[] data;
}