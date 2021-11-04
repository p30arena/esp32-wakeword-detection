#include "STFT.h"

STFT::STFT(int channels_, int frame_, int shift_)
{
  int i;
  channels = channels_;
  frame_size = frame_;
  shift_size = shift_;
  ol = frame_size - shift_size;

  hw = new HannWindow(frame_size, shift_size);
  fft = new Ooura_FFT(frame_size, channels);
  ap = new PostProcessor(frame_size, shift_size, channels);

  buf = new double *[channels];
  for (i = 0; i < channels; i++)
  {
    buf[i] = new double[frame_size];
    memset(buf[i], 0, sizeof(double) * frame_size);
  }
}

STFT::~STFT()
{
  int i;
  delete hw;
  delete fft;
  delete ap;
  for (i = 0; i < channels; i++)
    delete[] buf[i];
  delete[] buf;
}

void STFT::stft(short *in, int length, double **out)
{
  int i, j;
  /*** Shfit & Copy***/
  for (j = 0; j < channels; j++)
  {
    for (i = 0; i < ol; i++)
    {
      buf[j][i] = buf[j][i + shift_size];
    }
  }
  //// EOF
  if (length != shift_size * channels)
  {

    length = length / channels;
    for (i = 0; i < length; i++)
    {
      for (j = 0; j < channels; j++)
        buf[j][i + ol] = (double)(in[i * channels + j]);
    }
    for (i = length; i < shift_size; i++)
    {
      for (j = 0; j < channels; j++)
        buf[j][i + ol] = 0;
    }
    //// continue
  }
  else
  {
    for (i = 0; i < shift_size; i++)
    {
      for (j = 0; j < channels; j++)
      {
        buf[j][i + ol] = (double)(in[i * channels + j]);
      }
    }
  }
  /*** Copy input -> hann_input buffer ***/
  for (i = 0; i < channels; i++)
    memcpy(out[i], buf[i], sizeof(double) * frame_size);

  // scaling for precision
  for (i = 0; i < channels; i++)
    for (j = 0; j < frame_size; j++)
      out[i][j] /= MATLAB_scale;

  /*** Window ***/
  hw->Process(out, channels);

  /*** FFT ***/
  fft->FFT(out);
}

void STFT::stft(short *in, int length, double **out, int target_channels)
{
  int tmp = channels;
  channels = target_channels;
  stft(in, length, out);
  channels = tmp;
}

void STFT::istft(double **in, short *out)
{
  /*** iFFT ***/
  fft->iFFT(in);

  /*** Window ***/
  hw->Process(in, channels);

  // scaling for precision
  for (int i = 0; i < channels; i++)
    for (int j = 0; j < frame_size; j++)
      in[i][j] *= MATLAB_scale;

  /*** Output ***/
  memcpy(out, ap->Overlap(in), sizeof(short) * shift_size * channels);
}

// Single-Channel double
void STFT::stft(short *in, double *out)
{
  int i;
  /*** Shfit & Copy***/
  for (i = 0; i < ol; i++)
  {
    buf[0][i] = buf[0][i + shift_size];
  }
  for (i = 0; i < shift_size; i++)
    buf[0][ol + i] = static_cast<double>(in[i]);

  memcpy(out, buf[0], sizeof(double) * frame_size);

  /*** Window ***/
  hw->Process(out);

  /*** FFT ***/
  fft->FFT(out);
}
void STFT::stft(double *in, double *out)
{
  int i;
  /*** Shfit & Copy***/
  for (i = 0; i < ol; i++)
  {
    buf[0][i] = buf[0][i + shift_size];
  }
  for (i = 0; i < shift_size; i++)
    buf[0][ol + i] = in[i];

  memcpy(out, buf[0], sizeof(double) * frame_size);

  /*** Window ***/
  hw->Process(out);

  /*** FFT ***/
  fft->FFT(out);
}

void STFT::stft(double **in, double **out)
{
  /*** Shfit & Copy***/
#pragma omp parallel for
  for (int j = 0; j < channels; j++)
  {
    for (int i = 0; i < ol; i++)
    {
      buf[j][i] = buf[j][i + shift_size];
    }
    for (int i = 0; i < shift_size; i++)
    {
      buf[j][ol + i] = in[j][i];
      memcpy(out[j], buf[j], sizeof(double) * frame_size);
    }
  }

  // scaling for precision
  for (int i = 0; i < channels; i++)
    for (int j = 0; j < frame_size; j++)
    {
      out[i][j] /= MATLAB_scale;
    }

  /*** Window ***/
  hw->Process(out, channels);

  /*** FFT ***/
  fft->FFT(out);
}

void STFT::stft(double **in, double **out, int target_channels)
{
  /*** Shfit & Copy***/
#pragma omp parallel for
  for (int j = 0; j < target_channels; j++)
  {
    for (int i = 0; i < ol; i++)
    {
      buf[j][i] = buf[j][i + shift_size];
    }
    for (int i = 0; i < shift_size; i++)
    {
      buf[j][ol + i] = in[j][i];
      memcpy(out[j], buf[j], sizeof(double) * frame_size);
    }
  }

  // scaling for precision
  for (int i = 0; i < target_channels; i++)
    for (int j = 0; j < frame_size; j++)
    {
      out[i][j] /= MATLAB_scale;
    }

  /*** Window ***/
  hw->Process(out, target_channels);

  /*** FFT ***/
  fft->FFT(out, target_channels);
}

//for separated 3-channels wav
void STFT::stft(short *in_1, short *in_2, short *in_3, int length, double **out)
{
  int i, j;
  short **in;
  in[0] = in_1;
  in[1] = in_2;
  in[2] = in_3;

  /*** Shfit & Copy***/
  for (j = 0; j < channels; j++)
  {
    for (i = 0; i < ol; i++)
    {
      buf[j][i] = buf[j][i + shift_size];
    }
  }
  //// EOF
  if (length != shift_size * channels)
  {

    length = length / channels;
    for (i = 0; i < length; i++)
    {
      for (j = 0; j < channels; j++)
        buf[j][i + ol] = (double)(in[j][i]);
    }
    for (i = length; i < shift_size; i++)
    {
      for (j = 0; j < channels; j++)
        buf[j][i + ol] = 0;
    }
    //// continue
  }
  else
  {
    for (i = 0; i < shift_size; i++)
    {
      for (j = 0; j < channels; j++)
      {
        buf[j][i + ol] = (double)(in[j][i]);
      }
    }
  }
  /*** Copy input -> hann_input buffer ***/
  for (i = 0; i < channels; i++)
    memcpy(out[i], buf[i], sizeof(double) * frame_size);

  // scaling for precision
  for (i = 0; i < channels; i++)
    for (j = 0; j < frame_size; j++)
      out[i][j] /= MATLAB_scale;

  /*** Window ***/
  hw->Process(out, channels);

  /*** FFT ***/
  fft->FFT(out);
}

void STFT::istft(double *in, short *out)
{
  /*** iFFT ***/
  fft->iFFT(in);

  /*** Window ***/
  hw->Process(in);

  for (int j = 0; j < frame_size; j++)
    in[j] *= MATLAB_scale;

  /*** Output ***/
  memcpy(out, ap->Overlap(in), sizeof(short) * shift_size);
}