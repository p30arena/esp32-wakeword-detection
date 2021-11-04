#ifndef _H_OOURA_FFT_
#define _H_OOURA_FFT_

#include <cmath>

class Ooura_FFT
{
private:
  int frame_size;
  int channels;
  double **a, **w;
  int **ip;

public:
  inline Ooura_FFT(int _frame_size, int _channels);
  inline ~Ooura_FFT();

  inline void FFT(double **);
  inline void FFT(double **, int target_channels);
  inline void iFFT(double **);
  inline void FFT(double *);
  inline void iFFT(double *);
  inline void SingleFFT(double *);
  inline void SingleiFFT(double *);
};

#endif
