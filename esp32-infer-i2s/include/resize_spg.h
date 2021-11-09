#include <stdint.h>

typedef struct
{
  double *pixels;
  uint32_t w;
  uint32_t h;
} image_t;

typedef struct
{
  double **pixels;
  uint32_t w;
  uint32_t h;
} image_32_512_t;

#define imgWriteChannel(channel, width, r, c, v) \
  (channel)[(r) * (width) + (c)] = (v)

double imgReadChannel(image_32_512_t *channel, int width, int r, int c)
{
  int n = (r) * (width) + (c);
  int block = n / 512;
  int pos = n % 512;
  return channel->pixels[block][pos];
}

void resize(image_32_512_t *src, image_t *dst, int newWidth, int newHeight)
{
  double sr = 0.0; // row scale
  double sc = 0.0; // column scale
  int height = 0;
  int width = 0;

  height = src->h;
  width = src->w;
  sr = (double)height / (double)newHeight;
  sc = (double)width / (double)newWidth;

  double rf = 0.0;
  for (int rNew = 0; rNew < newHeight; rNew++, rf += sr)
  {
    int r = (int)rf;
    r = (r > height - 2) ? height - 2 : r;
    double deltaR = rf - r;
    double oneMinusDeltaR = 1.0 - deltaR;

    double cf = 0.0;
    for (int cNew = 0; cNew < newWidth; cNew++, cf += sc)
    {
      int c = (int)cf;
      c = (c > width - 2) ? width - 2 : c;
      double deltaC = cf - c;
      double w1 = oneMinusDeltaR * (1.0 - deltaC);
      double w2 = deltaR * (1.0 - deltaC);
      double w3 = oneMinusDeltaR * deltaC;
      double w4 = deltaR * deltaC;

      double v = imgReadChannel(src, width, r, c) * w1 + imgReadChannel(src, width, r + 1, c) * w2 + imgReadChannel(src, width, r, c + 1) * w3 + imgReadChannel(src, width, r + 1, c + 1) * w4;

      imgWriteChannel(dst->pixels, newWidth, rNew, cNew, v);
    }
  }
}
