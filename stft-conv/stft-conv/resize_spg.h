#include <stdint.h>

typedef struct
{
	double *pixels;
	uint32_t w;
	uint32_t h;
} image_t;

#define imgReadChannel(channel, width, r, c)        \
    channel[(r) * (width) + (c)]

#define imgWriteChannel(channel, width, r, c, v)    \
    (channel)[(r) * (width) + (c)] = (v)

void resize(image_t *src, image_t *dst, int newWidth, int newHeight)
{
	double           sr = 0.0;               // row scale
	double           sc = 0.0;               // column scale
	int          height = 0;
	int          width = 0;

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

			double v = imgReadChannel(src->pixels, width, r, c) * w1
				+ imgReadChannel(src->pixels, width, r + 1, c) * w2
				+ imgReadChannel(src->pixels, width, r, c + 1) * w3
				+ imgReadChannel(src->pixels, width, r + 1, c + 1) * w4;

			imgWriteChannel(dst->pixels, newWidth, rNew, cNew, v);
		}
	}
}
