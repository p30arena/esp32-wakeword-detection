#include <stdint.h>

typedef struct
{
	double *pixels;
	uint32_t w;
	uint32_t h;
} image_t;

double getpixel(image_t *image, unsigned int x, unsigned int y)
{
	return image->pixels[(y * image->w) + x];
}

float max(float a, float b) { return (a < b) ? a : b; };
double lerp(double s, double e, double t) { return s + (e - s) * t; }
double blerp(double c00, double c10, double c01, double c11, double tx, double ty)
{
	return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

void putpixel(image_t *image, unsigned int x, unsigned int y, double color)
{
	image->pixels[(y * image->w) + x] = color;
}

void resize(image_t *src, image_t *dst, int newWidth, int newHeight)
{
	const float xScale = (float)newWidth * src->w;
	const float yScale = (float)newHeight * src->h;
	int x, y;
	for (x = 0, y = 0; y < newHeight; x++)
	{
		if (x > newWidth)
		{
			x = 0;
			y++;
		}

		// Image should be clamped at the edges and not scaled.
		//float gx = max(x / (float)(xScale - 0.5f), (float)(src->w - 1));
		//float gy = max(y / (float)(yScale - 0.5f), (float)(src->h - 1));
		float gx = x / (xScale + 0.5f);
		float gy = y / (yScale + 0.5f);
		int gxi = (int)gx;
		int gyi = (int)gy;
		double result = 0;
		double c00 = getpixel(src, gxi, gyi);
		double c10 = getpixel(src, gxi + 1, gyi);
		double c01 = getpixel(src, gxi, gyi + 1);
		double c11 = getpixel(src, gxi + 1, gyi + 1);

		result = blerp(c00, c10, c01, c11, gx - gxi, gy - gyi);

		putpixel(dst, x, y, result);
	}
}
