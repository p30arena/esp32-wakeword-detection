// stft-conv.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "resize_spg.h"

#define max_spec_val 340.0

void abs_vector(double *b, int len) {
	for (int i = 0; i < len; i++)
	{
		b[i] = abs(b[i]);
	}
}

void normalize_MEAN_SD(double *b, int len) {
	double sum = 0.0, mean, tmp_std = 0.0, std;
	int i;
	for (i = 0; i < len; ++i) {
		sum += b[i];
	}
	mean = sum / len;
	for (i = 0; i < len; ++i) {
		tmp_std += pow(b[i] - mean, 2);
	}
	std = sqrt(tmp_std / len);

	for (i = 0; i < len; ++i) {
		b[i] = (b[i] - mean) / std;
	}
}

void normalize(double *b, int len) {
	for (int i = 0; i < len; i++)
	{
		b[i] = b[i] >= max_spec_val ? 1 : b[i] / max_spec_val;
	}
}

int main(int argc, char *argv[])
{
	if (argc < 3) {
		std::cout << "stft-conv.exe 0/1/2 fin fout";
		exit(1);
	}

	// based on https://octave.sourceforge.io/signal/function/specgram.html
	const int ch = 1;
	const int rate = 16000;
	const int frame = 512;
	const int shift = 512;
	// out_size must eq rate
	const int out_size = 125 * 128; // 16000
	const int out_padding = 128 * 128 - out_size;
	int length;

	WAV input;
	FILE *output;
	STFT process(ch, frame, shift);

	input.OpenFile(argv[2]);
	if (argv[1][0] == '0' || argv[1][0] == '2') {
		output = stdout;
		_setmode(_fileno(stdout), _O_BINARY);
	}
	else {
		output = fopen(argv[3], "wb");
	}

	short buf_in[ch*shift];
	double **data;
	double buf_out[out_size + out_padding] = { 0 };
	double buf_img[32 * 32] = { 0 };

	data = new double*[ch];
	for (int i = 0; i < ch; i++) {
		data[i] = new double[frame + 2];
		memset(data[i], 0, sizeof(double)*(frame + 2));
	}

	int offset = 0;
	int cnt = 0;
	while (!input.IsEOF()) {
		length = input.ReadUnit(buf_in, shift*ch);
		if (length == 0) {
			break;
		}
		process.stft(buf_in, length, data);
		memcpy(&buf_out[offset], data[0], sizeof(double)*length);
		offset += length;
		cnt++;
	}

	for (int i = 0; i < ch; i++)
		delete[] data[i];
	delete[] data;

	if (argv[1][0] == '2') {
		// used for analysis
		fwrite(buf_out, sizeof(double), 128 * 128, output);
	}
	else {
		image_t srcImg = { buf_out, 128, 128 };
		image_t dstImg = { buf_img, 32, 32 };

		resize(&srcImg, &dstImg, 32, 32);
		abs_vector(buf_img, 32 * 32);
		normalize(buf_img, 32 * 32);

		fwrite(buf_img, sizeof(double), 32 * 32, output);
	}

	if (output != stdout) {
		fclose(output);
	}

	return 0;
}
