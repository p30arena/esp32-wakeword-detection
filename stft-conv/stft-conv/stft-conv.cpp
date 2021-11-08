// stft-conv.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "resize_spg.h"


int main(int argc, char *argv[])
{
	if (argc < 3) {
		std::cout << "stft-conv.exe 0/1 fin fout";
		exit(1);
	}

	// based on https://octave.sourceforge.io/signal/function/specgram.html
	const int ch = 1;
	const int rate = 16000;
	const int frame = 512;
	const int shift = 512;
	int length;

	WAV input;
	FILE *output;
	STFT process(ch, frame, shift);

	input.OpenFile(argv[2]);
	if (argv[1][0] == '0') {
		output = stdout;
		_setmode(_fileno(stdout), _O_BINARY);
	}
	else {
		output = fopen(argv[3], "wb");
	}

	short buf_in[ch*shift];
	double **data;
	double buf_out[128 * 128] = { 0 };
	double buf_img[32 * 32] = { 0 };

	data = new double*[ch];
	for (int i = 0; i < ch; i++) {
		data[i] = new double[frame + 2];
		memset(data[i], 0, sizeof(double)*(frame + 2));
	}

	int offset = 0;
	while (!input.IsEOF()) {
		length = input.ReadUnit(buf_in, shift*ch);
		if (length == 0) {
			break;
		}
		process.stft(buf_in, length, data);
		memcpy(&buf_out[offset], data[0], sizeof(double)*length);
		offset += length;
	}

	for (int i = 0; i < ch; i++)
		delete[] data[i];
	delete[] data;

	image_t srcImg = { buf_out, 128, 128 };
	image_t dstImg = { buf_img, 32, 32 };
	resize(&srcImg, &dstImg, 32, 32);
	fwrite(buf_img, sizeof(double), 32 * 32, output);
	//fwrite(buf_out, sizeof(double), 128 * 128, output);

	if (output != stdout) {
		fclose(output);
	}

	return 0;
}
