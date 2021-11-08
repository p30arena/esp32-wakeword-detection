// stft-conv.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


int main(int argc, char *argv[])
{
	if (argc < 3) {
		std::cout << "stft-conv.exe 0/1 fin fout";
		exit(1);
	}

	stdout;

	// based on https://octave.sourceforge.io/signal/function/specgram.html
	const int ch = 1;
	const int rate = 16000;
	const int frame = 1024;
	const int shift = 560;
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

	data = new double*[ch];
	for (int i = 0; i < ch; i++) {
		data[i] = new double[frame + 2];
		memset(data[i], 0, sizeof(double)*(frame + 2));
	}

	while (!input.IsEOF()) {
		length = input.ReadUnit(buf_in, shift*ch);
		if (length == 0) {
			break;
		}
		process.stft(buf_in, length, data);
	}

	for (int i = 0; i < frame; i++) {
		data[0][i] /= 32768;
	}

	fwrite(data[0], sizeof(double), frame, output);

	for (int i = 0; i < ch; i++)
		delete[] data[i];
	delete[] data;

	if (output != stdout) {
		fclose(output);
	}

	return 0;
}
