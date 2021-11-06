// stft-conv.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


int main(int argc, char *argv[])
{
	if (argc != 3) {
		std::cout << "args < 2 - must pass wav and file path";
		exit(1);
	}

	const char* wavPath = argv[1];
	const char* filePath = argv[2];

	const int ch = 1;
	const int rate = 16000;
	const int frame = 1024;
	const int shift = 128;
	int length;

	WAV input;
	FILE *output;
	STFT process(ch, frame, shift);

	input.OpenFile(wavPath);
	output = fopen(filePath, "wb");

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
		char*b_ = (char*)&data[0][i];
		for (int j = 0; j < sizeof(double); j++) {
			fputc(b_[j], output);
		}
	}

	for (int i = 0; i < ch; i++)
		delete[] data[i];
	delete[] data;

	fclose(output);

	return 0;
}
