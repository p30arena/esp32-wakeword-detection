// mfcc-conv.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "compute-mfcc-master/mfcc.h"

// Process each file
int processFile(MFCC &mfccComputer, const char* wavPath, const char* mfcPath) {
	// Initialise input and output streams    
	FILE *wavFp;
	FILE *mfcFp;

	// Check if input is readable
	wavFp = fopen(wavPath, "rb");

	// Check if output is writable
	if (mfcPath == NULL) {
		mfcFp = stdout;
	}
	else {
		mfcFp = fopen(mfcPath, "wb");
	}

	// Extract and write features
	if (mfccComputer.process(wavFp, mfcFp))
		std::cerr << "Error processing " << wavPath << std::endl;

	fclose(wavFp);
	if (mfcPath != NULL) {
		fclose(mfcFp);
	}
	return 0;
}

// Main
int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cout << "mfcc-conv.exe 0/1 fin fout";
		exit(1);
	}

	char *wavPath = argv[2];

	// Assign variables
	int numCepstra = 12;
	int numFilters = 40;
	int samplingRate = 16000;
	int winLength = 25;
	int frameShift = 10;
	int lowFreq = 50;
	int highFreq = samplingRate / 2;

	// Initialise MFCC class instance
	MFCC mfccComputer(samplingRate, numCepstra, winLength, frameShift, numFilters, lowFreq, highFreq);

	if (argv[1][0] == '0') {
		_setmode(_fileno(stdout), _O_BINARY);
	}

	// Process wav files
	if (processFile(mfccComputer, wavPath, argv[1][0] == '0' ? NULL : argv[3]))
		return 1;

	return 0;
}
