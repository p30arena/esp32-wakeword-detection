#ifndef _H_AFTER_PROCESSOR_
#define _H_AFTER_PROCESSOR_

#include <cstdint>
#include <cstdlib>
#include <cstring>

class PostProcessor
{
private:
  uint32_t frame_size;
  uint32_t shift_size;
  uint32_t channels;
  // frame_size / shift_size;
  uint32_t num_block;
  short *output;
  double **buf;
  uint32_t buf_offset;

public:
  inline PostProcessor(uint32_t _frame_size,
                       uint32_t _shift_size,
                       uint32_t _channels);
  inline ~PostProcessor();

  inline short *Overlap(double **in);
  inline short *Overlap(double *in);
  inline short *Array2WavForm(double **in);

  inline short *Frame2Wav(double *in);
};

#endif
