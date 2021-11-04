#include "tensorflow/lite/experimental/microfrontend/lib/kiss_fft_common.h"

#define FIXED_POINT 16
namespace kissfft_fixed16
{
#include "kiss_fft.h"
#include "tools/kiss_fftr.h"
} // namespace kissfft_fixed16
#undef FIXED_POINT
