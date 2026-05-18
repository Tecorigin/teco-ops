#ifndef DEVICE_DEVICE_H_  // NOLINT
#define DEVICE_DEVICE_H_

#if USE_TECO
#include "device/sdaa/profiler.h"
#include "device/sdaa/memory.h"
#include "device/sdaa/helper.h"
#include "device/sdaa/common.h"
using namespace ::optest::sdaa;
#define initLDM(stream) 0
#define calcHash(stream, in_data, size, out_hash) 0
#define checkHBM(stream, in_data, size, value, result) 0
#define initHBM() 0

#elif USE_CUDA
#include "device/cuda/profiler.h"
#include "device/cuda/memory.h"
#include "device/cuda/common.h"
using namespace ::optest::cuda;

#define initLDM(stream) 0
#define calcHash(stream, in_data, size, out_hash) 0
#define checkHBM(stream, in_data, size, value, result) 0
#define initHBM() 0
#endif

namespace optest {
void scdaMalloc(void **p, size_t size);
bool scdaFree(void *p);
int scdaFreeCheck(scdaStream_t stream);
void scdaDestroy();

void showVersions();

namespace Profiler {
void start();
void end();
ProfilePerfInfo duration();
}  // namespace Profiler

}  // namespace optest

#endif  // DEVICE_DEVICE_H_  // NOLINT
