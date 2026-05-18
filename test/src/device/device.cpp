#include "device/device.h"
#include "common/context.h"

#if USE_TECO
#include "device/sdaa/memory.h"
#include "device/sdaa/version.h"
#include "device/sdaa/profiler.h"
#include "device/sdaa/common.h"
using namespace ::optest::sdaa;
#elif USE_CUDA
#include "device/cuda/memory.h"
#include "device/cuda/version.h"
#include "device/cuda/profiler.h"
#include "device/cuda/common.h"
using namespace ::optest::cuda;
#endif

namespace optest {

void scdaMalloc(void **p, size_t size) {
    if (Context::instance()->checkAllMemory()) {
        DeviceMemoryPool::instance()->deviceMalloc(p, size);
    } else {
        DeviceMemory::instance()->deviceMalloc(p, size);
    }
}
bool scdaFree(void *p) {
    if (Context::instance()->checkAllMemory()) {
        return DeviceMemoryPool::instance()->deviceFree(p);
    } else {
        return DeviceMemory::instance()->deviceFree(p);
    }
}
int scdaFreeCheck(scdaStream_t stream) {
    if (Context::instance()->checkAllMemory()) {
        return DeviceMemoryPool::instance()->check(stream);
    } else {
        return DeviceMemory::instance()->check(stream);
    }
}

void scdaDestroy() {
    if (Context::instance()->checkAllMemory()) {
        DeviceMemoryPool::instance()->destroy();
    } else {
        DeviceMemory::instance()->destroy();
    }
}

void showVersions() { printVersions(); }

namespace Profiler {
void start() { DeviceProfiler::instance()->start(); }
void end() { DeviceProfiler::instance()->end(); }
ProfilePerfInfo duration() { return DeviceProfiler::instance()->duration(); }
}  // namespace Profiler

}  // namespace optest
