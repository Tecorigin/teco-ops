#ifndef ZOO_TECO_FLATTEN_RAYS_H_  // NOLINT
#define ZOO_TECO_FLATTEN_RAYS_H_

#include "zoo/teco/executor.h"

namespace optest {
class FlattenRaysExecutor : public TecoExecutor {
 public:
    FlattenRaysExecutor() {}
    ~FlattenRaysExecutor() {}

    void paramCheck();
    void paramParse();
    void paramGeneration();
    void compute();
    void cpuCompute();
    void gpuCompute();
    int64_t getTheoryOps() override;
    int64_t getTheoryIoSize() override;
    void destroy();

 private:
    void *rays, *res;
    uint32_t N;
    uint32_t M;
};

}  // namespace optest

#endif
