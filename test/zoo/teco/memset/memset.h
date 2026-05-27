#ifndef ZOO_TECO_MEMSET_MEMSET_H_  // NOLINT
#define ZOO_TECO_MEMSET_MEMSET_H_

#include "zoo/teco/executor.h"

namespace optest {

class MemsetExecutor : public TecoExecutor {
 public:
    MemsetExecutor() {}
    ~MemsetExecutor() {}

    void paramCheck();
    void paramParse();
    void paramGeneration();
    void compute();
    void cpuCompute();
    int64_t getTheoryOps() override;
    int64_t getTheoryIoSize() override;

 private:
    void *x_;
    int value_;
    size_t size_;
};
};  // namespace optest

#endif  // ZOO_TECO_MEMSET_MEMSET_H_  // NOLINT
