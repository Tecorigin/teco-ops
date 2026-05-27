#ifndef ZOO_TECO_REDUCE_VARIANCE_REDUCE_VARIANCE_H_  // NOLINT
#define ZOO_TECO_REDUCE_VARIANCE_REDUCE_VARIANCE_H_

#include "interface/include/tecoops.h"
#include "zoo/teco/executor.h"


namespace optest {

class ReduceVarianceExecutor : public TecoExecutor {
 public:
    ReduceVarianceExecutor() {}
    ~ReduceVarianceExecutor() {}

    void paramCheck();
    void paramParse();
    void paramGeneration();
    void compute();
    void cpuCompute();
    void gpuCompute();
    int64_t getTheoryOps() override;
    int64_t getTheoryIoSize() override;

 private:
    int axis_ = 0;
    int correction_ = 0;
    tecoopsTensorDescriptor_t xDesc_;
    const void *x_;
    tecoopsTensorDescriptor_t yDesc_;
    void *y_;
};
};  // namespace optest

#endif  // ZOO_TECO_REDUCE_VARIANCE_REDUCE_VARIANCE_H_  // NOLINT