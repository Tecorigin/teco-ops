// MIT License
// 
// Copyright (c) 2024, Tecorigin Co., Ltd.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
 
#ifndef ZOO_teco_EXECUTE_H_  // NOLINT
#define ZOO_teco_EXECUTE_H_

#include <tecoops.h>
#include <vector>
#include "suite/executor.h"
#include "zoo/tecozoo.h"


namespace optest {
#define checkTECOOPS(func)                                                                        \
    {                                                                                             \
        tecoopsStatus_t status = func;                                                            \
        if (status != TECOOPS_STATUS_SUCCESS) {                                                   \
            printf("file:%s, func:%s, line: %d,TECOOPS Error:%d\n", __FILE__, __func__, __LINE__, \
                   status);                                                                       \
            tecoopsGetErrorString(status);                                                        \
        }                                                                                         \
    }

class TecoExecutor : public Executor {
 public:
    TecoExecutor();
    virtual ~TecoExecutor();
    virtual void getWorkspaceSize() {}
    virtual void getReservespaceSize() {}

 protected:
    tecoopsHandle_t handle_;
    size_t workSpaceSizeInBytes_ = 0;
    size_t reserveSpaceSizeInBytes_ = 0;
    void *workSpace_, *reserveSpace_;
    std::vector<void *> input_desc_;
    std::vector<void *> output_desc_;

    void setContext();

    void getNCHW(std::vector<int> shape, tecoopsTensorFormat_t format, int *N, int *C, int *H,
                 int *W);
    tecoopsTensorDescriptor_t createTensorDesc(MetaTensor *mt);

    void createDesc();
    void destroyDesc();

    template <typename T>
    T getInputDesc(int index) {
        return (T)input_desc_[index];
    }

    template <typename T>
    T getOutputDesc(int index) {
        return (T)output_desc_[index];
    }

    void workspaceMalloc();
    void workspaceFree();
};

}  // namespace optest

#endif  // ZOO_teco_EXECUTE_H_  // NOLINT
