// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#ifndef TECOOPS_UAL_OPS_BASE_OP_HPP_
#define TECOOPS_UAL_OPS_BASE_OP_HPP_

#include <type_traits>
#include <cassert>
#include "ual/com/log.h"

namespace tecoops {
namespace ual {
namespace ops {

#define RUN_KERNEL(func_ptr, stream_id, arg) \
    ({                                       \
        func_ptr<<<1, stream_id>>>(arg);     \
        0;                                   \
    })

template <typename T, typename Ty>
struct BaseOp {
 public:
    using ArgsType = typename Ty::ArgsType;
    using PatchType = typename Ty::PatchType;
    using RetType = typename Ty::RetType;
    using PImplType = typename Ty::PImplType;

    BaseOp() = default;
    ~BaseOp() = default;
    BaseOp(const BaseOp &other) = delete;
    BaseOp(BaseOp &&other) = delete;
    BaseOp &operator=(const BaseOp &other) = delete;
    BaseOp &operator=(BaseOp &&other) = delete;

    common::Status find(const PatchType *args) {
        return static_cast<T *>(this)->findImpl(args);
    }

    __attribute__((noinline, used)) common::Status run(const ArgsType *arg,
                                                        sdaaStream_t stream_id) {
        if (instance_ == nullptr || discription_ == nullptr) {
            ERROR("input args is bad param!");
            return common::Status::BAD_PARAMETER;
        }
        RUN_KERNEL(instance_, stream_id, *arg);
        return common::Status::SUCCESS;
    }

    void setInstance(PImplType instance, const char *discription) {
        instance_ = instance;
        discription_ = discription;
    }

 private:
    PImplType instance_ = nullptr;
    const char *discription_ = nullptr;
};

}  // namespace ops
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_OPS_BASE_OP_HPP_