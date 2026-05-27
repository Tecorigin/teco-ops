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

#include "zoo/teco/executor.h"
#include <string>
#include "zoo/teco/convert.h"


namespace optest {
TecoExecutor::TecoExecutor() { tecoopsCreate(&handle_); }
TecoExecutor::~TecoExecutor() { tecoopsDestroy(handle_); }

void TecoExecutor::setContext() {
    if (exe_context_ == nullptr) {
        ALLOG(ERROR) << "exe_context_ is nullptr";
        throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    } else {
        checkTECOOPS(tecoopsSetStream(handle_, exe_context_->stream));
    }
}

void TecoExecutor::getNCHW(std::vector<int> shape, tecoopsTensorFormat_t format, int *N, int *C,
                          int *H, int *W) {
    switch (format) {
        case TECOOPS_TENSOR_NCHW: {
            *N = shape[0];
            *C = shape[1];
            *H = shape[2];
            *W = shape[3];
            break;
        }
        case TECOOPS_TENSOR_NHWC: {
            *N = shape[0];
            *H = shape[1];
            *W = shape[2];
            *C = shape[3];
            break;
        }
        case TECOOPS_TENSOR_CHWN: {
            *C = shape[0];
            *H = shape[1];
            *W = shape[2];
            *N = shape[3];
            break;
        }
        case TECOOPS_TENSOR_NWHC: {
            *N = shape[0];
            *W = shape[1];
            *H = shape[2];
            *C = shape[3];
            break;
        }
        default:
            ALLOG(ERROR) << "Don't support this layout.";
            throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }
}
tecoopsTensorDescriptor_t TecoExecutor::createTensorDesc(MetaTensor *mt) {
    if (unlikely((mt->null()))) {
        ALLOG(WARNING) << "TecoExecutor: skip creating tensor " << mt->name << ", set it as nullptr.";
        // if don't have this tensor, set it as nullptr;
        // push an desc as nullptr, and is_output marked as false.
        return nullptr;
    }

    tecoopsTensorDescriptor_t desc = nullptr;
    tecoopsDataType_t dataType = convert::toTecoopsDataType(mt->dtype);

    checkTECOOPS(tecoopsCreateTensorDescriptor(&desc));
    if (mt->shape.size() == 4 && mt->layout != testpt::LAYOUT_ARRAY) {
        tecoopsTensorFormat_t format = convert::toTecoopsFormat(mt->layout);
        int N = 1, C = 1, H = 1, W = 1;
        getNCHW(mt->shape, format, &N, &C, &H, &W);
        checkTECOOPS(tecoopsSetTensor4dDescriptor(desc, format, dataType, N, C, H, W));
    } else {
        checkTECOOPS(tecoopsSetTensorNdDescriptor(desc, dataType, mt->shape.size(),
                                                  mt->shape.data(), mt->stride.data()))
    }
    return desc;
}

void TecoExecutor::createDesc() {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *mt = parser_->input(i);
        switch (mt->ttype) {
            case testpt::TENSOR: input_desc_.push_back((void *)createTensorDesc(mt)); break;
            /*case testpt::FILTER: input_desc_.push_back((void *)createFilterDesc(mt)); break;*/
            case testpt::VALID: input_desc_.push_back(nullptr); break;
            default:
                ALLOG(ERROR) << "Don't support this ttype.";
                throw std::invalid_argument(std::string(__FILE__) + " +" +
                                            std::to_string(__LINE__));
        }
    }

    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *mt = parser_->output(i);
        switch (mt->ttype) {
            case testpt::TENSOR: output_desc_.push_back(createTensorDesc(mt)); break;
            /*case testpt::FILTER: output_desc_.push_back(createFilterDesc(mt)); break;*/
            case testpt::VALID: output_desc_.push_back(nullptr); break;
            default:
                ALLOG(ERROR) << "Don't support this ttype.";
                throw std::invalid_argument(std::string(__FILE__) + " +" +
                                            std::to_string(__LINE__));
        }
    }
}

void TecoExecutor::destroyDesc() {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *mt = parser_->input(i);
        switch (mt->ttype) {
            case testpt::TENSOR:
                checkTECOOPS(
                    tecoopsDestroyTensorDescriptor((tecoopsTensorDescriptor_t)input_desc_[i]));
                break;
            /*case testpt::FILTER:
                checkTECOOPS(
                    tecoopsDestroyFilterDescriptor((tecoopsFilterDescriptor_t)input_desc_[i]));
                break;*/
            case testpt::VALID: break;
            default:
                ALLOG(ERROR) << "Don't support this ttype.";
                throw std::invalid_argument(std::string(__FILE__) + " +" +
                                            std::to_string(__LINE__));
        }
        input_desc_[i] = nullptr;
    }

    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *mt = parser_->output(i);
        switch (mt->ttype) {
            case testpt::TENSOR:
                checkTECOOPS(
                    tecoopsDestroyTensorDescriptor((tecoopsTensorDescriptor_t)output_desc_[i]));
                break;
            /*case testpt::FILTER:
                checkTECOOPS(
                    tecoopsDestroyFilterDescriptor((tecoopsFilterDescriptor_t)output_desc_[i]));
                break;*/
            case testpt::VALID: break;
            default:
                ALLOG(ERROR) << "Don't support this ttype.";
                throw std::invalid_argument(std::string(__FILE__) + " +" +
                                            std::to_string(__LINE__));
        }
        output_desc_[i] = nullptr;
    }
}

void TecoExecutor::workspaceMalloc() {
    if (parser_->getProtoNode()->has_workspace()) {
        workSpaceSizeInBytes_ = parser_->workspace()->total_count;
    } else {
        getWorkspaceSize();
    }
    ALLOG(VLOG) << "workspace_size = " << workSpaceSizeInBytes_;
    if (workSpaceSizeInBytes_ != 0) scdaMalloc(&workSpace_, workSpaceSizeInBytes_);

    if (parser_->getProtoNode()->has_reservespace()) {
        reserveSpaceSizeInBytes_ = parser_->reservespace()->total_count;
    } else {
        getReservespaceSize();
    }
    if (reserveSpaceSizeInBytes_ != 0) scdaMalloc(&reserveSpace_, reserveSpaceSizeInBytes_);
}



void TecoExecutor::workspaceFree() {
    if (workSpaceSizeInBytes_ != 0) {
        if (!scdaFree(workSpace_)) {
            ALLOG(ERROR) << "workspace memory out of bounds!!!\n";
            ADD_FAILURE() << "workspace memory out of bounds!!!\n";
            eva_res_.status = TECOTEST_STATUS_MEMORY_OUT_ERROR1;
        }
    }
    if (reserveSpaceSizeInBytes_ != 0) {
        if (!scdaFree(reserveSpace_)) {
            ALLOG(ERROR) << "reservespace memory out of bounds!!!\n";
            ADD_FAILURE() << "reservespace memory out of bounds!!!\n";
            eva_res_.status = TECOTEST_STATUS_MEMORY_OUT_ERROR1;
        }
    }
}

}  // namespace optest
