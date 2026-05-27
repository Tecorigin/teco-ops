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

#include "zoo/teco/convert.h"
#include <string>
#include "common/tecoallog.h"
namespace optest {
namespace convert {
    tecoopsDataType_t toTecoopsDataType(testpt::DataType dtype) {
  switch (dtype) {
    case testpt::DTYPE_HALF:
      return TECOOPS_DATA_HALF;
    case testpt::DTYPE_FLOAT:
      return TECOOPS_DATA_FLOAT;
    case testpt::DTYPE_INT8:
      return TECOOPS_DATA_INT8;
    case testpt::DTYPE_INT16:
      return TECOOPS_DATA_INT16;
    case testpt::DTYPE_INT32:
      return TECOOPS_DATA_INT32;
    case testpt::DTYPE_INT64:
      return TECOOPS_DATA_INT64;
    case testpt::DTYPE_UINT8:
      return TECOOPS_DATA_UINT8;
    case testpt::DTYPE_BOOL:
      return TECOOPS_DATA_BOOL;
    case testpt::DTYPE_DOUBLE:
      return TECOOPS_DATA_DOUBLE;
    case testpt::DTYPE_BFLOAT16:
      return TECOOPS_DATA_BFLOAT16;
    case testpt::DTYPE_COMPLEX_FLOAT:
      return TECOOPS_DATA_COMPLEX_FLOAT;
    case testpt::DTYPE_UINT16:
    case testpt::DTYPE_UINT32:
    case testpt::DTYPE_UINT64:
      ALLOG(ERROR) << "Don't support this dtype. Not supported now";
      throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    default:
      ALLOG(ERROR) << "Don't support this dtype.";
      throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
  }
  return TECOOPS_DATA_FLOAT;
}

tecoopsTensorFormat_t toTecoopsFormat(testpt::TensorLayout layout) {
  switch (layout) {
    case testpt::LAYOUT_NCHW:
      return TECOOPS_TENSOR_NCHW;
    case testpt::LAYOUT_NHWC:
      return TECOOPS_TENSOR_NHWC;
    case testpt::LAYOUT_CHWN:
      return TECOOPS_TENSOR_CHWN;
    case testpt::LAYOUT_NWHC:
      return TECOOPS_TENSOR_NWHC;
    case testpt::LAYOUT_NDHWC:
      return TECOOPS_TENSOR_NDHWC;
    case testpt::LAYOUT_NCDHW:
    case testpt::LAYOUT_CDHWN:
      return TECOOPS_TENSOR_CDHWN;
    case testpt::LAYOUT_ARRAY:
    case testpt::LAYOUT_HWCN:
    case testpt::LAYOUT_TNC:
    case testpt::LAYOUT_NTC:
    case testpt::LAYOUT_NLC:
    case testpt::LAYOUT_NC:
      ALLOG(ERROR) << "Don't support this layout. Not supported now";
      break;
    default:
      throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
  }
  return TECOOPS_TENSOR_NCHW;
}
}  // namespace convert
}  // namespace optest
