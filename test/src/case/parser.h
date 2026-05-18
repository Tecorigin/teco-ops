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
 
#ifndef CASE_PARSER_H_  // NOLINT
#define CASE_PARSER_H_
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <functional>
#include "case/pb_tools.h"
#include "case/criterion.h"
#include "test_proto/optest.pb.h"
#include "common/tools.h"
#include "common/context.h"

namespace optest {

// value source
enum ValueType {
    VALUE_F,
    VALUE_I,
    VALUE_L,
    VALUE_H,
    VALUE_BF,
    VALUE_RANDOM,
    VALUE_PATH,
    VALUE_INVALID,
};

enum DataType {
    DTYPE_INVALID = 0,
    DTYPE_HALF = 1,
    DTYPE_FLOAT = 2,
    DTYPE_DOUBLE = 3,
    DTYPE_INT8 = 4,
    DTYPE_INT16 = 5,
    DTYPE_INT32 = 6,
    DTYPE_INT64 = 7,
    DTYPE_UINT8 = 8,
    DTYPE_UINT16 = 9,
    DTYPE_UINT32 = 10,
    DTYPE_UINT64 = 11,
    DTYPE_BOOL = 12,
    DTYPE_COMPLEX_HALF = 13,
    DTYPE_COMPLEX_FLOAT = 14,
     DTYPE_BFLOAT16 = 15,
};

// OpTensorLayout_t,
enum TensorLayout {
    LAYOUT_NCHW = 0,
    LAYOUT_NHWC = 1,
    LAYOUT_HWCN = 2,
    LAYOUT_NDHWC = 3,
    LAYOUT_ARRAY = 4,
    LAYOUT_NCDHW = 5,
    LAYOUT_TNC = 6,
    LAYOUT_NTC = 7,
    LAYOUT_NC = 8,
    LAYOUT_NLC = 9,
    LAYOUT_NWHC = 10,
    LAYOUT_CHWN = 11,
    LAYOUT_CDHWN = 12,
};

enum TensorType {
    TENSOR = 0,
    FILTER = 1,
    SEQDATA = 2,
};

// only the tensor info saved in *pb.
struct MetaTensor {
    std::string name = "unknown";
    bool is_null = false;  // null means desc and ptr both null;

    // these size are for memory malloc
    size_t shape_count = 0;  // not include stride.
    // count is 0, means 0 elements and ptr is null.
    // but desc may not null.
    size_t total_count = 0;    // include stride_count
    size_t size_in_bytes = 0;  // size_in_bytes = total_count * sizeof(device_dtype)
    size_t sizeof_dtype = 0;

    // these for set to tensor desc
    testpt::DataType dtype = testpt::DTYPE_INVALID;      // TODO(maliang)
    testpt::TensorLayout layout = testpt::LAYOUT_ARRAY;  // TODO(maliang)
    testpt::TensorType ttype = testpt::TENSOR;           // TODO(maliang)

    // shape may not equal to total_count.
    // maybe total_count is 0 (for nullptr), but shape is not 0 element
    // for testing api foolproof
    std::vector<int> shape;
    std::vector<int> stride;

    std::unordered_map<std::string, std::unordered_map<std::string, double>> gpu_diffs_;

    ValueType value_type = VALUE_INVALID;
    inline bool empty() { return shape_count == 0 || total_count == 0; }
    inline bool null() { return is_null; }

    bool input_reuse = false;
    int inplace = -1;

    void print() {
        std::string shape_str = "[]";
        std::ostringstream shape_oss;
        if (!shape.empty()) {
            std::copy(shape.begin(), shape.end() - 1, std::ostream_iterator<int>(shape_oss, ","));
            shape_oss << shape.back();
            shape_str = "[" + shape_oss.str() + "]";
        }
        std::string stride_str = "[]";
        std::ostringstream stride_oss;
        if (!stride.empty()) {
            std::copy(stride.begin(), stride.end() - 1,
                      std::ostream_iterator<int>(stride_oss, ","));
            stride_oss << stride.back();
            stride_str = "[" + stride_oss.str() + "]";
        }
        std::string res = "";
        res += name + ":[";
        res += shape_str + ", ";
        res += stride_str + ", ";
        res += DataType_Name(dtype) + ", ";
        res += TensorLayout_Name(layout);
        res += "]";
        std::cout << res << std::endl;
    }
};

class Parser {
 public:
    Parser() {}
    virtual ~Parser();
    void parse(const std::string &file);

    inline const std::vector<MetaTensor> &inputs() { return inputs_; }
    inline const std::vector<MetaTensor> &outputs() { return outputs_; }
    inline MetaTensor *input(size_t index) { return &(inputs_.at(index)); }
    inline MetaTensor *output(size_t index) { return &(outputs_.at(index)); }
    inline MetaTensor *workspace() { return &(workspace_); }
    inline MetaTensor *reservespace() { return &(reservespace_); }

    void getInputTensorValue(size_t index, void *data, size_t count);
    void getOutputTensorValue(size_t index, void *data, size_t count);
    void getBaselineInputTensorValue(size_t index, void *data, size_t count);
    void getBaselineOutputTensorValue(size_t index, void *data, size_t count);

    // op params
    inline testpt::Node *node() { return proto_node_; }
    inline std::string getOpName() { return proto_node_->op_name(); }
    void getTestInfo();

    // else
    inline testpt::Device device() { return device_; }
    std::vector<int> threshold_use();
    bool common_threshold();
    bool check_threshold();
    inline std::set<Criterion> criterions() { return criterions_; }
    std::set<Criterion> criterions(int index, std::vector<int> criterions_use);

    MetaTensor &getMetaTensor(const std::string &name);
    MetaTensor &getMetaTensor(int index);
    inline int getInputNum() { return inputs_.size(); }
    inline int getOutputNum() { return outputs_.size(); }
    inline bool inputIsNull(int index) { return inputs_.at(index).is_null; }
    inline bool outputIsNull(int index) { return outputs_.at(index).is_null; }
    inline testpt::DataType getInputDataType(int index) { return inputs_.at(index).dtype; }
    inline testpt::DataType getOutputDataType(int index) { return outputs_.at(index).dtype; }
    inline testpt::TensorLayout getInputLayout(int index) { return inputs_.at(index).layout; }
    inline testpt::TensorLayout getOutputLayout(int index) { return outputs_.at(index).layout; }
    inline size_t getInputDataCount(int index) { return inputs_.at(index).shape_count; }
    inline size_t getOutputDataCount(int index) { return outputs_.at(index).shape_count; }
    inline int getInputDimSize(int index) { return inputs_.at(index).shape.size(); }
    inline int getOutputDimSize(int index) { return outputs_.at(index).shape.size(); }
    inline int getInputDimStrideSize(int index) { return inputs_.at(index).stride.size(); }
    inline int getOutputDimStrideSize(int index) { return outputs_.at(index).stride.size(); }
    inline void getInputData(int index, void *data) {
        getInputTensorValue(index, data, inputs_[index].total_count);
    }
    inline void getOutputData(int index, void *data) {
        getOutputTensorValue(index, data, outputs_[index].total_count);
    }
    inline void getInputDims(int index, int dim_size, int *dim_array, int *dim_stride = nullptr) {
        auto ts = inputs_.at(index);
        memcpy(dim_array, ts.shape.data(), ts.shape.size() * sizeof(int));
        if (dim_stride) {
            memcpy(dim_stride, ts.stride.data(), ts.stride.size() * sizeof(int));
        }
    }
    inline void getOutputDims(int index, int dim_size, int *dim_array, int *dim_stride = nullptr) {
        auto ts = outputs_.at(index);
        memcpy(dim_array, ts.shape.data(), ts.shape.size() * sizeof(int));
        if (dim_stride) {
            memcpy(dim_stride, ts.stride.data(), ts.stride.size() * sizeof(int));
        }
    }
    inline ValueType getInputValueType(int index) { return inputs_.at(index).value_type; }
    inline testpt::Node *getProtoNode() { return proto_node_; }

 private:
    testpt::Node *proto_node_ = nullptr;
    std::vector<MetaTensor> inputs_;
    std::vector<MetaTensor> outputs_;
    MetaTensor workspace_;
    MetaTensor reservespace_;
    std::set<Criterion> criterions_;
    std::string op_name_;
    std::string pb_path_;
    testpt::Device device_ = testpt::CPU;

    ValueType getValueType(const testpt::Tensor *t);
    DataType getDataType(const testpt::DataType dtype);
    TensorLayout getLayout(const testpt::TensorLayout layout);
    TensorType getTensorType(const testpt::TensorType ttype);
    // size_t getSizeOfDataType(DataType dtype);

    void getTensorValue(testpt::Tensor *pt, void *data, ValueType value_type, size_t count,
                        testpt::Value &value, std::string path);  // NOLINT
    void getTensorValueH(testpt::Tensor *pt, void *data, size_t count,
                         testpt::Value &value);  // NOLINT
    void getTensorValueF(const testpt::Tensor *pt, void *data, size_t count,
                         testpt::Value &value);  // NOLINT
    void getTensorValueI(const testpt::Tensor *pt, void *data, size_t count,
                         testpt::Value &value);  // NOLINT
    void getTensorValueL(const testpt::Tensor *pt, void *data, size_t count,
                         testpt::Value &value);  // NOLINT
    void getTensorValueBF(testpt::Tensor *pt, void *data, size_t count,
                          testpt::Value &value);  // NOLINT
    void getTensorValueRandom(testpt::Tensor *pt, void *data, size_t count);
    void getTensorValueByFile(testpt::Tensor *pt, void *data, size_t count, std::string path);

    void checkTensorValid(MetaTensor *mt, testpt::Tensor *t);
    void checkRandomParam(testpt::Tensor *t);

    // if no stride return shape count
    size_t getTensorStrideCount(testpt::Tensor *pt, ValueType type);
    // only return shape count, from value_size or shape
    size_t getTensorShapeCount(testpt::Tensor *pt);

    Formula cvtProtoEvaluationCriterion(testpt::EvaluationCriterion c);
    bool readMessageFromFile(const std::string &filename, testpt::Node *proto);
    size_t getTensorSize(testpt::Tensor *pt);
    void setCurPbPath(const std::string &file);
};

}  // namespace optest
#endif  // CASE_PARSER_H_  // NOLINT
