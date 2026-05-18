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

#include <vector>
#include <utility>
#include <unordered_map>
#include <functional>
#include <string>
#include "case/parser.h"

namespace optest {

Parser::~Parser() {
    if (proto_node_ != nullptr) {
        delete proto_node_;
        proto_node_ = nullptr;
    }
}

void Parser::parse(const std::string &file) {
    proto_node_ = new testpt::Node;
    setCurPbPath(file);
    GTEST_CHECK_CUSTOM(readMessageFromFile(file, proto_node_), ParsePtFault,
                       "Parser: parse *pb/*prototxt failed.");
    GTEST_CHECK_CUSTOM(proto_node_->has_op_name(), ParsePtFault,
                       "Parser: missing op name in prototxt.");

    // 1.get device  todo(maliang)
    if (proto_node_->has_device()) {
        device_ = proto_node_->device();
    } else if (proto_node_->has_test_param()) {
        if (proto_node_->mutable_test_param()->has_baseline_device()) {
            device_ = proto_node_->mutable_test_param()->baseline_device();
        } else {
            device_ = testpt::Device::CPU;  // default cpu
        }
    } else {
        device_ = testpt::Device::CPU;  // default cpu
    }

    // 2.get criterion
    criterions_.clear();
    if (proto_node_->has_test_param()) {
        auto test_param = proto_node_->mutable_test_param();
        // GTEST_CHECK_CUSTOM(test_param->error_func_size() != 0);
        GTEST_CHECK_CUSTOM(test_param->error_func_size() == test_param->error_threshold_size(),
                           ParsePtFault,
                           "Parser: error_func's number should equal to error_threshold's "
                           "number, now they are not equal.");
        GTEST_CHECK_CUSTOM((test_param->golden_threshold_size() == test_param->error_func_size() ||
                            test_param->golden_threshold_size() == 0),
                           ParsePtFault,
                           "Parser: error_func's number should equal to error_threshold's "
                           "number, now they are not equal.");
        GTEST_CHECK_CUSTOM((test_param->calc_eps_size() == test_param->error_func_size() ||
                            test_param->calc_eps_size() == 0),
                           ParsePtFault,
                           "Parser: error_func's number should equal to error_threshold's "
                           "number, now they are not equal.");

        auto num = test_param->error_func_size();
        for (int i = 0; i < num; ++i) {
            auto func = cvtProtoEvaluationCriterion(test_param->error_func(i));
            optest::Criterion criterion = optest::Criterion();
            criterion.formula = func;
            criterion.enable = true;
            criterion.error_threshold = test_param->error_threshold(i);
            criterion.golden_eps = criterion.error_threshold;
            if (test_param->golden_threshold_size() > 0) {
                criterion.golden_threshold = test_param->golden_threshold(i);
            }

            if (func == DIFF3) {
                if (test_param->ratio_threshold_size() > 0) {
                    criterion.ratio_threshold = test_param->ratio_threshold(0);
                }
            }

            if (test_param->calc_eps_size() > 0) {
                criterion.calc_eps = test_param->calc_eps(i);
            } else {
                criterion.calc_eps = 0.0;
            }

            criterions_.insert(std::move(criterion));
        }
    }

    // 3. inputs/outputs
    auto parse_tensor = [=](MetaTensor *mt, testpt::Tensor *pt) {
        mt->is_null = (pt->id().find("NULL") != std::string::npos) ? true : false;
        if (unlikely(mt->is_null)) {
            ALLOG(WARNING) << "Found tensor is null, skip parsing else data.";
            return;  // if null, don't need parse other info.
        }
        mt->name = pt->id();
        mt->value_type = getValueType(pt);
        mt->input_reuse = pt->reused();
        mt->inplace = pt->inplace();
        // 1.shape set to tensor desc
        GTEST_CHECK_CUSTOM(pt->has_shape(), ParsePtFault,
                           "Parser: missing tensor shape in prototxt.");
        mt->shape.resize(pt->mutable_shape()->dims_size());
        for (size_t i = 0; i < mt->shape.size(); ++i) {
            mt->shape[i] = pt->mutable_shape()->dims(i);
        }

        mt->stride.resize(pt->mutable_shape()->dim_stride_size());
        for (size_t i = 0; i < mt->stride.size(); ++i) {
            mt->stride[i] = pt->mutable_shape()->dim_stride(i);
        }

        // 2.else info
        mt->dtype = pt->dtype();    // getDataType(pt->dtype());
        mt->layout = pt->layout();  // getLayout(pt->layout());
        mt->ttype = pt->ttype();    // getTensorType(pt->ttype());

        // 3.size to malloc memory. (shape may not equal to size)
        // stride_count include stride, if no stride stride_count == shape_count
        mt->total_count = getTensorStrideCount(pt, mt->value_type);
        // shape_count come from value_f/value_i/value_h and shape.
        // not include stride
        mt->shape_count = getTensorShapeCount(pt);
        mt->sizeof_dtype = getSizeOfDataType(mt->dtype);
        mt->size_in_bytes = mt->total_count * mt->sizeof_dtype;

        // get gpu diffs
        mt->gpu_diffs_.clear();
        if (Context::instance()->reserveDataFlag()) {
            std::string filename = subReplaceFirst(file, ".prototxt", "/cuda_cpu_diff.json");
            if (filename != "") {
                struct stat buf;
                if (stat(filename.c_str(), &buf) != -1) {
                    try {
                        std::ifstream in(filename);
                        json info;
                        in >> info;
                        in.close();

                        auto data = info[mt->name];
                        for (auto it = data.begin(); it != data.end(); ++it) {
                            std::string formula = (std::string)(it.key());
                            auto diff = it.value();
                            std::unordered_map<std::string, double> diff_value;
                            for (auto iter = diff.begin(); iter != diff.end(); ++iter) {
                                // // nan and inf --> null, so do not know inf or nan
                                // if (iter.value().is_null()) {
                                //     diff_value.emplace((std::string)(iter.key()), (double)(0.0 /
                                //     0.0));
                                // } else {
                                diff_value.emplace((std::string)(iter.key()),
                                                   (double)(iter.value()));
                                // }
                            }

                            mt->gpu_diffs_.emplace(formula, diff_value);
                        }
                    } catch (std::exception &e) {
                        ALLOG(WARNING) << "catched " << e.what() << " in read cuda_cpu_diff.json.";
                        mt->gpu_diffs_.clear();
                        remove(filename.c_str());
                    }
                }
            }
        }

        // 4.check
        checkTensorValid(mt, pt);
    };

    inputs_.resize(proto_node_->input_size());
    for (size_t i = 0; i < proto_node_->input_size(); ++i) {
        parse_tensor(&inputs_[i], proto_node_->mutable_input(i));
    }

    outputs_.resize(proto_node_->output_size());
    for (size_t i = 0; i < proto_node_->output_size(); ++i) {
        parse_tensor(&outputs_[i], proto_node_->mutable_output(i));
    }
    if (proto_node_->has_workspace())
        parse_tensor(&workspace_, (testpt::Tensor *)&proto_node_->workspace());
    if (proto_node_->has_reservespace())
        parse_tensor(&reservespace_, (testpt::Tensor *)&proto_node_->reservespace());
}

// check if tensor value is equal to shape.
// when found value_i value_f value_h, just check value_size and shape_size
// when found random_param, check random param
// here allow shape_size != data size saved in pb
// cuz shape_size is for create tensor and data_size is for malloc space
// so just print warning.
void Parser::checkTensorValid(MetaTensor *mt, testpt::Tensor *pt) {
    if (mt->size_in_bytes >= (1ULL << 32)) {
        ALLOG(ERROR) << "Tensor size is too large, size=0x" << std::hex << mt->size_in_bytes;
        throw ParsePtFault(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }
    int shape_count = 1;
    switch (mt->value_type) {
        case VALUE_F:
        case VALUE_I:
        case VALUE_L:
        case VALUE_H:
            shape_count = std::accumulate(mt->shape.begin(), mt->shape.end(), shape_count,
                                          std::multiplies<int>());
            GTEST_WARNING(mt->shape_count == shape_count,
                          "Parser: found shape count is not equal to value "
                          "size.(mt->shape_count is value_size)");
            break;
        case VALUE_RANDOM:
            if (!mt->empty()) {
                checkRandomParam(pt);
            }
            break;
        case VALUE_PATH: {
            // if found path(only) in pb, but can't access this
            // path, throw.
            auto cur_pb_path = pb_path_ + pt->prev_path();
            GTEST_CHECK_CUSTOM((access(cur_pb_path.c_str(), 4) != -1), ParsePtFault,
                               "Parser: open path saved in *prototxt failed.");
            break;
        }
        case VALUE_INVALID:
            // check output, if may shape empty, value is empty, and not random
            // param, so don't need check.
            break;
        default: {
            GTEST_CHECK_CUSTOM(false, ParsePtFault,
                               "Parser: got unsupported value type, parse tensor failed.");
        }
    }
}

void Parser::checkRandomParam(testpt::Tensor *pt) {
    GTEST_CHECK_CUSTOM(pt->has_random_data(), ParsePtFault,
                       "Parser: missing random param of tensor saved in *pb");
    GTEST_CHECK_CUSTOM(pt->mutable_random_data()->has_distribution(), ParsePtFault,
                       "Parser: missing distribution of random param saved in *pb");

    auto random_data = pt->mutable_random_data();
    if (random_data->distribution() == testpt::UNIFORM ||
        random_data->distribution() == testpt::UNIQUE) {
        GTEST_CHECK_CUSTOM(random_data->has_seed(), ParsePtFault,
                           "Parser: missing seed of UNIFORM/UNIQUE random param in *pb");
        GTEST_CHECK_CUSTOM(random_data->has_upper_bound(), ParsePtFault,
                           "Parser: missing upper bound of UNIFORM/UNIQUE random param in *pb");
        GTEST_CHECK_CUSTOM(random_data->has_lower_bound(), ParsePtFault,
                           "Parser: missing lower bound of UNIFORM/UNIQUE random param in *pb");
    } else if (random_data->distribution() == testpt::GAUSSIAN) {
        GTEST_CHECK_CUSTOM(random_data->has_seed(), ParsePtFault,
                           "Parser: missing seed of GAUSSIAN random param in *pb");
        GTEST_CHECK_CUSTOM(random_data->has_mu(), ParsePtFault,
                           "Parser: missing mu of GAUSSIAN random param in *pb");
        GTEST_CHECK_CUSTOM(random_data->has_sigma(), ParsePtFault,
                           "Parser: missing sigma of GAUSSIAN random param in *pb");
    } else {
        GTEST_CHECK_CUSTOM(false, ParsePtFault,
                           "Parser: got unsupported distribution when check tensor "
                           "valid.");
    }
}

// return value's dtype is according to value type.
// if value type is value_*, return dtype is dtype in proto.
// if value type is random, return dtype is fp32
void Parser::getInputTensorValue(size_t index, void *data, size_t count) {
    testpt::Tensor *pt = proto_node_->mutable_input(index);
    testpt::Value value;
    if (pt->has_prev_value()) value = pt->prev_value();
    std::string path = "";
    if (pt->has_prev_path()) path = pt->prev_path();
    getTensorValue(pt, data, inputs_[index].value_type, count, value, path);
}

// return value's dtype is according to value type.
// if value type is value_*, return dtype is dtype in proto.
// if value type is random, return dtype is fp32
void Parser::getOutputTensorValue(size_t index, void *data, size_t count) {
    testpt::Tensor *pt = proto_node_->mutable_output(index);
    testpt::Value value;
    if (pt->has_value()) value = pt->value();
    std::string path = "";
    if (pt->has_path()) path = pt->path();
    getTensorValue(pt, data, outputs_[index].value_type, count, value, path);
}

void Parser::getBaselineInputTensorValue(size_t index, void *data, size_t count) {
    testpt::Tensor *pt = proto_node_->mutable_input(index);
    testpt::Value value;
    if (pt->has_value()) value = pt->value();
    std::string path = "";
    if (pt->has_path()) path = pt->path();
    getTensorValue(pt, data, inputs_[index].value_type, count, value, path);
}

void Parser::getBaselineOutputTensorValue(size_t index, void *data, size_t count) {
    testpt::Tensor *pt = proto_node_->mutable_output(index);
    testpt::Value value;
    if (pt->has_value()) value = pt->value();
    std::string path = "";
    if (pt->has_path()) path = pt->path();
    getTensorValue(pt, data, outputs_[index].value_type, count, value, path);
}

// get value from field value_f
// but this way have precision problem
// we will abandon value_f
void Parser::getTensorValueF(const testpt::Tensor *pt, void *data, size_t count,
                             testpt::Value &value) {
    GTEST_CHECK_CUSTOM(value.value_f_size() == count, ParsePtFault,
                       "Parser: when read value_f, expected element num is not equal "
                       "to real element num.");
    switch (pt->dtype()) {
        case testpt::DTYPE_FLOAT:
            for (int i = 0; i < count; ++i) {
                ((float *)data)[i] = value.value_f(i);
            }
            break;
        case testpt::DTYPE_HALF:
            for (int i = 0; i < count; ++i) {
                ((int16_t *)data)[i] = cvtFloatToHalf(value.value_f(i));
            }
            break;
        default:
            GTEST_CHECK_CUSTOM(false, ParsePtFault,
                               "Parser: found unsuppored dtype in value_f, value_f only "
                               "supporte float/half.");
    }
}

// get value from value_i
// no quant intx, saved in value_i
void Parser::getTensorValueI(const testpt::Tensor *pt, void *data, size_t count,
                             testpt::Value &value) {
    GTEST_CHECK_CUSTOM(value.value_i_size() == count, ParsePtFault,
                       "Parser: when read value_i, expected element num is not "
                       "equal to real element num.");

    switch (pt->dtype()) {
        case testpt::DTYPE_INT8:
            for (int i = 0; i < count; ++i) {
                ((int8_t *)data)[i] = value.value_i(i);
            }
            break;
        case testpt::DTYPE_UINT8:
            for (int i = 0; i < count; ++i) {
                ((uint8_t *)data)[i] = value.value_i(i);
            }
            break;
        case testpt::DTYPE_INT16:
            for (int i = 0; i < count; ++i) {
                ((int16_t *)data)[i] = value.value_i(i);
            }
            break;
        case testpt::DTYPE_INT32:
            for (int i = 0; i < count; ++i) {
                ((int32_t *)data)[i] = value.value_i(i);
            }
            break;
        case testpt::DTYPE_INT64:
            for (int i = 0; i < count; ++i) {
                ((int64_t *)data)[i] = value.value_i(i);
            }
            break;
        case testpt::DTYPE_BOOL:  // parser value_i == BOOL
            for (int i = 0; i < count; ++i) {
                ((int8_t *)data)[i] = value.value_i(i);
            }
            break;
        case testpt::DTYPE_HALF:
            for (int i = 0; i < count; ++i) {
                ((int16_t *)data)[i] = value.value_i(i);
            }
            break;
        case testpt::DTYPE_FLOAT:
            for (int i = 0; i < count; ++i) {
                int value_i = value.value_i(i);
                ((float *)data)[i] = *((float *)(&value_i));
            }
            break;
        case testpt::DTYPE_COMPLEX_HALF:
            for (int i = 0; i < 2 * count; i += 2) {
                ((int16_t *)data)[i] = value.value_i(i);
                ((int16_t *)data)[i + 1] = value.value_i(i + 1);
            }
            break;
        case testpt::DTYPE_COMPLEX_FLOAT:
            for (int i = 0; i < 2 * count; i += 2) {
                int value_i = value.value_i(i);
                int value_i_imag = value.value_i(i + 1);
                ((float *)data)[i] = *((float *)(&value_i));
                ((float *)data)[i + 1] = *((float *)(&value_i_imag));
            }
            break;
        default:
            GTEST_CHECK_CUSTOM(false, ParsePtFault,
                               "Parser: found unsuppored dtype in value_i, value_i only "
                               "support "
                               "int8/uint8/int16/int32/int64/bool.");
    }
}

// get value from value_l
// no quant intx, saved in value_l
void Parser::getTensorValueL(const testpt::Tensor *pt, void *data, size_t count,
                             testpt::Value &value) {
    GTEST_CHECK_CUSTOM(value.value_l_size() == count, ParsePtFault,
                       "Parser: when read value_l, expected element num is not equal "
                       "to real element num.");
    switch (pt->dtype()) {
        case testpt::DTYPE_INT64:
            for (int i = 0; i < count; ++i) {
                ((int64_t *)data)[i] = value.value_l(i);
            }
            break;
        default:
            GTEST_CHECK_CUSTOM(false, ParsePtFault,
                               "Parser: found unsuppored dtype in value_l, value_l only "
                               "support int64.");
    }
}

inline float str2fp32(const std::string &in_str) {
    uint32_t res = 0x0000;
    for (int i = 0; i < in_str.size(); ++i) {
        char byte = in_str.c_str()[i];  // 0~f
        res = res << 4;
        res |= 0xf & (byte >= 'a') ? byte - 'a' + 10 : byte - '0';  // 0~15
    }
    return *(float *)(&res);
}

inline uint16_t str2fp16(const std::string &in_str) {
    uint16_t res = 0x00;
    for (int i = 0; i < in_str.size(); ++i) {
        char byte = in_str.c_str()[i];
        res = res << 4;
        res |= 0xf & (byte >= 'a') ? byte - 'a' + 10 : byte - '0';
    }
    return res;
}

// get value by value_h (hex)
// we hope all float value come from value_h to keep precision
void Parser::getTensorValueH(testpt::Tensor *pt, void *data, size_t count, testpt::Value &value) {
    GTEST_CHECK_CUSTOM(value.value_h_size() == count, ParsePtFault,
                       "Parser: when read value_h, expected element num is not equal "
                       "to real element num.");
    switch (pt->dtype()) {
        case testpt::DTYPE_HALF:
            for (int i = 0; i < count; ++i) {
                ((uint16_t *)data)[i] = str2fp16(value.value_h(i));
            }
            break;
        case testpt::DTYPE_FLOAT:
            for (int i = 0; i < count; ++i) {
                ((float *)data)[i] = str2fp32(value.value_h(i));
            }
            break;
        default:
            GTEST_CHECK_CUSTOM(false, ParsePtFault,
                               "Parser: found unsuppored dtype in value_h, value_h only "
                               "supporte float/half.");
    }
}

void Parser::getTensorValueBF(testpt::Tensor *pt, void *data, size_t count, testpt::Value &value) {
    GTEST_CHECK_CUSTOM(value.value_bf_size() == count, ParsePtFault,
                       "Parser: when read value_h, expected element num is not equal "
                       "to real element num.");
    switch (pt->dtype()) {
        case testpt::DTYPE_HALF:
            for (int i = 0; i < count; ++i) {
                ((uint16_t *)data)[i] = str2fp16(value.value_h(i));
            }
            break;
        case testpt::DTYPE_FLOAT:
            for (int i = 0; i < count; ++i) {
                ((float *)data)[i] = str2fp32(value.value_h(i));
            }
            break;
        case testpt::DTYPE_BFLOAT16:
            for (int i = 0; i < count; ++i) {
                ((uint16_t *)data)[i] = str2fp16(value.value_h(i));
            }
            break;

        default:
            GTEST_CHECK_CUSTOM(false, ParsePtFault,
                               "Parser: found unsuppored dtype in value_h, value_h only "
                               "supporte float/half/bfloat16.");
    }
}

// get value by random data param
void Parser::getTensorValueRandom(testpt::Tensor *pt, void *data, size_t count) {
    switch (pt->dtype()) {
        case testpt::DTYPE_HALF:
            generateRandomData((half_float::half *)data, count, pt->mutable_random_data());
            break;
        case testpt::DTYPE_FLOAT:
            generateRandomData((float *)data, count, pt->mutable_random_data());
            break;
        case testpt::DTYPE_BOOL:
            generateRandomData((bool *)data, count, pt->mutable_random_data());
            break;
        case testpt::DTYPE_INT8:
            generateRandomData((int8_t *)data, count, pt->mutable_random_data());
            break;
        case testpt::DTYPE_INT16:
            generateRandomData((int16_t *)data, count, pt->mutable_random_data());
            break;
        case testpt::DTYPE_INT32:
            generateRandomData((int32_t *)data, count, pt->mutable_random_data());
            break;
        case testpt::DTYPE_INT64:
            generateRandomData((int64_t *)data, count, pt->mutable_random_data());
            break;
        case testpt::DTYPE_DOUBLE:
            generateRandomData((double *)data, count, pt->mutable_random_data());
            break;
        case testpt::DTYPE_UINT8:
            generateRandomData((uint8_t *)data, count, pt->mutable_random_data());
            break;
        case testpt::DTYPE_UINT16:
            generateRandomData((uint16_t *)data, count, pt->mutable_random_data());
            break;
        case testpt::DTYPE_UINT32:
            generateRandomData((uint32_t *)data, count, pt->mutable_random_data());
            break;
        case testpt::DTYPE_UINT64:
            generateRandomData((uint64_t *)data, count, pt->mutable_random_data());
            break;
        case testpt::DTYPE_BFLOAT16:
            generateRandomDataBf16((uint16_t *)data, count, pt->mutable_random_data());
            break;
        case testpt::DTYPE_COMPLEX_HALF:
        case testpt::DTYPE_COMPLEX_FLOAT:
            ALLOG(ERROR)
                << "Generate random data failed. DTYPE_COMPLEX_HALF and DTYPE_COMPLEX_FLOAT not "
                   "supported now";
            throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
        default:
            ALLOG(ERROR) << "Generate random data failed. ";
            throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }
}

// get value by path
void Parser::getTensorValueByFile(testpt::Tensor *pt, void *data, size_t count, std::string path) {
    // readDataFromFile(pt->path(), data, count);
    auto cur_pb_path = pb_path_ + path;
    std::ifstream fin(cur_pb_path, std::ios::in | std::ios::binary);
    if (!fin) {
        ALLOG(ERROR) << "read data in file failed.";
        throw FileOpenFault(std::string(__FILE__) + "+" + std::to_string(__LINE__));
    }
    size_t tensor_length = count * getTensorSize(pt);
    fin.read((char *)data, tensor_length);
}

// set value in proto to meta_tensor.ptr
// random data(for cpu compute) value is fp32 definitely
// valueh valuef valuei dtype is according dtype in proto
void Parser::getTensorValue(testpt::Tensor *pt, void *data, ValueType value_type, size_t count,
                            testpt::Value &value, std::string path) {
    switch (value_type) {
        case VALUE_H: getTensorValueH(pt, data, count, value); break;
        case VALUE_F: getTensorValueF(pt, data, count, value); break;
        case VALUE_I: getTensorValueI(pt, data, count, value); break;
        case VALUE_L: getTensorValueL(pt, data, count, value); break;
        case VALUE_BF: getTensorValueBF(pt, data, count, value); break;
        case VALUE_RANDOM: getTensorValueRandom(pt, data, count); break;
        case VALUE_PATH: getTensorValueByFile(pt, data, count, path); break;
        case VALUE_INVALID:
            GTEST_WARNING(false, "Parser: trying to get value of tensor, but missing data "
                                 "source.");
            break;
        default:
            GTEST_CHECK_CUSTOM(false, ParsePtFault,
                               "Parser: get tensor data failed, unsupported value type.");
    }
}

std::vector<int> Parser::threshold_use() {
    std::vector<int> res;
    for (int i = 0; i < outputs_.size(); ++i) {
        if (proto_node_->mutable_output(i)->has_threshold_use()) {
            int threshold_use = (int)proto_node_->mutable_output(i)->threshold_use();
            res.push_back(threshold_use);
        } else {
            res.push_back(1);
        }
    }
    return res;
}

bool Parser::check_threshold() {
    std::vector<int> threshold_use = Parser::threshold_use();
    bool res = false;

    if (proto_node_->has_test_param()) {
        if (0 != proto_node_->mutable_test_param()->error_threshold_size()) {
            res = true;
        }
    } else if (0 != proto_node_->evaluation_threshold_size()) {
        res = true;
    } else {
        for (int i = 0; i < outputs_.size(); ++i) {
            testpt::Tensor *pt = proto_node_->mutable_output(i);  // pt for proto_tensor
            if (0 == threshold_use[i]) {
                // pass
            } else if (0 != pt->thresholds().evaluation_threshold_size()) {
                res = true;
            } else {
                // pass
            }
        }
    }
    return res;
}

bool Parser::common_threshold() {
    bool res = false;
    if (proto_node_->has_test_param()) {
        if (0 != proto_node_->mutable_test_param()->error_threshold_size()) {
            res = true;
        }
    } else if (0 != proto_node_->evaluation_threshold_size()) {
        res = true;
    } else {
        // pass
    }
    return res;
}

bool Parser::readMessageFromFile(const std::string &filename, testpt::Node *proto) {
    std::ifstream fin(filename, std::ios::in);
    if (!fin.is_open()) {
        ALLOG(ERROR) << "File not found: " << filename;
        fin.close();
        return false;
    }

    bool status = false;
    google::protobuf::io::IstreamInputStream input(&fin);
    if (filename.find(".prototxt") != std::string::npos) {
        status = google::protobuf::TextFormat::Parse(&input, proto);
    } else if (filename.find(".pb") != std::string::npos) {
        google::protobuf::io::CodedInputStream coded_input(&input);
        coded_input.SetTotalBytesLimit(INT_MAX);
        status = proto->ParseFromCodedStream(&coded_input);
    }
    fin.close();
    return status;
}

optest::Formula Parser::cvtProtoEvaluationCriterion(testpt::EvaluationCriterion f) {
    switch (f) {
        case testpt::EvaluationCriterion::DIFF1: return optest::Formula::DIFF1;
        case testpt::EvaluationCriterion::DIFF2: return optest::Formula::DIFF2;
        case testpt::EvaluationCriterion::DIFF3: return optest::Formula::DIFF3;
        case testpt::EvaluationCriterion::DIFF3_MAX: return optest::Formula::DIFF3_MAX;
        case testpt::EvaluationCriterion::DIFF3_MEAN: return optest::Formula::DIFF3_MEAN;
        case testpt::EvaluationCriterion::DIFF4: return optest::Formula::DIFF4;
        case testpt::EvaluationCriterion::MAPE: return optest::Formula::MAPE;
        case testpt::EvaluationCriterion::MAE: return optest::Formula::MAE;
        default:
            ALLOG(ERROR) << "NOT support this evaluation critertion yet";
            throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }
}

DataType Parser::getDataType(const testpt::DataType dtype) { return (DataType)dtype; }

TensorLayout Parser::getLayout(const testpt::TensorLayout layout) { return (TensorLayout)layout; }

TensorType Parser::getTensorType(const testpt::TensorType ttype) { return (TensorType)ttype; }

/*
size_t Parser::getSizeOfDataType(DataType dtype) {
  switch (dtype) {
    case DTYPE_BOOL:
    case DTYPE_INT8:
    case DTYPE_UINT8: {
      return 1;
    }
    case DTYPE_UINT16:
    case DTYPE_INT16:
    case DTYPE_HALF: {
      return 2;
    }
    case DTYPE_UINT32:
    case DTYPE_INT32:
    case DTYPE_FLOAT: {
      return 4;
    }
    case DTYPE_UINT64:
    case DTYPE_INT64:
    case DTYPE_DOUBLE: {
      return 8;
    }
    default: {
      return 0;
    }
  }
}
*/

ValueType Parser::getValueType(const testpt::Tensor *t) {
    // value_h > value_f > value_i > random data > path
    if (t->has_prev_value()) {
        if (t->prev_value().value_h_size() != 0) {
            return VALUE_H;
        } else if (t->prev_value().value_f_size() != 0) {
            return VALUE_F;
        } else if (t->prev_value().value_i_size() != 0) {
            return VALUE_I;
        } else if (t->prev_value().value_l_size() != 0) {
            return VALUE_L;
        }
        return VALUE_H;
    } else if (t->has_prev_path()) {
        return VALUE_PATH;
    } else if (t->has_random_data()) {
        return VALUE_RANDOM;
    } else {
        return VALUE_INVALID;
    }
}

// include stride
// compute this by shape(stride + dims)
inline size_t Parser::getTensorShapeCount(testpt::Tensor *pt) {
    GTEST_CHECK_CUSTOM(pt->has_shape(), ParsePtFault, "Parser: missing tensor shape in prototxt.");
    return shapeElementCount(pt->mutable_shape());
}

// if have value_x return value_size
// else (random/path) return shape_count
inline size_t Parser::getTensorStrideCount(testpt::Tensor *pt, ValueType value_type) {
    if (pt->mutable_shape()->dim_stride_size() > 0) {
        GTEST_CHECK_CUSTOM(pt->has_shape(), ParsePtFault,
                           "Parser: missing tensor shape in prototxt.");
        return shapeStrideCount(pt->mutable_shape());
    }
    switch (value_type) {
        case VALUE_H: return pt->prev_value().value_h_size();
        case VALUE_I: return pt->prev_value().value_i_size();
        case VALUE_L: return pt->prev_value().value_l_size();
        case VALUE_F: return pt->prev_value().value_f_size();
        case VALUE_RANDOM:
        case VALUE_PATH:
        case VALUE_INVALID:
            GTEST_CHECK_CUSTOM(pt->has_shape(), ParsePtFault,
                               "Parser: missing tensor shape in prototxt.");
            return shapeStrideCount(pt->mutable_shape());
        default:
            GTEST_CHECK_CUSTOM(false, ParsePtFault,
                               "Parser: got unsupported value type, parse tensor failed.");
    }
}

// so we can get tensor by "parser_->get("input0").tensor"
MetaTensor &Parser::getMetaTensor(const std::string &name) {
    auto it = find_if(inputs_.begin(), inputs_.end(),
                      [=](const MetaTensor &t) { return t.name == name; });
    if (it != inputs_.end()) {
        return *it;
    } else {
        auto it = find_if(outputs_.begin(), outputs_.end(),
                          [=](const MetaTensor &t) { return t.name == name; });
        if (it != outputs_.end()) {
            return *it;
        }
    }
    ALLOG(ERROR) << "Miss tensor: " << name << " in prototxt.";
    throw ParsePtFault(std::string(__FILE__) + " +" + std::to_string(__LINE__));
}

// can get tensor by "parser_->get(2).tensor"
// but this index is the tensor index in *prototxt(input/output together)
MetaTensor &Parser::getMetaTensor(int index) {
    if (index < inputs_.size()) {
        return inputs_.at(index);
    } else {
        return outputs_.at(index - inputs_.size());
    }
}

void Parser::setCurPbPath(const std::string &filename) {
    if (filename.find("/") == std::string::npos) {
        pb_path_ = filename;
    } else {
        auto pos_pb = filename.find_last_of("/");
        pb_path_ = filename.substr(0, pos_pb + 1);
    }
}

size_t Parser::getTensorSize(testpt::Tensor *pt) {
#define GET_WIDTH_TENSOR_TYPE(TENSOR_DTYPE, WIDTH) \
    case TENSOR_DTYPE: return WIDTH;
    switch (pt->dtype()) {
        GET_WIDTH_TENSOR_TYPE(testpt::DTYPE_DOUBLE, 8);
        GET_WIDTH_TENSOR_TYPE(testpt::DTYPE_INT64, 8);
        GET_WIDTH_TENSOR_TYPE(testpt::DTYPE_UINT64, 8);
        GET_WIDTH_TENSOR_TYPE(testpt::DTYPE_FLOAT, 4);
        GET_WIDTH_TENSOR_TYPE(testpt::DTYPE_INT32, 4);
        GET_WIDTH_TENSOR_TYPE(testpt::DTYPE_UINT32, 4);
        GET_WIDTH_TENSOR_TYPE(testpt::DTYPE_INT16, 2);
        GET_WIDTH_TENSOR_TYPE(testpt::DTYPE_UINT16, 2);
        GET_WIDTH_TENSOR_TYPE(testpt::DTYPE_INT8, 1);
        GET_WIDTH_TENSOR_TYPE(testpt::DTYPE_UINT8, 1);
        GET_WIDTH_TENSOR_TYPE(testpt::DTYPE_HALF, 2);
        GET_WIDTH_TENSOR_TYPE(testpt::DTYPE_BOOL, 1);
        GET_WIDTH_TENSOR_TYPE(testpt::DTYPE_BFLOAT16, 2);
        default: GTEST_CHECK_CUSTOM(false, ParsePtFault, "Parser: Unknown tensor DTYPE.");
    }
#undef GET_WIDTH_TENSOR_TYPE
}

}  // namespace optest
