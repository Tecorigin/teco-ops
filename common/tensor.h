#ifndef COMMON_TENSOR_H_
#define COMMON_TENSOR_H_

#include <vector>
#include <iostream>
#include <vector>
#include <memory>

#include <memory>
#include <vector>
#include <cstdint>

class TensorBase {
public:
    TensorBase(void* data, const std::vector<int>& shape) 
        : data_(data), shape_(shape) {
        int nbdims = shape_.size();
        stride_.assign(nbdims, 0);
        if (nbdims > 0) {
            stride_[nbdims-1] = 1;
            for (int i = nbdims-2; i >= 0; i--) {
                stride_[i] = stride_[i+1] * shape_[i+1];
            }
        }
    }

    TensorBase(void* data, const std::vector<int>& shape, const std::vector<int>& stride) 
        : data_(data), shape_(shape), stride_(stride) {
    }

    TensorBase(const TensorBase& other): data_(other.data_), shape_(other.shape_), stride_(other.stride_){
    }

    TensorBase& operator=(const TensorBase& other) {
        if (this != &other) {
        data_ = other.data_;
        shape_ = other.shape_;
        stride_ = other.stride_;
        }
        return *this;
    }
    
    virtual ~TensorBase() = default;

    const std::vector<int>& shape() const { return shape_; }
    int64_t numel() const { 
        int64_t numel_ = 1;
        for (int dim : shape_) {
            numel_ *= dim;
        }
        return numel_;
    }
    const std::vector<int>& stride() const { return stride_; }
    void* data() { return data_; }

protected:
    std::vector<int> shape_;
    std::vector<int> stride_;
    void* data_;
};

template <typename T>
class Tensor : public TensorBase {
public:
    Tensor(T* data, const std::vector<int>& shape) 
        : TensorBase(data, shape) {}

    Tensor(T* data, const std::vector<int>& shape, const std::vector<int>& stride) 
        : TensorBase(data, shape, stride) {}
    
    Tensor(const Tensor& other) : TensorBase(other) {}  // 调用基类拷贝构造

    Tensor& operator=(const Tensor& other) {
        TensorBase::operator=(other);  // 调用基类拷贝赋值
        return *this;
    }

    T* data() { return static_cast<T*>(data_); }
    
    // 提供const版本
    const T* data() const { return static_cast<const T*>(data_); }
};

#endif  // COMMON_TENSOR_H_
