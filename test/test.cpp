#include <vector>
#include <iostream>
#include <vector>
#include <memory>

#include <memory>
#include <vector>
#include <cstdint>

// class TensorBase {
// public:
//     TensorBase(void* data, const std::vector<int>& shape) 
//         : data_(data), shape_(shape) {
//         for (int dim : shape_) {
//             if (dim <= 0) throw std::invalid_argument("Invalid shape dimension");
//         }
//         calculateNumelAndStride();
//     }
// 
//     TensorBase(void* data, const std::vector<int>& shape, const std::vector<int>& stride) 
//         : data_(data), shape_(shape), stride_(stride) {
//         numel_ = 1;
//         for (int dim : shape) {
//             if (dim <= 0) throw std::invalid_argument("Invalid shape dimension");
//             numel_ *= dim;
//         }
//     }
// 
//     TensorBase(const TensorBase& other) 
//         : data_(other.data_), 
//           shape_(other.shape_), 
//           stride_(other.stride_),
//           numel_(other.numel_) {}  // 显式拷贝所有成员
// 
//     TensorBase& operator=(const TensorBase& other) {
//         if (this != &other) {
//             data_ = other.data_;
//             shape_ = other.shape_;
//             stride_ = other.stride_;
//             numel_ = other.numel_;
//         }
//         return *this;
//     }
//     
//     virtual ~TensorBase() = default;
// 
//     const std::vector<int>& shape() const { return shape_; }
//     int64_t numel() const { return numel_; }
//     const std::vector<int>& stride() const { return stride_; }
//     void* data() { return data_; }
// 
// protected:
//     void calculateNumelAndStride() {
//         numel_ = 1;
//         for (int dim : shape_) {
//             numel_ *= dim;
//         }
// 
//         int nbdims = shape_.size();
//         stride_.assign(nbdims, 0);
//         if (nbdims > 0) {
//             stride_[nbdims-1] = 1;
//             for (int i = nbdims-2; i >= 0; i--) {
//                 stride_[i] = stride_[i+1] * shape_[i+1];
//             }
//         }
//     }
// 
//     std::vector<int> shape_;
//     std::vector<int> stride_;
//     int64_t numel_ = 0;  // 初始化默认值
//     void* data_ = nullptr;
// };
// 
// template <typename T>
// class Tensor : public TensorBase {
// public:
//     Tensor(T* data, const std::vector<int>& shape) 
//         : TensorBase(data, shape) {}
// 
//     Tensor(T* data, const std::vector<int>& shape, const std::vector<int>& stride) 
//         : TensorBase(data, shape, stride) {}
// 
//     // 显式实现拷贝构造函数和赋值运算符
//     Tensor(const Tensor& other) : TensorBase(other) {}  // 调用基类拷贝构造
// 
//     Tensor& operator=(const Tensor& other) {
//         TensorBase::operator=(other);  // 调用基类拷贝赋值
//         return *this;
//     }
// 
//     T* data() { return static_cast<T*>(data_); }
//     const T* data() const { return static_cast<const T*>(data_); }
// };

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


int main() {
    float* a = (float*)malloc(1000 * sizeof(float));  // 确保分配足够内存
    std::shared_ptr<Tensor<float>> Input = std::make_shared<Tensor<float>>(a, std::vector<int>({1, 2, 3, 4}));
    
    // std::vector<TensorBase*> inputs_tensor;
    inputs_tensor.push_back(Input.get());
    
    auto Input2 = *(static_cast<Tensor<float>*>(inputs_tensor[0]));  // 安全拷贝

    std::vector<TensorBase*> inputs_tensor;


}