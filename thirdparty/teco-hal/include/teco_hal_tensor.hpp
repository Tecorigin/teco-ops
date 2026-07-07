#ifndef TECO_HAL_INCLUDE_TECO_HAL_TENSOR_HPP
#define TECO_HAL_INCLUDE_TECO_HAL_TENSOR_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "teco_hal_config.h"
namespace teco {
namespace hal {

/**
 * @brief 零开销的多维张量视图
 *
 * @tparam T       元素数据类型（如 float, double, int）
 * @tparam Dims    编译期固定的维度数（Dims > 0）
 *
 * 设计特点：
 * - 仅支持行优先排布
 * - 不拥有数据内存，仅持有一个外部传入的裸指针。
 * - 所有元数据（shape, strides）存储在栈上的 std::array 中，完全零堆分配。
 * -
 * 提供多维下标访问、切片、转置、广播等操作——这些操作仅调整视图元数据，完全零数据拷贝。
 * - 编译期维度数 Dims 使得循环可被完全展开，配合折叠表达式实现高效的地址计算。
 * - 布局策略通过标签类型在编译期确定，不会引入运行时开销。
 */
template <typename T, std::size_t Dims>
class Tensor {
    // 维度数必须大于0
    static_assert(Dims > 0, "Tensor dimension must be > 0");

 public:
    // 标准类型别名，便于泛型编程
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;
    using ptr_type = T *;
    using const_ptr_type = const T *;
    using IndexType = std::size_t;
    using shape_type = std::array<IndexType, Dims>;
    using strides_type = std::array<IndexType, Dims>;

    // ---------- 构造函数 ----------

    /// 默认构造，空视图（data_ 为 nullptr）
    HAL_DEVICE Tensor() = default;

    /**
     * @brief 从外部指针和形状构造
     * @param data  指向连续内存块首元素的指针（生命周期由调用者保证）
     * @param shape 各维度大小
     */
    HAL_DEVICE explicit Tensor(ptr_type data, const shape_type &shape) :
        data_(data), shape_(shape) {
        compute_default_strides();  // 编译期分派计算对应排布的步长
    }

    /**
     * @brief 从外部指针、形状和自定义步长构造（用于非连续视图）
     * @param data    数据首元素指针（可能已偏移）
     * @param shape   各维度大小
     * @param stride 各维度的步长（元素个数，非字节）
     */
    HAL_DEVICE Tensor(ptr_type data, const shape_type &shape, const strides_type &stride) :
        data_(data), shape_(shape), strides_(stride) {}

    // ---------- 访问器 ----------

    /// 获取数据裸指针（注意：仅在内存连续时指向完整数据块）
    HAL_DEVICE FORCE_INLINE ptr_type data() { return data_; }
    HAL_DEVICE FORCE_INLINE const_ptr_type data() const { return data_; }

    /// 形状
    HAL_DEVICE FORCE_INLINE const shape_type &shape() const { return shape_; }
    /// 步长
    HAL_DEVICE FORCE_INLINE const strides_type &stride() const { return strides_; }

    /// 逻辑元素总数（各维度大小的乘积）
    HAL_DEVICE FORCE_INLINE IndexType num_elements() const {
        IndexType res = 1;
        for (std::size_t i = 0; i < Dims; ++i) res *= shape_[i];
        return res;
    }

    HAL_DEVICE FORCE_INLINE static constexpr IndexType get_dims() { return Dims; }

    // ---------- 元素访问 ----------

    /**
     * @brief 多维下标访问（可变参数版本）
     * @param indices 各维度下标，数量必须等于 Dims
     * @return 元素的引用
     *
     * 使用折叠表达式计算线性偏移：offset = SUM(indices[i] * stride[i])
     * 编译器会完全展开循环，生成与手写乘加完全一致的机器码。
     */
    template <typename... Is>
    HAL_DEVICE reference operator()(Is... indices) {
        static_assert(sizeof...(Is) == Dims, "Number of indices must equal tensor dimension");
        return data_[linear_index(static_cast<IndexType>(indices)...)];
    }

    /// const 版本
    template <typename... Is>
    HAL_DEVICE const_reference operator()(Is... indices) const {
        static_assert(sizeof...(Is) == Dims, "Number of indices must equal tensor dimension");
        return data_[linear_index(static_cast<IndexType>(indices)...)];
    }

    // ---------- 内存连续性判断 ----------

    /**
     * @brief 判断张量在内存中是否紧密连续排列
     * 连续条件：对于行优先布局，从最后一维开始 stride 应呈逆序乘积。
     * 广播维度（shape=1）的步长可任意，通常为0，不影响连续性判断。
     */
    HAL_DEVICE FORCE_INLINE bool is_contiguous() const {
        IndexType expected = 1;
        for (int i = static_cast<int>(Dims) - 1; i >= 0; --i) {
            if (shape_[i] == 1) continue;  // 跳过可广播维度
            if (strides_[i] != expected) return false;
            expected *= shape_[i];
        }
        return true;
    }

    // ---------- 迭代器支持（仅在连续时安全） ----------

    HAL_DEVICE FORCE_INLINE ptr_type begin() {
        assert(is_contiguous() && "Only contiguous tensors can be iterated as flat range");
        return data_;
    }
    HAL_DEVICE FORCE_INLINE const_ptr_type begin() const {
        assert(is_contiguous());
        return data_;
    }
    HAL_DEVICE FORCE_INLINE ptr_type end() {
        assert(is_contiguous());
        return data_ + num_elements();
    }
    HAL_DEVICE FORCE_INLINE const_ptr_type end() const {
        assert(is_contiguous());
        return data_ + num_elements();
    }

    // ---------- 零拷贝视图操作（返回新的轻量 Tensor） ----------

    /**
     * @brief 切片：沿指定维度取 [start, end) 子视图
     * @param dim   维度索引
     * @param start 起始位置（包含）
     * @param end   结束位置（不包含）
     * @return 新的视图，数据指针向前移动，形状更新，步长不变
     *
     */
    HAL_DEVICE FORCE_INLINE Tensor slice(IndexType dim, IndexType start, IndexType end) const {
        assert(dim < Dims);
        assert(start < end && end <= shape_[dim]);
        shape_type new_shape = shape_;
        new_shape[dim] = end - start;
        // 数据指针移动到切片起始位置
        return Tensor(data_ + start * strides_[dim], new_shape, strides_);
    }

 private:
    // ----- 成员变量（仅 3 个，完全栈分配）-----
    ptr_type data_ = nullptr;  ///< 当前视图数据首指针（可能非原始分配地址）
    shape_type shape_{};       ///< 各维度大小
    strides_type strides_{};   ///< 各维度步长（单位：元素个数）

    /**
     * @brief 计算默认紧凑步长
     */
    HAL_DEVICE FORCE_INLINE void compute_default_strides() {
        // 行优先：最后一维步长为 1，向前依次乘以右侧维度大小
        strides_[Dims - 1] = 1;
        for (int i = static_cast<int>(Dims) - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    /**
     * @brief 将多维下标转换为线性偏移
     * @param indices 各维度下标（数量必须等于 N）
     * @return 线性偏移量（相对于 data_）
     *
     * 使用折叠表达式 (add(indices), ...) 展开循环，
     * 等效于: idx = SUM(indices[i] * strides_[i])
     */
    template <typename... Is>
    HAL_DEVICE FORCE_INLINE IndexType linear_index(Is... indices) const {
        IndexType idx = 0;
        IndexType i = 0;
        // 立即调用的 lambda，用于在折叠表达式中累积
        auto add = [&](IndexType ind) { idx += ind * strides_[i++]; };
        (add(static_cast<IndexType>(indices)), ...);
        return idx;
    }
};

// ============================================================================
// 一维特化：提供 operator[] 以便像普通数组一样使用
// ============================================================================
template <typename T>
class Tensor<T, 1> {
 public:
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;
    using ptr_type = T *;
    using const_ptr_type = const T *;
    using IndexType = std::size_t;
    using shape_type = std::array<IndexType, 1>;
    using strides_type = std::array<IndexType, 1>;

    HAL_DEVICE Tensor() = default;

    HAL_DEVICE explicit Tensor(ptr_type data, const shape_type &shape) :
        data_(data), shape_(shape) {
        compute_default_strides();  // strides_[0] = 1
    }

    HAL_DEVICE Tensor(ptr_type data, const shape_type &shape, const strides_type &stride) :
        data_(data), shape_(shape), strides_(stride) {}

    HAL_DEVICE FORCE_INLINE ptr_type data() { return data_; }
    HAL_DEVICE FORCE_INLINE const_ptr_type data() const { return data_; }
    HAL_DEVICE FORCE_INLINE const shape_type &shape() const { return shape_; }
    HAL_DEVICE FORCE_INLINE const strides_type &stride() const { return strides_; }
    HAL_DEVICE FORCE_INLINE IndexType num_elements() const { return shape_[0]; }

    // 一维下标访问（两种形式）
    HAL_DEVICE reference operator[](IndexType idx) {
        assert(idx < shape_[0]);
        return data_[idx * strides_[0]];
    }
    HAL_DEVICE const_reference operator[](IndexType idx) const {
        assert(idx < shape_[0]);
        return data_[idx * strides_[0]];
    }

    HAL_DEVICE reference operator()(IndexType idx) { return (*this)[idx]; }
    HAL_DEVICE const_reference operator()(IndexType idx) const { return (*this)[idx]; }

    HAL_DEVICE FORCE_INLINE bool is_contiguous() const {
        // 一维张量连续的条件：大小为1 或 步长为1
        return shape_[0] == 1 || strides_[0] == 1;
    }

    HAL_DEVICE FORCE_INLINE ptr_type begin() {
        assert(is_contiguous());
        return data_;
    }
    HAL_DEVICE FORCE_INLINE const_ptr_type begin() const {
        assert(is_contiguous());
        return data_;
    }
    HAL_DEVICE FORCE_INLINE ptr_type end() {
        assert(is_contiguous());
        return data_ + num_elements();
    }
    HAL_DEVICE FORCE_INLINE const_ptr_type end() const {
        assert(is_contiguous());
        return data_ + num_elements();
    }

    HAL_DEVICE FORCE_INLINE Tensor slice(IndexType dim, IndexType start, IndexType end) const {
        assert(dim == 0);
        assert(start < end && end <= shape_[0]);
        shape_type new_shape{end - start};
        return Tensor(data_ + start * strides_[0], new_shape, strides_);
    }

 private:
    ptr_type data_ = nullptr;
    shape_type shape_{0};
    strides_type strides_{0};

    HAL_DEVICE FORCE_INLINE void compute_default_strides() {
        strides_[0] = 1;  // 一维总是连续步长为1
    }
};

// ============================================================================
// 便捷工厂函数：自动推导模板参数，简化对象创建
// ============================================================================
template <std::size_t Dims, typename T>
HAL_DEVICE Tensor<T, Dims> make_tensor(T *data, const std::array<std::size_t, Dims> &shape) {
    return Tensor<T, Dims>(data, shape);
}

template <std::size_t Dims, typename T>
HAL_DEVICE Tensor<T, Dims> make_tensor(T *data, const std::array<std::size_t, Dims> &shape,
                                       const std::array<std::size_t, Dims> &stride) {
    return Tensor<T, Dims>(data, shape, stride);
}
}  // namespace hal
}  // namespace teco
#endif  // TECO_HAL_INCLUDE_TECO_HAL_TENSOR_HPP
