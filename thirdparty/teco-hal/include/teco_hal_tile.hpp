#ifndef TECO_HAL_INCLUDE_TECO_HAL_TILE_HPP
#define TECO_HAL_INCLUDE_TECO_HAL_TILE_HPP
#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "teco_hal_config.h"
#include "teco_hal_tensor.hpp"
namespace teco {
namespace hal {

/**
 * @brief 核心逻辑布局与 Tile 切分/广播模式枚举
 *
 * 该枚举描述一个 tile 数据在 32 个核心上的分配方式，每个枚举值由 1～3 个部分组成，
 * 分别表示核心的逻辑排布、tile 维度与核心排布维度的对应切分关系、以及可选的广播方式。
 *
 * -- 第一部分：核心逻辑排布 --
 * 表示将 32 个核心从逻辑上看成一个二维网格，格式为 R<行数>C<列数>。
 * 例如 R1C32 表示 1 行 32 列，R2C16 表示 2 行 16 列，R4C8 表示 4 行 8 列。
 *
 * -- 第二部分：Tile 与网格维度的对应关系 --
 * 用 RC 或 CR 表示 tile 的行/列如何映射到核心网格的 R/C 上进行切分。
 *   - RC: Tile 的 行(Row) 在网格的 R 方向上切分，
 *         Tile 的 列(Col) 在网格的 C 方向上切分。
 *   - CR: Tile 的 列(Col) 在网格的 R 方向上切分，
 *         Tile 的 行(Row) 在网格的 C 方向上切分。
 *
 * 当网格的某个维度为 1 时（如 R1C32 中的 R=1），另一维度上的切分即为唯一的分布方向。
 * 例如 R1C32_RC: 行在 R=1 上切分（即不切分），列在 C=32 上切分，实际效果是 tile
 * 被切成 1x32 的行条带，32 个核心各得一个条带；R1C32_CR 则将 tile 的列在 R=1 上
 * 切分（不切分），行在 C=32 上切分，等效为切成 32x1 的列条带。
 *
 * -- 第三部分：行/列广播（可选） --
 * 若在切分基础上需要在某一个维度上广播数据，则以 RBCAST 或 CBCAST 作为后缀。
 *   - RBCAST: 在网格的 行(R) 方向广播，即同一行内的所有核心获得相同的数据。
 *   - CBCAST: 在网格的 列(C) 方向广播，即同一列内的所有核心获得相同的数据。
 *
 * 无第三部分时，表示不进行广播，每个核心获得独立切分后的唯一数据块。
 *
 * 特殊值 ALL_BCAST 表示全广播，所有核心处理完全相同的整个 tile 数据，
 * 无需关心核心排布和切分方式。
 */
enum class SplitMode {
    ALL_BCAST,  // 全广播，所有核心拥有相同 tile 数据

    R1C32_RC,  // 1x32 网格，tile 行在 R(1) 上切分(不切)，tile 列在 C(32)上 切分 -> 行条带
    R1C32_CR,  // 1x32 网格，tile 列在 R(1) 上切分(不切)，tile 行在 C(32) 上切分 -> 列条带

#if 0  // 待后续支持
    R2C16_RC,         // 2x16 网格，tile 行在 R(2)上切分，tile 列在 C(16) 上切分 -> 2x16 分块
    R2C16_RC_RBCAST,  // 2x16 网格，RC 切分 + 行广播（同行核心数据相同）
    R2C16_RC_CBCAST,  // 2x16 网格，RC 切分 + 列广播（同列核心数据相同）
    R2C16_CR,         // 2x16 网格，tile 列在 R(2)上切分，tile 行在 C(16) 上切分 -> 转置分块
    R2C16_CR_RBCAST,  // 2x16 网格，CR 切分 + 行广播
    R2C16_CR_CBCAST,  // 2x16 网格，CR 切分 + 列广播
#endif

    R4C8_RC,         // 4x8 网格，tile 行在 R(4) 上切分，tile 列在 C(8) 上切分 -> 4x8 分块
    R4C8_RC_RBCAST,  // 4x8 网格，RC 切分 + 行广播
    R4C8_RC_CBCAST,  // 4x8 网格，RC 切分 + 列广播
    R4C8_CR,         // 4x8 网格，tile 列在 R(4) 上切分，tile 行在 C(8) 上切分 -> 转置分块
    R4C8_CR_RBCAST,  // 4x8 网格，CR 切分 + 行广播
    R4C8_CR_CBCAST,  // 4x8 网格，CR 切分 + 列广播

    SPLIT_NOT_SUPPORTED,
};

/**
 * @brief 多维 Tile 描述类，负责多核数据划分与坐标映射。
 *
 * 模板参数：
 *   splitMode  - 多核数据分割模式（网格划分、广播等）
 *   Dims       - 维度数
 */
template <SplitMode splitMode, std::size_t Dims>
class Tile {
    static_assert(splitMode < SplitMode::SPLIT_NOT_SUPPORTED, "Unsupported split mode");

 public:
    using IndexType = std::size_t;
    using IndexArray = std::array<IndexType, Dims>;  // 多维索引类型

    /**
     * @brief 构造函数，根据 Tensor 对象和 tile 形状初始化 Tile。
     *
     * 从 Tensor 拷贝全局形状和步长，并检测每个维度是否存在尾块。
     *
     * @param tensor     输入 Tensor，提供全局形状 shape() 和步长 stride()
     * @param tile_shape 每个 tile（逻辑数据块）的各维度大小
     *
     * @note tile_shape 是“逻辑数据块”的大小，不是单个核心的数据块大小。
     *       在网格划分或广播模式下，核心数据块大小由 tile_shape 根据 splitMode
     *       进一步切分。
     */
    template <typename T>
    HAL_DEVICE constexpr Tile(const Tensor<T, Dims> &tensor, const IndexArray &tile_shape) :
        gm_shape_{},
        gm_stride_{},
        gm_tile_shape_(tile_shape),
        lm_row_blocks_(0),
        lm_col_blocks_(0),
        lm_row_blocks_tail_(0),
        lm_col_blocks_tail_(0) {
        // 从 Tensor 拷贝全局形状与步长
        for (std::size_t i = 0; i < Dims; ++i) {
            gm_shape_[i] = static_cast<IndexType>(tensor.shape()[i]);
            gm_stride_[i] = static_cast<IndexType>(tensor.stride()[i]);
            current_tile_coord_[i] = 0;
        }

        // default: R1C32_RC || R1C32_CR || ALL_BCAST
        int core_rid_ = 0;
        int core_cid_ = threadIdx;

        // 根据 splitMode 配置每个核心的 rid 和 cid
        if constexpr (splitMode == SplitMode::R4C8_RC || splitMode == SplitMode::R4C8_RC_RBCAST ||
                      splitMode == SplitMode::R4C8_RC_CBCAST || splitMode == SplitMode::R4C8_CR ||
                      splitMode == SplitMode::R4C8_CR_RBCAST ||
                      splitMode == SplitMode::R4C8_CR_CBCAST) {
            core_rid_ = threadIdx / 8;
            core_cid_ = threadIdx % 8;
        }

        // 根据 splitMode 计算每个核心处理数据的行数和列数
        if constexpr (splitMode == SplitMode::R1C32_RC || splitMode == SplitMode::R4C8_RC) {
            lm_row_blocks_ = gm_tile_shape_[Dims - 2] / core_rows();
            lm_col_blocks_ = gm_tile_shape_[Dims - 1] / core_cols();
            start_offset_[Dims - 2] = core_rid_ * lm_row_blocks_;
            start_offset_[Dims - 1] = core_cid_ * lm_col_blocks_;
        } else if constexpr (splitMode == SplitMode::R4C8_RC_RBCAST) {
            lm_row_blocks_ = gm_tile_shape_[Dims - 2] / core_rows();
            lm_col_blocks_ = gm_tile_shape_[Dims - 1];
            start_offset_[Dims - 2] = core_rid_ * lm_row_blocks_;
            start_offset_[Dims - 1] = 0;
        } else if constexpr (splitMode == SplitMode::R4C8_RC_CBCAST) {
            lm_row_blocks_ = gm_tile_shape_[Dims - 2];
            lm_col_blocks_ = gm_tile_shape_[Dims - 1] / core_cols();
            start_offset_[Dims - 2] = 0;
            start_offset_[Dims - 1] = core_cid_ * lm_col_blocks_;
        } else if constexpr (splitMode == SplitMode::R1C32_CR || splitMode == SplitMode::R4C8_CR) {
            lm_row_blocks_ = gm_tile_shape_[Dims - 2] / core_cols();
            lm_col_blocks_ = gm_tile_shape_[Dims - 1] / core_rows();
            start_offset_[Dims - 2] = core_cid_ * lm_row_blocks_;
            start_offset_[Dims - 1] = core_rid_ * lm_col_blocks_;
        } else if constexpr (splitMode == SplitMode::R4C8_CR_RBCAST) {
            lm_row_blocks_ = gm_tile_shape_[Dims - 2];
            lm_col_blocks_ = gm_tile_shape_[Dims - 1] / core_rows();
            start_offset_[Dims - 2] = 0;
            start_offset_[Dims - 1] = core_rid_ * lm_col_blocks_;
        } else if constexpr (splitMode == SplitMode::R4C8_CR_CBCAST) {
            lm_row_blocks_ = gm_tile_shape_[Dims - 2] / core_cols();
            lm_col_blocks_ = gm_tile_shape_[Dims - 1];
            start_offset_[Dims - 2] = 0;
            start_offset_[Dims - 1] = core_cid_ * lm_row_blocks_;
        } else if constexpr (splitMode == SplitMode::ALL_BCAST) {
            lm_row_blocks_ = gm_tile_shape_[Dims - 2];
            lm_col_blocks_ = gm_tile_shape_[Dims - 1];
            start_offset_[Dims - 2] = 0;
            start_offset_[Dims - 1] = 0;
        }
        current_offset_ = start_offset_;

        const int gm_row_tail = gm_shape_[Dims - 2] % gm_tile_shape_[Dims - 2];
        const int gm_col_tail = gm_shape_[Dims - 1] % gm_tile_shape_[Dims - 1];
        const bool core_has_row_tail = (lm_row_blocks_ + start_offset_[Dims - 2] > gm_row_tail);
        const bool core_has_col_tail = (lm_col_blocks_ + start_offset_[Dims - 1] > gm_col_tail);
        lm_row_blocks_tail_ = (gm_row_tail && core_has_row_tail) ?
                                  gm_row_tail - start_offset_[Dims - 2] :
                                  lm_row_blocks_;
        lm_col_blocks_tail_ = (gm_col_tail && core_has_col_tail) ?
                                  gm_col_tail - start_offset_[Dims - 1] :
                                  lm_col_blocks_;
        lm_row_blocks_tail_ = lm_row_blocks_tail_ > 0 ? lm_row_blocks_tail_ : 0;
        lm_col_blocks_tail_ = lm_col_blocks_tail_ > 0 ? lm_col_blocks_tail_ : 0;

        // clang-format off
        // printf("Tile: tid:%d, splitMode:%d\n", threadIdx, splitMode);
        // printf(
        //     "Tile: tid:%d, gm_row_tail:%d, core_has_row_tail:%d, gm_row_tail:%d, start_offset:[%d,%d]\n",
        //     threadIdx, gm_row_tail, core_has_row_tail, gm_col_tail, start_offset_[0],
        //     start_offset_[1]);
        // printf(
        //     "Tile: tid:%d, gm_shape:[%d,%d], gm_tile_shape:[%d,%d], lm_row_blocks:%d, lm_col_blocks:%d, lm_row_blocks_tail:%d, lm_col_blocks_tail:%d\n",
        //     threadIdx, gm_shape_[0], gm_shape_[1], gm_tile_shape_[0], gm_tile_shape_[1],
        //     lm_row_blocks_, lm_col_blocks_, lm_row_blocks_tail_, lm_col_blocks_tail_);
        // clang-format on
    }

    // 默认拷贝/移动（编译器生成）
    HAL_DEVICE Tile(const Tile &) = default;             // 拷贝构造，接受左值
    HAL_DEVICE Tile(Tile &&) = default;                  // 移动构造，只接受右值
    HAL_DEVICE Tile &operator=(const Tile &) = default;  // 拷贝赋值，接受左值
    HAL_DEVICE Tile &operator=(Tile &&) = default;       // 移动赋值，只接受右值

    /**
     * @brief 手动更新当前 Tile 坐标
     * @param coord Dims 维坐标组成的数组
     * @return void
     */
    HAL_DEVICE FORCE_INLINE void set_current_tile_coord(const IndexArray &coord) noexcept {
        current_tile_coord_ = coord;

        // 同步计算 current_offset_
        current_offset_ = start_offset_;
        for (std::size_t d = 0; d < Dims; ++d) {
            current_offset_[d] += coord[d] * gm_tile_shape_[d];
        }
    }
    /**
     * @brief 获取当前 Tile 坐标
     * @return IndexArray 各维度坐标组成的数组
     */
    HAL_DEVICE FORCE_INLINE IndexArray get_current_tile_coord() const noexcept {
        return current_tile_coord_;
    }

    /**
     * @brief 获取当前 tile 坐标时刻每个核心所负责数据块的全局偏移。
     *
     * @return IndexArray 各维度坐标偏移组成的数组
     */
    HAL_DEVICE FORCE_INLINE constexpr IndexArray get_core_current_offset() const {
        return current_offset_;
    }

    /**
     * @brief 将任意 Dims 维坐标转换为全局线性索引（行主序）。
     *
     * 根据全局 stride 数组将多维坐标映射为全局内存中的一维偏移量，
     * 偏移以元素个数为单位，从 0 开始计数。
     *
     * @param coord 待转换的 Dims 维坐标数组
     * @return IndexType 对应的全局一维偏移
     */
    HAL_DEVICE FORCE_INLINE constexpr IndexType compute_linear_index(
        const IndexArray &coord) const noexcept {
        IndexType idx = 0;
        for (std::size_t i = 0; i < Dims; ++i) {
            idx += coord[i] * gm_stride_[i];
        }
        return idx;
    }

    /**
     * @brief 获取当前核心所负责数据块的全局线性起始索引。
     *
     * 基于已计算好的当前核心起始偏移 @ref current_offset_ 和全局 stride
     * 计算一维偏移，等价于 compute_linear_index(current_offset_)。
     * 该值表示当前核心应访问的全局内存起始位置（以元素为单位）。
     *
     * @return IndexType 当前核心数据块在全局内存中的起始一维偏移
     */
    HAL_DEVICE FORCE_INLINE constexpr IndexType compute_linear_index() const noexcept {
        IndexType idx = 0;
        for (std::size_t i = 0; i < Dims; ++i) {
            idx += current_offset_[i] * gm_stride_[i];
        }
        return idx;
    }

    /**
     * @brief 根据核心网格坐标 (rid, cid) 计算当前线程所负责数据块的实际尺寸（考虑尾块）。
     *
     * 边界线程的数据块可能小于正常块，此方法自动处理。
     *
     * @return 各维度大小
     */
    HAL_DEVICE FORCE_INLINE constexpr IndexArray compute_core_block_shape() const {
        IndexArray shape{};
        for (std::size_t d = 0; d < Dims - 2; ++d) shape[d] = gm_tile_shape_[d];

        // 通用越界截断判断：如果 偏移 + 块大小 > 全局大小，则截断为剩余大小
        shape[Dims - 2] = (current_offset_[Dims - 2] + lm_row_blocks_ > gm_shape_[Dims - 2]) ?
                              lm_row_blocks_tail_ :
                              lm_row_blocks_;
        shape[Dims - 1] = (current_offset_[Dims - 1] + lm_col_blocks_ > gm_shape_[Dims - 1]) ?
                              lm_col_blocks_tail_ :
                              lm_col_blocks_;
        return shape;
    }

    /**
     * @brief 返回全局形状。
     * @return const IndexArray& 全局形状
     */
    HAL_DEVICE FORCE_INLINE constexpr const IndexArray &global_shape() const noexcept {
        return gm_shape_;
    }

    /**
     * @brief 返回 tile 形状。
     * @return const IndexArray& tile 形状
     */
    HAL_DEVICE FORCE_INLINE constexpr const IndexArray &tile_shape() const noexcept {
        return gm_tile_shape_;
    }

 private:
    // ──── 编译期网格尺寸 ────
    /** @brief 核心网格行数（编译期常量） */
    HAL_DEVICE FORCE_INLINE static constexpr IndexType core_rows() noexcept {
        if constexpr (splitMode == SplitMode::R1C32_RC || splitMode == SplitMode::R1C32_CR ||
                      splitMode == SplitMode::ALL_BCAST) {
            return 1;
        } else if constexpr (splitMode == SplitMode::R4C8_RC ||
                             splitMode == SplitMode::R4C8_RC_RBCAST ||
                             splitMode == SplitMode::R4C8_RC_CBCAST ||
                             splitMode == SplitMode::R4C8_CR_RBCAST ||
                             splitMode == SplitMode::R4C8_CR_CBCAST) {
            return 4;
        } else {
            return 1;
        }
    }
    /** @brief 核心网格列数（编译期常量） */
    HAL_DEVICE FORCE_INLINE static constexpr IndexType core_cols() noexcept {
        if constexpr (splitMode == SplitMode::R1C32_RC || splitMode == SplitMode::R1C32_CR ||
                      splitMode == SplitMode::ALL_BCAST) {
            return 32;
        } else if constexpr (splitMode == SplitMode::R4C8_RC ||
                             splitMode == SplitMode::R4C8_RC_RBCAST ||
                             splitMode == SplitMode::R4C8_RC_CBCAST ||
                             splitMode == SplitMode::R4C8_CR_RBCAST ||
                             splitMode == SplitMode::R4C8_CR_CBCAST) {
            return 8;
        } else {
            return 1;
        }
    }
    /** @brief 核心总数（始终为 32） */
    HAL_DEVICE FORCE_INLINE static constexpr IndexType core_count() noexcept {
        return core_rows() * core_cols();
    }

    // ---------- 私有成员（全固定数组，无动态分配）----------
    // gm : global memory，主存 HBM 相关
    // lm : local memory，每个核心本地内存 SPM 相关
    std::array<IndexType, Dims> gm_shape_;       // 全局形状
    std::array<IndexType, Dims> gm_stride_;      // 全局步长
    std::array<IndexType, Dims> gm_tile_shape_;  // tile 尺寸（逻辑上整个数据块的大小，非单核）
    IndexArray current_tile_coord_;              // 当前 tile 的坐标
    IndexArray start_offset_;                    // 初始 tile 每个核心对应的偏移量
    IndexArray current_offset_;                  // 当前 tile 每个核心对应的偏移量
    int lm_row_blocks_;       // 针对 gm_tile_shape_ 行方向切块大小（仅规则网格模式）
    int lm_col_blocks_;       // 针对 gm_tile_shape_ 列方向切块大小（仅规则网格模式）
    int lm_row_blocks_tail_;  // 行尾块所在核的行块数（仅规则网格模式，考虑尾块）
    int lm_col_blocks_tail_;  // 列尾块所在核的列块数（仅规则网格模式，考虑尾块）
};

template <SplitMode S, typename T, std::size_t Dims>
constexpr Tile<S, Dims> make_tile(const Tensor<T, Dims> &tensor,
                                  const std::array<std::size_t, Dims> &tile_shape) {
    return Tile<S, Dims>(tensor, tile_shape);
}

}  // namespace hal
}  // namespace teco
#endif  // TECO_HAL_INCLUDE_TECO_HAL_TILE_HPP