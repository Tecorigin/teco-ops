/**
 * tool.h
 * @author: HPE
 */

#ifndef TEST_FRAME_WORK_TECOTEST_SRC_COMMON_BFLOAT16_H_
#define TEST_FRAME_WORK_TECOTEST_SRC_COMMON_BFLOAT16_H_

#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstring>


class BFloat16 {
 public:
    enum class Mode {
        ROUND_TO_NEAREST_EVEN = 0,
        ROUND_TOWARDS_ZERO = 1,
        ROUND_DOWN = 2,
        ROUND_UP = 3,
    };
    // ingroup CUDA_MATH__BFLOAT16_MISC
    // brief Converts nv_bfloat16 number to float.
    //
    // details Converts nv_bfloat16 number a to float.
    // param[in] a - float. Is only being read.
    //
    // returns float
    // a converted to float.
    // internal
    // exception-guarantee no-throw guarantee
    // behavior reentrant, thread safe
    // endinternal
    //
    static float bfloat162float(const uint16_t a);
    // ingroup CUDA_MATH__BFLOAT16_MISC
    // brief Converts float number to nv_bfloat16 precision in round-to-nearest-even mode
    // and returns nv_bfloat16 with converted value.
    //
    // details Converts float number \p a to nv_bfloat16 precision in round-to-nearest-even mode.
    // param[in] a - float. Is only being read.
    // returns nv_bfloat16
    // a converted to nv_bfloat16.
    // internal
    // exception-guarantee no-throw guarantee
    // behavior reentrant, thread safe
    // endinternal
    //
    static uint16_t float2bfloat16Rn(const float a);
    // ingroup CUDA_MATH__BFLOAT16_MISC
    // brief Converts float number to nv_bfloat16 precision in round-towards-zero mode
    // and returns nv_bfloat16 with converted value.
    // details Converts float number a to nv_bfloat16 precision in round-towards-zero mode.
    // param[in] a - float. Is only being read.
    // returns nv_bfloat16
    // a converted to nv_bfloat16.
    // internal
    // exception-guarantee no-throw guarantee
    // behavior reentrant, thread safe
    // endinternal
    //
    static uint16_t float2bfloat16Rz(const float a);
    // ingroup CUDA_MATH__BFLOAT16_MISC
    // brief Converts float number to nv_bfloat16 precision in round-down mode
    // and returns \p nv_bfloat16 with converted value.
    // details Converts float number \p a to nv_bfloat16 precision in round-down mode.
    // param[in] a - float. Is only being read.
    // returns nv_bfloat16
    // a converted to nv_bfloat16.
    // internal
    // exception-guarantee no-throw guarantee
    // behavior reentrant, thread safe
    // endinternal
    //
    static uint16_t float2bfloat16Rd(const float a);

    // ingroup CUDA_MATH__BFLOAT16_MISC
    // brief Converts float number to nv_bfloat16 precision in round-up mode
    // and returns \p nv_bfloat16 with converted value.
    //
    // details Converts float number \p a to nv_bfloat16 precision in round-up mode.
    // param[in] a - float. Is only being read.
    //
    // returns nv_bfloat16
    // a converted to nv_bfloat16.
    // internal
    // exception-guarantee no-throw guarantee
    // behavior reentrant, thread safe
    // endinternal
    //
    static uint16_t float2bfloat16Ru(const float a);

 private:
   static unsigned short internalFloat2BFloat16(const float f, unsigned int &sign, unsigned int &remainde); // NOLINT
};
class Tools {
 public:
    static float castHalfToFloat32(int16_t src);
    static int mkdirRecursive(const char *pathname);

    static float castBfloat16ToFloat32(const uint16_t a);
    static uint16_t castFloat32ToBFloat16(
        const float a, BFloat16::Mode mode = BFloat16::Mode::ROUND_TO_NEAREST_EVEN);

 private:
    static int mkdirIfNotExist(const char *pathname);
};

// struct bfloat16{
//    uint16_t val;
//    bfloat16(uint16_t f){
//       val = f;
//    }
//    bfloat16& operator=(uint16_t value){
//       val = value;
//       return *this;
//    }
//    bfloat16& operator=(float f){
//       val = BFloat16::float2bfloat16Rd(f);
//       return *this;
//    }
//    bfloat16& operator=(double f){
//       float m = (float) f;
//       val = BFloat16::float2bfloat16Rd(m);
//       return *this;
//    }

// };
#endif  // TEST_FRAME_WORK_TECOTEST_SRC_COMMON_BFLOAT16_H_
