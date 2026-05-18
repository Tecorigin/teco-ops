
/**
 * tool.h
 * @author: HPE
 */

#include <string>
#include <stdexcept>
#include "common/bfloat16.h"


float BFloat16::bfloat162float(const uint16_t a) {
    float f;
    unsigned int u = static_cast<unsigned int>(a) << 16;
    memcpy(&f, &u, sizeof(f));
    return f;
}

unsigned short BFloat16::internalFloat2BFloat16(const float f, unsigned int &sign,
    unsigned int &remainder) {
    unsigned int x;
    memcpy(&x, &f, sizeof(f));
    if ((x & 0x7fffffffU) > 0x7f800000U) {
        sign = 0U;
        remainder = 0U;
        return static_cast<unsigned short>(0x7fffU);
    }
    sign = x >> 31U;
    remainder = x << 16U;
    return static_cast<unsigned short>(x >> 16U);
}

uint16_t BFloat16::float2bfloat16Rn(const float a) {
    uint16_t r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r = internalFloat2BFloat16(a, sign, remainder);
    if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((r & 0x1U) != 0U))) {
        r++;
    }
    return r;
}

uint16_t BFloat16::float2bfloat16Rz(const float a) {
    uint16_t r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r = internalFloat2BFloat16(a, sign, remainder);
    return r;
}

uint16_t BFloat16::float2bfloat16Rd(const float a) {
    uint16_t r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r = internalFloat2BFloat16(a, sign, remainder);
    if ((remainder != 0U) && (sign != 0U)) {
        r++;
    }
    return r;
}

uint16_t BFloat16::float2bfloat16Ru(const float a) {
    uint16_t r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r = internalFloat2BFloat16(a, sign, remainder);
    if ((remainder != 0U) && (sign == 0U)) {
        r++;
    }
    return r;
}

float Tools::castBfloat16ToFloat32(const uint16_t a) {
    return BFloat16::bfloat162float(a);
}

uint16_t Tools::castFloat32ToBFloat16(const float a, BFloat16::Mode mode) {
    switch (mode) {
        case BFloat16::Mode::ROUND_TO_NEAREST_EVEN: {
            return BFloat16::float2bfloat16Rn(a);
        } break;
        case BFloat16::Mode::ROUND_TOWARDS_ZERO: {
            return BFloat16::float2bfloat16Rz(a);
        } break;
        case BFloat16::Mode::ROUND_DOWN: {
            return BFloat16::float2bfloat16Rd(a);
        } break;
        case BFloat16::Mode::ROUND_UP: {
            return BFloat16::float2bfloat16Ru(a);
        } break;
        default: {
            throw std::runtime_error("convert mode is not exist!");
        } break;
    }
}

float Tools::castHalfToFloat32(int16_t src) {
    if (sizeof(int16_t) == 2) {
        int re = src;
        float f = 0.;
        int sign = (re >> 15) ? (-1) : 1;
        int exp = (re >> 10) & 0x1f;
        int eff = re & 0x3ff;
        float half_max = 65504.;
        float half_min = -65504.;  // or to be defined as infinity
        if (exp == 0x1f && eff) {
            // when half is nan, float also return nan, reserve sign bit
            int tmp = (sign > 0) ? 0xffffffff : 0x7fffffff;
            return *(float *)&tmp;
        } else if (exp == 0x1f && sign == 1) {
            // add upper bound of half. 0x7bff： 0 11110 1111111111 =  65504
            return half_max;
        } else if (exp == 0x1f && sign == -1) {
            // add lower bound of half. 0xfbff： 1 11110 1111111111 = -65504
            return half_min;
        }
        if (exp > 0) {
            exp -= 15;
            eff = eff | 0x400;
        } else {
            exp = -14;
        }
        int sft;
        sft = exp - 10;
        if (sft < 0) {
            f = (float)sign * eff / (1 << (-sft));
        } else {
            f = ((float)sign) * (1 << sft) * eff;
        }
        return f;
    } else if (sizeof(int16_t) == 4) {
        // using float
        return src;
    }
}

int Tools::mkdirIfNotExist(const char *pathname) {
        struct stat dir_stat = {};
        if (stat(pathname, &dir_stat) != 0) {
            if (mkdir(pathname, 0777) != 0) {
                return errno;
            }
            return 0;
        } else if (!S_ISDIR(dir_stat.st_mode)) {
            return ENOTDIR;
        }
        return 0;
    }

    int Tools::mkdirRecursive(const char *pathname) {
        // let caller ensure pathname is not null
        const char path_token = '/';
        size_t pos = 0;
        const std::string pathname_view(pathname);
        while (pos < pathname_view.size()) {
            auto find_path_token = pathname_view.find(path_token, pos);
            if (find_path_token == std::string::npos) {
                return mkdirIfNotExist(pathname_view.c_str());
            }
            int ret = mkdirIfNotExist(pathname_view.substr(0, find_path_token + 1).c_str());
            if (ret) return ret;
            pos = find_path_token + 1;
        }
        return 0;
    }


