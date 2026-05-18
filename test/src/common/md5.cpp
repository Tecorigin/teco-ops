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
 
#include "common/md5.h"
namespace optest {
MD5::MD5() {
    a_ = MD5_A;
    b_ = MD5_B;
    c_ = MD5_C;
    d_ = MD5_D;
}

MD5::~MD5() {
    delete str_bytes_;
    str_bytes_ = nullptr;
}

void MD5::padding(std::string input) {
    // 涉及大端和小端（小端）
    unsigned int groups = ((input.length() + 1) * 8 / 512) + 1;  // 512bit
    str_bytes_length_ = groups * 16;  // int:4*16=64bit, so 512/64=16 16个int
    str_bytes_ = new unsigned int[str_bytes_length_];
    // 全置0
    for (unsigned int i = 0; i < str_bytes_length_; i++) {
        str_bytes_[i] = 0;
    }
    // 将4个char字节合成一个int
    for (unsigned int i = 0; i < input.length(); i++) {
        str_bytes_[i >> 2] |= (input[i]) << ((i % 4) * 8);
    }
    // 补1 // 尾部添加1 一个unsigned int保存4个字符信息,所以用128左移
    str_bytes_[input.length() >> 2] |= 0x80 << (((input.length() % 4)) * 8);
    // 2字节记录原长度
    str_bytes_[str_bytes_length_ - 2] = input.length() * 8;
}

void MD5::loopChange(unsigned int *M) {
    unsigned int f, g;
    unsigned int atemp = a_;
    unsigned int btemp = b_;
    unsigned int ctemp = c_;
    unsigned int dtemp = d_;
    for (unsigned int i = 0; i < 16 * 4; i++) {
        if (i < 16) {
            f = MD5_F(btemp, ctemp, dtemp);
            g = i;
        } else if (i < 32) {
            f = MD5_G(btemp, ctemp, dtemp);
            g = (5 * i + 1) % 16;
        } else if (i < 48) {
            f = MD5_H(btemp, ctemp, dtemp);
            g = (3 * i + 5) % 16;
        } else {
            f = MD5_I(btemp, ctemp, dtemp);
            g = (7 * i) % 16;
        }
        unsigned int tmp = dtemp;
        dtemp = ctemp;
        ctemp = btemp;
        btemp += md5_shift((atemp + f + MD5_AC[i] + M[g]), MD5_SHIFT[i]);
        atemp = tmp;
    }
    a_ += atemp;
    b_ += btemp;
    c_ += ctemp;
    d_ += dtemp;
}

std::string MD5::changeHex(int a) {
    std::string str = "";
    for (int i = 0; i < 4; i++) {
        std::string str_tmp = "";
        int b = ((a >> i * 8) % (1 << 8)) & 0xff;  // 逆序处理每个字节
        for (int j = 0; j < 2; j++) {
            str_tmp.insert(0, 1, MD5_STR16[b % 16]);
            b = b / 16;
        }
        str += str_tmp;
    }
    return str;
}

std::string MD5::getMD5(std::string input) {
    padding(input);
    for (int i = 0; i < str_bytes_length_ / 16; i++) {
        loopChange(str_bytes_ + 16 * i);
    }
    return changeHex(a_).append(changeHex(b_)).append(changeHex(c_)).append(changeHex(d_));
}

std::string MD5::getMD5(size_t input) { return getMD5(std::to_string(input)); }

}  // namespace optest
