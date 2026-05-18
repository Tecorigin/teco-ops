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
 
#include "common/tools.h"
#include "common/half.hpp"
namespace optest {

std::string subReplaceFirst(std::string str, std::string old_str, std::string new_str) {
    std::string dst_str = str;
    auto pos = dst_str.find(old_str, 0);
    if (pos == std::string::npos) return dst_str;
    return dst_str.replace(pos, old_str.size(), new_str);
}

bool isPathExist(std::string path) {
    if (access(path.c_str(), 0) == -1) return false;
    return true;
}

void createDir(std::string file) {
    std::string::size_type pos_start = 1;
    std::string::size_type pos = 0;
    pos = file.find('/', pos_start);
    while (pos != std::string::npos) {
        std::string dir = file.substr(0, pos);
        if (dir.empty()) return;
        if (!isPathExist(dir)) mkdir(dir.c_str(), 0755);

        pos_start = pos + 1;
        pos = file.find('/', pos_start);
    }
}

void removeDir(std::string path) {
    DIR *dir;
    struct dirent *dirp;
    struct stat buf;
    char *current_dir = getcwd(NULL, 0);
    if ((dir = opendir(path.c_str())) == NULL) {
        // ALLOG(WARNING) << path << " open fault.";
        return;
    }
    if (chdir(path.c_str()) == -1) {
        ALLOG(WARNING) << path << " enter fault.";
        return;
    }

    while (dirp = readdir(dir)) {
        if ((strcmp(dirp->d_name, ".") == 0) || (strcmp(dirp->d_name, "..") == 0)) continue;
        if (stat(dirp->d_name, &buf) == -1) {
            ALLOG(ERROR) << dirp->d_name << " stat fault.";
        }
        if (S_ISDIR(buf.st_mode)) {
            removeDir(dirp->d_name);
        } else if (remove(dirp->d_name) == -1) {
            ALLOG(ERROR) << dirp->d_name << " remove fault.";
        }
    }
    closedir(dir);
    chdir(current_dir);
    if (rmdir(path.c_str()) == -1) {
        ALLOG(ERROR) << path << " remove fault.";
    }
}

void saveDataToFile(const std::string &file, void *data, size_t size) {
    createDir(file);
    std::ofstream fout(file, std::ios::out | std::ios::binary);
    if (!fout) {
        ALLOG(ERROR) << file << " open fault, and can not create";
        throw FileOpenFault(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }
    fout.write((char *)data, size);
    fout.close();
}

void readDataFromFile(const std::string &file, void *data, size_t size) {
    std::ifstream fin(file, std::ios::in | std::ios::binary);
    if (!fin) {
        ALLOG(ERROR) << file << " open fault.";
        throw FileOpenFault(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }
    fin.read((char *)data, size);
    fin.close();
}

bool isProtoChanged(const std::string &proto_path, const std::string &key_path) {
    // if (!proto_file.is_open()) {
    //     std::cerr << "prototxt open failed" << std::endl;
    //     return true;
    // }
    std::ifstream key_file(key_path);
    std::ifstream proto_file(proto_path);
    std::string proto_content =
        std::string(std::istreambuf_iterator<char>(proto_file), std::istreambuf_iterator<char>());
    std::string key_content =
        std::string(std::istreambuf_iterator<char>(key_file), std::istreambuf_iterator<char>());
    key_file.close();
    proto_file.close();

    if (proto_content.size() != key_content.size()) {
        return true;
    }
    for (int i = 0, j = proto_content.size() - 1; i < j; i++, j--) {
        if (proto_content[i] != key_content[i] || proto_content[j] != key_content[j]) {
            return true;
        }
    }

    return false;
}

bool getEnv(const std::string &env, bool default_ret) {
    char *env_temp = getenv(env.c_str());
    if (env_temp != NULL) {
        if (strcmp(env_temp, "ON") == 0 || strcmp(env_temp, "1") == 0) {
            return true;
        } else if (strcmp(env_temp, "OFF") == 0 || strcmp(env_temp, "0") == 0) {
            return false;
        } else {
            return default_ret;
        }
    } else {
        return default_ret;
    }
}

template <>
bool is_nan<half_float::half>(half_float::half arg) {
    // exponent all 0, significand not all 0: subnormal number
    // exponent all 1, significand not all 0: nan
    uint16_t value = *((uint16_t *)(&arg));
    return ((value & 0x7C00) == 0x7C00) && ((value & 0x03FF) != 0);
}

template <>
bool is_inf<half_float::half>(half_float::half arg) {
    // exponent all 1, significand all 0
    uint16_t value = *((uint16_t *)(&arg));
    return ((value & 0x7C00) == 0x7C00) && ((value & 0x03FF) == 0);
}

}  // namespace optest
