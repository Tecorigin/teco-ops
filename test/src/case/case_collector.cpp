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
 
#include <algorithm>
#include "case/case_collector.h"

extern optest::GlobalVar global_var;

Collector::Collector(const std::string &al_type, const std::string &name) {
    op_name_ = name;
    al_type_ = al_type;
}

size_t Collector::num() { return list().size(); }
bool IsDir(std::string path) {
    struct stat sb;
    if (stat(path.c_str(), &sb) == -1) return false;
    return S_ISDIR(sb.st_mode);
}

bool IsFile(std::string path) {
    struct stat sb;
    if (stat(path.c_str(), &sb) == -1) return false;
    return S_ISREG(sb.st_mode);
}

void Collector::grep_case(std::string dir, std::vector<std::string> &files) {
    DIR *dp = opendir(dir.c_str());
    if (dp == NULL) {  // it's not dir or not exit
        return;
    } else {
        // it is dir, grep all files
        struct dirent *dirp;
        while ((dirp = readdir(dp)) != NULL) {
            std::string file_path = dir + "/" + std::string(dirp->d_name);
            if (IsDir(file_path)) {
                std::string sub_dir = std::string(dirp->d_name);
                if (sub_dir[sub_dir.length() - 1] != '.') {
                    // is dir and not "." or ".."
                    grep_case(file_path, files);
                } else {
                    // is "." or ".."
                }
            } else if (IsFile(file_path) || dirp->d_type == DT_UNKNOWN) {
                // is file
                files.push_back(file_path);
            }
            // else maybe . or .. or else file type.
        }
        closedir(dp);
    }
}

void Collector::grep_dir(std::string dir, std::vector<std::string> &op_dirs) {
    DIR *dp = opendir(dir.c_str());
    if (dp == NULL) {  // it's not dir or not exit
        return;
    } else {
        // it is dir, grep all files
        struct dirent *dirp;
        while ((dirp = readdir(dp)) != NULL) {
            struct stat buf;
            std::string file_path = dir + "/" + std::string(dirp->d_name);
            stat(file_path.c_str(), &buf);
            if (S_ISDIR(buf.st_mode)) {
                // if (dirp->d_type == DT_DIR) {  // not all platform support, keng
                std::string sub_dir = std::string(dirp->d_name);
                if (strcmp(op_name_.c_str(), sub_dir.c_str()) == 0) {
                    op_dirs.push_back(dir + "/" + sub_dir);
                } else if (sub_dir[sub_dir.length() - 1] != '.') {
                    grep_dir(dir + "/" + sub_dir, op_dirs);
                } else {
                    // is "." or ".."
                }
            }
        }
        closedir(dp);
    }
}

std::vector<std::string> Collector::read_case(std::string cases_list) {
    // /**/add/add_***.pb\r
    std::vector<std::string> res;
    std::string op_dir = "/" + op_name_ + "/";
    std::ifstream fin(cases_list, std::ios::in);

    if (!fin.is_open()) {
        ALLOG(ERROR) << "Can't open " << cases_list;
        return res;  // return empty
    }

    std::string case_path;
    while (getline(fin, case_path)) {
        if (case_path.size() == 0 || case_path[0] == '#') {
            case_path.clear();
            continue;
        }
        if (case_path.find(op_dir) != std::string::npos) {
            if (case_path.back() == '\r' || case_path.back() == '\n') {
                res.push_back(case_path.substr(0, case_path.length() - 1));  // remove "\r"
            } else {
                res.push_back(case_path.substr(0, case_path.length()));
            }
        }
        case_path.clear();
    }
    fin.close();
    return res;
}

std::string Collector::current_dir() {
    char *buffer = NULL;
    if ((buffer = getcwd(NULL, 0)) == NULL) {
        ALLOG(ERROR) << "Get current dir failed.";
        return std::string("");  // return empty
    } else {
        std::string current_dir = std::string(buffer);
        free(buffer);
        return current_dir + "/";
    }
    return std::string("");
}

std::vector<std::string> Collector::list_by_case_path(std::string path) {
    if (path.find("/" + op_name_ + "/") == std::string::npos) {
        // case path is not belong this op
        return {};
    }
    if (path[0] != '/') {
        return {current_dir() + path};
    } else {
        return {path};
    }
}

std::vector<std::string> Collector::list_by_case_list(std::string list_file) {
    if (list_file[0] != '/') {
        // turn to abs
        list_file = current_dir() + list_file;
    }
    return read_case(list_file);
}

std::vector<std::string> Collector::list_by_case_dir(std::string case_dir) {
    std::vector<std::string> res;
    std::string suffix_txt = ".prototxt";
    std::string suffix_pb = ".pb";

    // if (case_dir.back() != '/') {
    //     case_dir += "/";
    // }

    // get all the files in dir
    std::vector<std::string> all_files;
    if (true == optest::getEnv("DEVICEOP_GTEST_CASE_RECURSIVE_SEARCH", false)) {
        // found env
        std::vector<std::string> op_dirs;
        grep_dir(case_dir, op_dirs);
        for (auto op_dir : op_dirs) {
            grep_case(op_dir, all_files);
        }
    } else {
        // no env
        if (case_dir.find("/" + op_name_ + "/") == std::string::npos) {
            std::vector<std::string> op_dirs;
            grep_dir(case_dir, op_dirs);
            for (auto op_dir : op_dirs) {
                grep_case(op_dir, all_files);
            }
        } else {
            grep_case(case_dir, all_files);
        }
    }

    for (auto name : all_files) {
        if (name.find("invalid") == std::string::npos) {  // no "invalid" in name
            if (name.substr(name.size() - suffix_pb.size(), suffix_pb.size()) == suffix_pb) {
                res.push_back(name);
            } else if (name.substr(name.size() - suffix_txt.size(), suffix_txt.size()) ==
                       suffix_txt) {
                res.push_back(name);
            }
        }
    }
    return res;
}

std::vector<std::string> Collector::list() {
    // for --case_path
    // vector<> size is 1
    if (!global_var.case_path_.empty()) {
        return list_by_case_path(global_var.case_path_);
    }

    // for --cases_list
    std::vector<std::string> case_names;
    if (!global_var.cases_list_.empty()) {
        case_names = list_by_case_list(global_var.cases_list_);
    } else if (!global_var.cases_dir_.empty()) {
        // for --cases_dir
        case_names = list_by_case_dir(global_var.cases_dir_);
    } else {
        std::string case_dir = "";
        const char *env = std::getenv("TECOAL_TECOTEST_PROJECT");
        if (env == nullptr) {
            case_dir = "../zoo/" + al_type_;
        } else {
            case_dir = std::string(env) + "/zoo/" + al_type_;
        }
        case_names = list_by_case_dir(case_dir);
    }

    auto fisher_shuffle = [](std::vector<std::string> res, int n) -> std::vector<std::string> {
        std::vector<std::string> res_n;
        n = std::min(n, int(res.size()));
        std::srand(std::time(nullptr));
        for (int i = 0; i < n; i++) {
            int rand = std::rand() % res.size();
            res_n.push_back(res[rand]);
            res.erase(res.begin() + rand);
        }
        return res_n;
    };
    // rand_n
    if (global_var.rand_n_ != -1) {
        case_names = fisher_shuffle(case_names, global_var.rand_n_);
    }

    // shuffle
    if (global_var.shuffle_) {
        std::random_shuffle(case_names.begin(), case_names.end());
    }
    return case_names;
}
