import os
dep_path = "thirdparty"

import platform
import sys

def get_arch():
    """获取当前平台架构，统一返回标准化标识"""
    machine = platform.machine().lower()

    # x86/x86_64 系列
    if machine in ('x86_64', 'amd64', 'i386', 'i686', 'x86'):
        return 'x86'

    # 龙芯 (LoongArch)
    if machine in ('loongarch64', 'loongarch32', 'loongarch'):
        return 'loongson'

    # ARM 系列
    if machine.startswith('arm') or machine.startswith('aarch'):
        return 'arm'
    return machine

deps = []
arch = get_arch()
if arch == 'x86':
    deps.append({
        "src": "http://wb.tecorigin.com:8082/repository/teco-3rd-repo/tecohal/ubuntu22.04/x86_64/0.0.1/tecohal_0.0.1b0.tar.gz",
        "build": "teco-hal"})
elif arch == 'loongson':
    deps.append({
        "src": "http://wb.tecorigin.com:8082/repository/teco-3rd-repo/tecohal/LoongnixServer/23.1/Loongson_loongarch64/0.0.1/tecohal_0.0.1b0.tar.gz",
        "build": "teco-hal"})

for dep in deps:
    target_dir = os.path.abspath(os.path.join(os.path.join(dep_path, dep["build"])))

    if not os.path.exists(dep_path):
        os.makedirs(dep_path, exist_ok=True)

    if not os.path.exists(dep['build']):
        # wget 下载
        os.system(f"wget -q {dep['src']}")
        
        # 解压到 teco-hal 目录下
        os.system(f"tar -zxf tecohal*.tar.gz -C {dep_path}")
        os.system("rm -f tecohal*.tar.gz")
        os.system(f"mv {dep_path}/tecohal* {target_dir}")
        
