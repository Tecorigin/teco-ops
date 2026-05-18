import os
dep_path = "thirdparty"

deps = [{
        "src": "protobuf-3.21.8",
        "build": "protobuf"},
        {
        "src": "googletest-release-1.12.1",
        "build": "googletest"
        }]

for dep in deps:
    build = os.path.abspath(os.path.join(os.path.join(dep_path, dep["build"])))
    src = os.path.abspath(os.path.join(os.path.join(dep_path, dep["src"])))
    if not os.path.exists(src):
        cmd = f"cd {dep_path}; tar -zxf {dep['src']}.tar.gz"
        os.system(cmd)

    if not os.path.exists(build):
        cmd = f"cd {src}; rm -rf build; mkdir build; cd build; cmake -DCMAKE_INSTALL_PREFIX={build} -Dprotobuf_BUILD_TESTS=OFF ..; make -j32 && make install; cd -"
        os.system(cmd)
