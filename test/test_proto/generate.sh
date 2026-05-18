dir=$(cd "$(dirname "$0")";pwd)

custom_protoc=${dir}/../thirdparty/protobuf/bin/protoc

if [ ! -f ${custom_protoc} ]; then
  echo [ERROR] ${custom_protoc} not found
  echo "----please rm -rf thirdparty, and rebuild"
  exit 1
fi

# set -x 

# python3 gen_proto.py
for al_type in "tecokernel"
do
  al_dir=${dir}/${al_type}
  
  find "${al_dir}" -name "*.pb.cc" -o -name "*.pb.h" -o -name "*_pb2.py" | while read -r pb_file; do
    rm -rf ${pb_file}
  done

  # ${custom_protoc} --python_out=${dir} --proto_path=${dir} ${al_dir}/common.proto
  # ${custom_protoc} --cpp_out=${dir} --proto_path=${dir} ${al_dir}/common.proto

  find "${al_dir}" -name "*.proto" ! -name "common.proto"| while read -r proto_file; do
  ${custom_protoc} --cpp_out=${dir} --proto_path=${dir} ${proto_file}
  ${custom_protoc} --python_out=${dir} --proto_path=${dir} ${proto_file}
  done
done

rm -rf *.pb.cc
rm -rf *.pb.h
rm -rf *_pb2.py

${custom_protoc} --cpp_out=$dir --proto_path=${dir} ${dir}/tensor.proto
${custom_protoc} --cpp_out=$dir --proto_path=${dir} ${dir}/tecokernel.proto
${custom_protoc} --cpp_out=$dir --proto_path=$dir ${dir}/optest.proto

${custom_protoc} --python_out=$dir --proto_path=$dir ${dir}/tensor.proto
${custom_protoc} --python_out=$dir --proto_path=$dir ${dir}/tecokernel.proto
${custom_protoc} --python_out=$dir --proto_path=$dir ${dir}/optest.proto
