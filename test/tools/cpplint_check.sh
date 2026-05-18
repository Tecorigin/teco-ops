root=$(dirname "$PWD")

cd $root
#cpplint --filter="-build/header_guard" --recursive . --exclude=src/host-implement-typeless --exclude=thirdParty --output 2>&1 | tee report/cpplint-result.xml
cpplint --exclude=test_proto --exclude=exter_lib --recursive . --output 2>&1 | tee report/cpplint-result.xml
