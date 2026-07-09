source /opt/tecoai/setvars.sh
curpath="$(dirname $(readlink -e "${BASH_SOURCE[0]}"))"
echo ${curpath}
err=0

if [ $# -eq 0 ]; then
  pathList="."
  testList=$(ls)
else
  pathList="$*"
  testList=$pathList
fi

for p in ${testList}; do
  if [ -d $p ]; then
    cd $p
    subDirs=""
    subDirsNum=$(find . -type d |wc -l)
    if [[ "${subDirsNum}" != "1" ]]; then
      subDirs=$(find . -type d)
      for subDir in $subDirs
        do
        if [[ -d $subDir && ${subDir} != "." ]]; then
          cd $subDir
          make clean
          make all
          testexelist=$(ls *.out)
          for testexe in ${testexelist}; do
            ./${testexe}
              if [ $? -ne 0 ]; then ((err++)); fi
          done
        fi
        cd $curpath/$p
      done
    else
      echo $(pwd)
      make clean
      make all
      testexelist=$(ls *.out)
	    for testexe in ${testexelist}; do
	      ./${testexe}
          if [ $? -ne 0 ]; then ((err++)); fi
	    done
    fi
    cd $curpath
  fi
done

if [ ${err} == 0 ]; then
  echo "TOTAL CASE SUCCESS !!"
  exit 0
else
  echo "EXIST FAILURE !!"
  exit 1
fi
