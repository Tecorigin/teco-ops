#! /bin/bash
#set -e
#set -x

_shell="$(ps -p $$ --no-headers -o comm=)"
if [[ $_shell == "zsh" ]]; then
  home_path="$(dirname $(readlink -e "${(%):-%N}"))"
elif [[ $_shell == "bash" || $_shell == "sh" ]]; then
  home_path="$(dirname $(readlink -e "${BASH_SOURCE[0]}"))"
fi

export PYTHONPATH=$home_path/tools:$PYTHONPATH
export PYTHONPATH=$home_path/tools/gen_case:$PYTHONPATH
export PYTHONPATH=$home_path/test_tools2:$PYTHONPATH
export PYTHONPATH=$home_path/test_tools2/nodes:$PYTHONPATH
export PYTHONPATH=$home_path/zoo/teco:$PYTHONPATH
export TECOAL_TECOTEST_PROJECT=$home_path

curr_path=${PWD}/.git
father_path=${PWD}/../../../.git/modules/test/frame_work/tecotest
if [[ ! -d "${curr_path}" && ! -d "${father_path}" ]]; then
  # pass
  echo "Ready for building."
else
  if [[ -d "${curr_path}" ]]; then
    used_path=${curr_path}
  else
    used_path=${father_path}
  fi

  if [[ -f "${used_path}/hooks/pre-commit" ]]; then
    rm -rf ${used_path}/hooks/pre-commit
  fi
  echo "-- pre-commit hook inserted to ${used_path}/hooks."
  echo "-- Use git commit -n to bypass pre-commit hook."
  cp -f ${PWD}/tools/pre-commit ${used_path}/hooks/pre-commit

  if [[ -f "${used_path}/hooks/commit-msg" ]]; then
    rm -rf ${used_path}/hooks/commit-msg
  fi
  echo "-- commit-msg hook inserted to ${used_path}/hooks."
  cp -f ${PWD}/tools/commit-msg ${used_path}/hooks/commit-msg

  git config --global commit.template ${PWD}/tools/commit_template
  echo "-- commit template configured."

  echo "Ready for building."
fi

export TECOTEST_READY_TO_BUILD=ON
