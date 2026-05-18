#! /bin/bash
#set -e
#set -x

if [[ ! -d "${PWD}/.git/" ]]; then
  echo "Ready for building."
else
  if [[ -f "${PWD}/.git/hooks/pre-commit" ]]; then
    rm -rf ${PWD}/.git/hooks/pre-commit
  fi
  echo "-- pre-commit hook inserted to ${PWD}/.git/hooks."
  cp -f ${PWD}/tools/pre-commit ${PWD}/.git/hooks/pre-commit

  if [[ -f "${PWD}/.git/hooks/commit-msg" ]]; then
    rm -rf ${PWD}/.git/hooks/commit-msg
  fi
  echo "-- commit-msg hook inserted to ${PWD}/.git/hooks."
  cp -f ${PWD}/tools/commit-msg ${PWD}/.git/hooks/commit-msg

  git config --global commit.template ${PWD}/tools/commit_template
  echo "-- commit template configured."

  echo "Ready for building."
fi

export TECO_READY_TO_BUILD=ON