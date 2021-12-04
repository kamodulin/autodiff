#!/usr/bin/env bash

tests=(
    tests/test_dual.py
    tests/test_operations.py
    tests/test_node.py
    tests/test_reverse_operations.py
)

test='pytest'

if [[ $# -gt 0 && ${1} != 'pytest' ]]; then
	test=${1}	
fi

if [[ $# -gt 1 && ${2} == 'coverage' ]]; then
	option='--cov=src/autodiff --cov-report=term-missing'
	if [[ $# -gt 2 ]]; then
		option=${@:3}
	fi
    driver="${test} ${option}"
elif [[ $# -gt 1 && ${2} == 'unittest'* ]]; then
    driver="${test} -v"
fi

export PYTHONPATH="$PWD/src/"
# run the tests
if [[ $# -gt 1 && ${2} == 'coverage' ]]; then
	${driver}
else
	${driver} ${tests[@]}
fi
