#!/usr/bin/env bash

tests=(
    autodiff/tests/test_dual.py
    autodiff/tests/test_operations.py
)

test='pytest'

if [[ $# -gt 0 && ${1} != 'pytest' ]]; then
	test=${1}	
fi

if [[ $# -gt 1 && ${2} == 'coverage' ]]; then
	option='--cov=autodiff --cov-report=term-missing'
	if [[ $# -gt 2 ]]; then
		option=${@:3}
	fi
    driver="${test} ${option}"
elif [[ $# -gt 1 && ${2} == 'unittest'* ]]; then
    driver="${test} -v"
fi

export PYTHONPATH="$PWD"
# run the tests
${driver} ${tests[@]}