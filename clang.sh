#!/bin/bash

for file in $(find . -name "*.cu" -o -name "*.cpp" -o -name "*.h" -o -name "*.c" | grep -v "build"); do
    clang-format -i $file
done
