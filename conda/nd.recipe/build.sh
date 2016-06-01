#!/bin/bash
set -ex

cd $RECIPE_DIR

echo Setting the compiler...
if [ `uname` == Linux ]; then
  EXTRA_CMAKE_ARGS=-DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++
elif [ `uname` == Darwin ]; then
  EXTRA_CMAKE_ARGS=-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9
fi

cd ../../
$PYTHON setup.py build_ext --target=nd --extra-cmake-args=$EXTRA_CMAKE_ARGS install --single-version-externally-managed --record=record.txt || exit 1
