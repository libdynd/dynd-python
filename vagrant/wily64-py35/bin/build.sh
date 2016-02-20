#!/usr/bin/env bash

pushd ~/libdynd/build || exit 1
cmake ..
make
make install
sudo ldconfig
popd

pushd ~/dynd-python/build || exit 1
cmake ..
make
make install
popd


