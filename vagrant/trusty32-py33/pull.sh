#!/usr/bin/env bash

pushd ~/libdynd || exit 1
git pull --rebase origin master || exit 1
popd

pushd ~/dynd-python || exit 1
git pull --rebase origin master || exit 1
popd

