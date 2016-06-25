#!/usr/bin/env sh

cd $RECIPE_DIR/../..

$PYTHON setup.py build_ext --target=ndt install --single-version-externally-managed --record=record.txt || exit 1
