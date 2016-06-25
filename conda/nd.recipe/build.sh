#!/usr/bin/env sh

cd $RECIPE_DIR/../..

$PYTHON setup.py build_ext --target=nd install --single-version-externally-managed --record=record.txt || exit 1
