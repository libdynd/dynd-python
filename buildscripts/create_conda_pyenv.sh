#!/bin/bash
#
# $1 is the python version
# $2 is the directory in which the conda env is created

rm -rf $2
~/anaconda/bin/conda create --yes -p $2 python=$1 cython scipy || exit 1
