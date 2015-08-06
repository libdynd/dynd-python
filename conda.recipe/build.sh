#!/bin/bash
set -ex

cd $RECIPE_DIR

echo Setting the compiler...
if [ `uname` == Linux ]; then
    export CC="$PREFIX/bin/gcc"
    export CXX="$PREFIX/bin/g++"
elif [ `uname` == Darwin ]; then
    CPPFLAGS="-stdlib=libc++"
    EXTRAOPTIONS="-DCMAKE_OSX_DEPLOYMENT_TARGET=10.8"
    MACOSX_DEPLOYMENT_TARGET=10.8
else
    CPPFLAGS=
    EXTRAOPTIONS=
fi

cd ..
sudo python setup.py install #|| exit 1
