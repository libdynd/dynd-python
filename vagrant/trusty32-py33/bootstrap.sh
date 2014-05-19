#!/usr/bin/env bash

# Make sure the package information is up-to-date
apt-get update || exit 1

# Compilers
apt-get install -y g++-4.8 || exit 1
apt-get install -y gfortran-4.8 || exit 1
apt-get install -y clang-3.4 || exit 1

# Configuration
apt-get install -y cmake || exit 1

# Source control
apt-get install -y git || exit 1

# Utilities
apt-get install -y dos2unix || exit 1

# Anaconda Python (miniconda) with Python dependencies
echo Downloading Miniconda...
curl -O http://repo.continuum.io/miniconda/Miniconda3-3.3.0-Linux-x86.sh || exit 1
su -c 'bash Miniconda3-*.sh -b -p ~/anaconda' vagrant || exit 1
# Install dependencies
su -c '~/anaconda/bin/conda install --yes ipython cython numpy scipy pyyaml nose' vagrant || exit 1
# Add anaconda to the PATH
printf '\nexport PATH=~/anaconda/bin:~/bin:$PATH\n' >> .bashrc
chown vagrant .bashrc
export PATH=~/anaconda/bin:~/bin:$PATH
mkdir ~/bin
chown -R vagrant ~/bin

export CC=gcc
export CXX=g++

# Clone and install libdynd
git clone https://github.com/ContinuumIO/libdynd.git || exit 1
mkdir libdynd/build
chown -R vagrant libdynd
pushd libdynd/build
su -c 'cmake -DCMAKE_INSTALL_PREFIX=~/anaconda -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} ..' vagrant || exit 1
su -c 'make' vagrant || exit 1
su -c 'make install' vagrant || exit 1
ldconfig
popd

# Clone and install dynd-python
git clone https://github.com/ContinuumIO/dynd-python.git || exit 1
mkdir dynd-python/build
chown -R vagrant dynd-python
pushd dynd-python/build
su -c 'cmake -DDYND_ELWISE_MAX=5 -DPYTHON_EXECUTABLE=~/anaconda/bin/python -DCYTHON_EXECUTABLE=~/anaconda/bin/cython -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} ..' vagrant || exit 1
su -c 'make' vagrant || exit 1
su -c 'make install' vagrant || exit 1
popd

# Utility scripts
for FILE in pull.sh build.sh refresh.sh
do
    chown -R vagrant ~/bin/${FILE}
    chmod u+x ~/bin/${FILE}
done

