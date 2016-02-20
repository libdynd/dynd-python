#!/usr/bin/env bash

set -xe

dynd_env=$HOME/miniconda3/envs/dynd
dynd_bin_dir=$dynd_env/bin


function install_conda()
{
    curl -sS -O https://repo.continuum.io/miniconda/$1 && \
        chmod +x ./$1 && \
        ./$1 -b && \
        echo 'PATH=$HOME/miniconda3/bin:$PATH' >> ~/.bashrc && \
        export PATH=$HOME/miniconda3/bin:$PATH && \
        tail -1 ~/.bashrc && \
        conda create -q -n dynd --yes ipython cython numpy scipy pyyaml nose
}


function install_libdynd()
{
    git clone https://github.com/libdynd/libdynd.git && \
        pushd libdynd && \
        git submodule update --init && \
        mkdir build && \
        pushd build && \
        cmake -DCMAKE_INSTALL_PREFIX=$dynd_env .. && \
        make && \
        make install && \
        sudo ldconfig && \
        popd && \
        popd
}

function install_dynd_python()
{
    git clone https://github.com/libdynd/dynd-python.git && \
        pushd dynd-python && \
        python setup.py install && \
        popd
}

# Anaconda Python (miniconda) with Python dependencies
install_conda Miniconda3-latest-Linux-x86_64.sh || exit 1
source activate dynd || exit 1
install_libdynd || exit 1
install_dynd_python || exit 1

# Utility scripts
for f in pull.sh build.sh refresh.sh
do
    chmod u+x /vagrant/bin/$f
done

unset dynd_env
unset dynd_bin_dir
