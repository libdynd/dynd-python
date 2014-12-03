#!/usr/bin/env bash

dynd_env=$HOME/miniconda3/envs/dynd
dynd_bin_dir=$dynd_env/bin


function install_conda()
{
    curl -O http://repo.continuum.io/miniconda/$1 && \
        chmod +x ./$1 && \
        ./$1 -b && \
        echo 'PATH=$HOME/miniconda3/bin:$PATH' >> ~/.bashrc && \
        export PATH=$HOME/miniconda3/bin:$PATH && \
        tail -1 ~/.bashrc && \
        conda create -n dynd --yes ipython cython numpy scipy pyyaml nose
}


function install_libdynd()
{
    git clone https://github.com/libdynd/libdynd.git && \
        mkdir libdynd/build && \
        pushd libdynd/build && \
        cmake -DCMAKE_INSTALL_PREFIX=$dynd_env .. && \
        make && \
        make install && \
        sudo ldconfig && \
        popd
}

function install_dynd_python()
{
    git clone https://github.com/libdynd/dynd-python.git && \
        mkdir dynd-python/build && \
        pushd dynd-python/build && \
        cmake -DDYND_ELWISE_MAX=5 \
        -DPYTHON_EXECUTABLE=$dynd_bin_dir/python \
        -DCYTHON_EXECUTABLE=$dynd_bin_dir/cython \
        -DCMAKE_INSTALL_PREFIX=$dynd_env .. && \
        make && \
        make install && \
        popd
}

# Anaconda Python (miniconda) with Python dependencies
install_conda Miniconda3-3.5.5-Linux-x86_64.sh || exit 1
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
