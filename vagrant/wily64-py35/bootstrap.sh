#!/usr/bin/env bash

set -xe

# Install some things using apt
apt-get update -y -q
apt-get install -y -q build-essential g++ gfortran clang cmake git dos2unix

# do this so we don't have to put chown everywhere and su -c all the thingz
su -c "source /vagrant/bin/vagrant.sh" vagrant
