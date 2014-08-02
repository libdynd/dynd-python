#!/usr/bin/env bash

# Make sure the package information is up-to-date
apt-get update -y && \
    apt-get install -y build-essential g++ gfortran clang cmake git dos2unix

# do this so we don't have to put chown everywhere and su -c all the thingz
su -c "source /vagrant/bin/vagrant.sh" vagrant || exit 1
