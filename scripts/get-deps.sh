#!/bin/bash -e

apt-get update || true #ignore fail here, because rocm docker is broken
apt-get install -y curl unzip

CMAKE=3.14.7
mkdir installers
pushd installers
curl -s -L https://github.com/Kitware/CMake/releases/download/v$CMAKE/cmake-$CMAKE-linux-x86_64.sh > cmake.sh
bash cmake.sh --prefix=/usr/local --skip-license

curl -s -L https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip > ninja.zip
unzip ninja.zip
mv ninja /usr/local/bin
popd

rm -rf installers
