#!/bin/sh

git submodule update --recursive --remote

cd lib/json || exit 1
mkdir -p build
cd build || exit 1
cmake ..
make -j16
sudo make install
cd ..
cd ../..


cd lib/quickcpplib || exit 1
mkdir build
cd build || exit 1
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -- _hl -j 16
make -j16
sudo make install
cd ..
cd ../..


cd lib/outcome || exit 1
mkdir build
cd build || exit 1
cmake ..
make -j16
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -- _hl -j 16
cd ..
cd ../..

cd lib/llfio || exit 1
mkdir build
cd build || exit 1
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -- _dl -j 16
make -j16 -DLLFIO_HEADERS_ONLY=0 -DLLFIO_DYN_LINK
sudo make install
cd ..
cd ../..