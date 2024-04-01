sudo apt update && apt upgrade -y
sudo apt remove "*libfranka*" -y
sudo apt install -y  build-essential cmake git libpoco-dev libeigen3-dev
cd libfranka 
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
cmake --build .
cpack -G DEB
sudo dpkg -i libfranka*.deb
