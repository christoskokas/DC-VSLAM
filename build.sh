echo "Downloading all apt-get packages needed ..."

sudo apt-get install cmake

sudo apt-get install libgoogle-glog-dev libgflags-dev

sudo apt-get install libatlas-base-dev

sudo apt-get install libeigen3-dev

sudo apt-get install libsuitesparse-dev

sudo apt-get install libyaml-cpp-dev

sudo apt install libopencv-dev python3-opencv

sudo apt-get install ros-noetic-navigation

sudo apt-get install ros-noetic-nav-msgs

echo "Downloading and building ceres-solver.2-1-0 ..."

mkdir packages
cd packages

git clone --branch 2.1.0 --depth 1 https://github.com/ceres-solver/ceres-solver.git


cd ceres-solver
mkdir build
cd build
cmake ..
make -j3
sudo make install

cd ../../

echo "Downloading and building Pangolin ..."

git clone --recursive --branch v0.8 --depth 1 https://github.com/stevenlovegrove/Pangolin.git

cd Pangolin

./scripts/install_prerequisites.sh recommended

cmake -B build
cmake --build build

cd ../../
