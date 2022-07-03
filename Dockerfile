# check more detail on: https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:10.2-devel-ubuntu18.04

# Just in case we need it
ENV DEBIAN_FRONTEND noninteractive

# ==========> BASIC ELEMENTS <===========
RUN apt update && apt install -y git wget locate

# ==========> INSTALL ROS melodic <=============
RUN apt update && apt install -y curl lsb-release
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update && apt install -y ros-melodic-desktop-full
RUN apt-get install -y python-catkin-pkg \
    python-catkin-tools \
    python-empy \
    python-nose \
    libgtest-dev \
    ros-melodic-catkin \
    python-pip \
    python3-pip \
    ros-melodic-grid-map

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
# ==========> INSTALL ROS <=============

# ==========> INSTALL CUPY <=============
RUN pip install cupy==6.7.0 torch==1.4.0 torchvision==0.5.0

RUN apt-get install -y ros-melodic-pybind11-catkin \
                       ros-melodic-grid-map \
                       ros-melodic-jsk

RUN apt-get install -y libopencv-dev libeigen3-dev libgmp-dev libmpfr-dev libboost-all-dev

# upgrade CMake
# RUN mkdir -p /opt && \
#     pushd /opt && \
#     wget 'https://cmake.org/files/v3.14/cmake-3.14.0-Linux-x86_64.sh' && \
#     bash cmake-3.14.0-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir && \
#     popd && \
#     ln -s /opt/cmake-3.14.0-Linux-x86_64/bin/* /usr/local/bin

# RUN echo "export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:/usr/local/lib/python3.6/dist-packages:$PYTHONPATH" >> ~/.bashrc

# ==============> 
# RUn pip install --user --upgrade pip
# RUN pip install -r requirements.txt
# apt-get install python3-yaml