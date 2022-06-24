FROM ros:foxy-ros-core

RUN apt update
RUN apt install -y curl

RUN sh -c 'echo "deb [arch=amd64,arm64] http://repo.ros2.org/ubuntu/main `lsb_release -cs` main" > /etc/apt/sources.list.d/ros2-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN apt update
RUN apt install -y python3-colcon-common-extensions python3-pip

WORKDIR /home/shade
RUN mkdir shade_ws
WORKDIR /home/shade/shade_ws
RUN mkdir src

COPY . ./src/resnet_ros2

RUN pip install -e ./src/resnet_ros2

RUN colcon build

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/foxy/setup.bash && source ./install/setup.bash && ros2 run resnet_ros2 resnet_ros2"]
