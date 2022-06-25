FROM ros:foxy-ros-core

WORKDIR /home/shade
RUN mkdir shade_ws
WORKDIR /home/shade/shade_ws
RUN mkdir src

RUN apt update && \
    apt install -y curl && \
    sh -c 'echo "deb [arch=amd64,arm64] http://repo.ros2.org/ubuntu/main `lsb_release -cs` main" > /etc/apt/sources.list.d/ros2-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt update && \
    apt install -y python3-colcon-common-extensions python3-pip

COPY . ./src/resnet_ros2

RUN pip3 install -e ./src/resnet_ros2 && \
    colcon build && \
    apt install -y python3-colcon-common-extensions python3-pip ros-foxy-cv-bridge ros-foxy-vision-opencv

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/foxy/setup.bash && source ./install/setup.bash && ros2 run resnet_ros2 resnet_ros2"]
