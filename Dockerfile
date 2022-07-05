ARG ROS_VERSION
FROM shaderobotics/ros:${ROS_VERSION}

ARG ROS_VERSION
ENV ROS_VERSION=$ROS_VERSION

WORKDIR /home/shade
RUN mkdir shade_ws
WORKDIR /home/shade/shade_ws
RUN mkdir src

RUN apt update
RUN apt install -y curl
RUN sh -c 'echo "deb [arch=amd64,arm64] http://repo.ros2.org/ubuntu/main `lsb_release -cs` main" > /etc/apt/sources.list.d/ros2-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update
RUN apt install -y python3-colcon-common-extensions python3-pip ros-${ROS_VERSION}-cv-bridge ros-${ROS_VERSION}-vision-opencv

RUN echo "#!/bin/bash" >> /home/shade/shade_ws/start.sh
RUN echo "source /opt/shade/setup.sh" >> /home/shade/shade_ws/start.sh
RUN echo "source ./install/setup.sh" >> ./start.sh
RUN echo "ros2 run resnet_ros2 resnet_ros2 2>&1 | tee ./output.txt" >> /home/shade/shade_ws/start.sh
RUN chmod +x ./start.sh

COPY . ./src/resnet_ros2

RUN pip3 install ./src/resnet_ros2
RUN colcon build

ENTRYPOINT ["/home/shade/shade_ws/start.sh"]
