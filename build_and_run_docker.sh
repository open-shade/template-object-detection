docker build . -t shaderobotics/resnet-ros2

echo "Starting container..."
echo "====================="

# docker run -it --entrypoint=/bin/bash shaderobotics/resnet-ros2 -i

docker run -a STDOUT \
  -v /dev/shm:/dev/shm \
  shaderobotics/resnet-ros2
