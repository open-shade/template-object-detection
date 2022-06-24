sudo docker build . -t shaderobotics/resnet-ros2

echo "Starting container..."
echo "====================="

sudo docker run -it \
  -v /dev/shm:/dev/shm --entrypoint=/bin/bash \
  shaderobotics/resnet-ros2 -i

#sudo docker run -a STDOUT \
#  -v /dev/shm:/dev/shm \
#  shaderobotics/resnet-ros2
