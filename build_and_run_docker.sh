sudo docker build . -t shaderobotics/resnet-ros2

echo "Starting container..."
echo "====================="

sudo docker run -a STDOUT \
  -v /dev/shm:/dev/shm \
  shaderobotics/resnet-ros2
