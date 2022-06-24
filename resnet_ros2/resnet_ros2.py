import numpy
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
from datasets import load_dataset
from PIL import Image as PilImage
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def predict(image: Image):
    feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/resnet-101')
    model = ResNetForImageClassification.from_pretrained('microsoft/resnet-101')

    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()

    return predicted_label, model.config.id2label[predicted_label]


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/image/image_raw',
            self.listener_callback,
            10)

    def listener_callback(self, msg: Image):
        # self.get_logger().info(msg.data)
        bridge = CvBridge()
        cv_image: numpy.ndarray = bridge.imgmsg_to_cv2(msg)
        # print(cv_image)
        # cv2.imshow('image', cv_image)
        # cv2.waitKey(0)
        converted_image = PilImage.fromarray(numpy.uint8(cv_image), 'RGB')
        # converted_image.show('image')
        print(predict(converted_image))


def main(args=None):
    print('Hi from resnet_ros2.')

    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    dataset = load_dataset('huggingface/cats-image')

    for i in dataset['test']['image']:
        print(predict(i))

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
