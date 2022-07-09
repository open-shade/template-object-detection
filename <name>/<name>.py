import numpy
import os
from transformers import AutoFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image as PilImage
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import String
from cv_bridge import CvBridge

ALGO_VERSION = os.getenv("MODEL_NAME")

if not ALGO_VERSION:
    ALGO_VERSION = '<default here>'


def predict(image: Image):
    feature_extractor = AutoFeatureExtractor.from_pretrained(ALGO_VERSION)
    model = DetrForObjectDetection.from_pretrained(ALGO_VERSION)

    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)

    # Convert output to be between 0 and 1
    logits = torch.nn.functional.softmax(output.logits[0], dim=2)
    bboxes = output.pred_boxes[0]
    
    return logits, bboxes


class RosIO(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.declare_parameter('pub_image', False)
        self.declare_parameter('pub_boxes', True)
        self.image_subscription = self.create_subscription(
            Image,
            '/<name>/sub/image_raw',
            self.listener_callback,
            10
        )

        self.image_publisher = self.create_publisher(
            String,
            '/<name>/pub/image',
            1
        )
    
        self.detection_publisher = self.create_publisher(
            String,
            '/<name>/pub/detection_boxes',
            1
        )

    def get_detection_arr(self, logits, boxes):
        dda = Detection2DArray()

        detections = []
        self.counter += 1
        
        for i in range(len(logits)):
            detection = Detection2D()

            # don't send boudning boxes that have no object in them
            if logits[i].argmax(-1).item() == len(logits[i]) - 1:
                continue

            detection.header.stamp = self.get_clock().now().to_msg()
            detection.header.frame_id = str(self.counter)

            hypothesises = []

            for _class in range(len(logits[i])):
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = _class
                hypothesis.score = logits[_class].item()
                hypothesis.pose.pose.position.x = boxes[i][0].item()
                hypothesis.pose.pose.position.y = boxes[i][1].item()
                hypothesises.append(hypothesis)

            detection.results = hypothesises

            detection.bbox.center.x = boxes[i][0].item()
            detection.bbox.center.y = boxes[i][1].item()
            detection.bbox.center.theta = 0.0

            detection.bbox.size_x = boxes[i][2].item()
            detection.bbox.size_y = boxes[i][3].item()

            detections.append(detection)

        dda.detections = detections
        dda.header.stamp = self.get_clock().now().to_msg()
        dda.header.frame_id = str(self.counter)
        return dda


    def listener_callback(self, msg: Image):
        bridge = CvBridge()
        cv_image: numpy.ndarray = bridge.imgmsg_to_cv2(msg)
        converted_image = PilImage.fromarray(numpy.uint8(cv_image), 'RGB')
        logits, boxes = str(predict(converted_image))
        print(f'Predicted Bounding Boxes')

        if self.get_parameter('pub_image').value:
            self.image_publisher.publish(msg)

        if self.get_parameter('pub_boxes').value:
            detections = self.get_detection_arr(logits, boxes)
            self.detection_publisher.publish(detections)

        


def main(args=None):
    print('<name> Started')

    rclpy.init(args=args)

    minimal_subscriber = RosIO()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
