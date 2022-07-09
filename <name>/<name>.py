import numpy
import os
from transformers import AutoFeatureExtractor
import torch
from PIL import Image as PilImage
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge

ALGO_VERSION = os.getenv("MODEL_NAME")

if not ALGO_VERSION:
    ALGO_VERSION = '<default here>'


def predict(image: Image):
    feature_extractor = AutoFeatureExtractor.from_pretrained(ALGO_VERSION)
    # model = <name>ForImageClassification.from_pretrained(ALGO_VERSION)
    # Enter line here

    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)

    logits = output.logits
    bboxes = output.pred_boxes
    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()

    return predicted_label, model.config.id2label[predicted_label], bboxes


class RosIO(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.declare_parameter('pub_image', False)
        self.declare_parameter('pub_json', False)
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
        self.json_publisher = self.create_publisher(
            String,
            '/<name>/pub/json',
            1
        )
        self.detection_publisher = self.create_publisher(
            String,
            '/<name>/pub/detection_boxes',
            1
        )

        self.get_logger("Initialized Node")

    def get_detection_arr(self, df):
        dda = Detection2DArray()

        detections = []
        self.counter += 1

        for row in df.itertuples():
            detection = Detection2D()

            detection.header.stamp = self.get_clock().now().to_msg()
            detection.header.frame_id = str(self.counter)

            hypothesises = []
            hypothesis = ObjectHypothesisWithPose()
            hypothesises.append(hypothesis)
            detection.results = hypothesises
            detection.results[0].score = row.confidence
            # this code will have to change
            detection.results[0].pose.pose.position.x = (int(row.xmin) + int(row.xmax)) / 2
            detection.results[0].pose.pose.position.y = (int(row.ymin) + int(row.ymax)) / 2

            detection.bbox.center.x = (int(row.xmin) + int(row.xmax)) / 2
            detection.bbox.center.y = (int(row.ymin) + int(row.ymax)) / 2
            detection.bbox.center.theta = 0.0

            detection.bbox.size_x = (int(row.xmax) - int(row.xmin)) / 2
            detection.bbox.size_y = (int(row.ymax) - int(row.ymin)) / 2

            detections.append(detection)

        dda.detections = detections
        dda.header.stamp = self.get_clock().now().to_msg()
        dda.header.frame_id = str(self.counter)

        return dda


    def listener_callback(self, msg: Image):
        # self.get_logger().info(msg.data)
        bridge = CvBridge()
        cv_image: numpy.ndarray = bridge.imgmsg_to_cv2(msg)
        # print(cv_image)
        # cv2.imshow('image', cv_image)
        # cv2.waitKey(0)
        converted_image = PilImage.fromarray(numpy.uint8(cv_image), 'RGB')
        # converted_image.show('image')
        label_index, prediction, boxes = str(predict(converted_image))
        print(f'Result: {result}')

        # send back resized images
        if self.get_parameter('pub_image').value:
            processed_image = self.br.cv2_to_imgmsg(results.imgs[0])
            self.image_publisher.publish(processed_image)

        if self.get_parameter('pub_json').value:
            json = String()
            json.data = results.pandas().xyxy[0].to_json(orient="records")
            self.json_publisher.publish(json)

        # send back bounding
        if self.get_parameter('pub_boxes').value:
            detections = self.getDetectionArray(boxes.pandas().xyxy[0])
            self.detection_publisher.publish(detections)

        self.get_logger("received image")


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = RosIO()

    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
