---
layout: default
title: Object Detection
parent: Applications and Demos
nav_order: 2
---

# **Object Detection**
{: .no_toc }

1. TOC
{:toc}
---

## **Overview**

Object detection is a computer technology related to computer vision and image
processing that deals with detecting instances of semantic objects of a certain
class (such as humans, buildings, or cars) in digital images and videos.
Well-researched domains of object detection include face detection and pedestrian
detection. Object detection has applications in many areas of computer vision,
including image retrieval and video surveillance [^1] .

## **Object Detection SSD**

### **Inference Engine and Algorithm**

![tfliteframework][tflite]

This demo uses:

 * TensorFlow Lite as an inference engine [^2] ;
 * Single Shot Detection as default algorithm [^3] .

More details on [eIQ™][eiq] page.

### **Running Object Detection SSD**

#### **Using Images for Inference**

##### **Default Image**

1. Run the _Object Detection_ demo using the following line:
```console
# pyeiq --run object_detection_tflite
```
  * This runs inference on a default image:
  ![detection][image_eIQObjectDetection]

##### **Custom Image**

1. Pass any image as an argument:
```console
# pyeiq --run object_detection_tflite --image=/path_to_the_image
```

#### **Using Video Source for Inference**

##### **Video File**

1. Run the _Object Detection_ using the following line:
```console
# pyeiq --run object_detection_tflite --video_src=/path_to_the_video
```
  * This runs inference on a video file:
  ![detection_video][video_eIQObjectDetection]

##### **Video Camera or Webcam**

1. Specify the camera device:
```console
# pyeiq --run object_detection_tflite --video_src=/dev/video<index>
```

### **Extra Parameters**

1. Use **--help** argument to check all the available configurations:
```console
# pyeiq --run object_detection_tflite --help
```

## **Object Detection DNN**

### **Inference Engine and Algorithm**

![opencvframework][opencv]

This demo uses:

 * OpenCV DNN as an inference engine [^4] ;
 * Deep Neural Networks as default algorithm [^5] .

More details on [eIQ™][eiq] page.

### **Running Object Detection DNN**

#### **Using Images for Inference**

##### **Default Image**

1. Run the _Object Detection_ demo using the following line:
```console
# pyeiq --run object_detection_dnn
```
  * This runs inference on a default image:
  ![image_dnn][image_eIQObjectDetectionDNN]

##### **Custom Image**

1. Pass any image as an argument:
```console
# pyeiq --run object_detection_dnn --image=/path_to_the_image
```

#### **Using Video Source for Inference**

##### **Video File**

1. Run the _Object Detection_ using the following line:
```console
# pyeiq --run object_detection_dnn --video_src=/path_to_the_video
```

##### **Video Camera or Webcam**

1. Specify the camera device:
```console
# pyeiq --run object_detection_dnn --video_src=/dev/video<index>
```

### **Extra Parameters**

1. Use **--help** argument to check all the available configurations:
```console
# pyeiq --run object_detection_dnn --help
```

## **Object Detection YOLOv3**

### **Inference Engine and Algorithm**

![tfliteframework][tflite]

This demo uses:

 * TensorFlow Lite as an inference engine [^2] ;
 * YOLOv3 as default algorithm [^6] .

More details on [eIQ™][eiq] page.

**NOTE:** This demo needs a quantized model to work properly.

### **Running Object Detection YOLOv3**

#### **Using Images for Inference**

##### **Default Image**

1. Run the _Object Detection_ demo using the following line:
```console
# pyeiq --run object_detection_yolov3
```
  * This runs inference on a default image:
  ![image_yolov3][image_eIQObjectDetectionYOLOv3]

##### **Custom Image**

1. Pass any image as an argument:
```console
# pyeiq --run object_detection_yolov3 --image=/path_to_the_image
```

#### **Using Video Source for Inference**

##### **Video File**

1. Run the _Object Detection_ using the following line:
```console
# pyeiq --run object_detection_yolov3 --video_src=/path_to_the_video
```

##### **Video Camera or Webcam**

1. Specify the camera device:
```console
# pyeiq --run object_detection_yolov3 --video_src=/dev/video<index>
```

### **Extra Parameters**

1. Use **--help** argument to check all the available configurations:
```console
# pyeiq --run object_detection_yolov3 --help
```

## **References**

[^1]: https://en.wikipedia.org/wiki/Object_detection
[^2]: https://www.tensorflow.org/lite
[^3]: https://arxiv.org/abs/1512.02325
[^4]: https://github.com/opencv/opencv/tree/master/samples/dnn
[^5]: https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html
[^6]: https://pjreddie.com/darknet/yolo/

[image_eIQObjectDetection]: ../media/demos/eIQObjectDetection/image_eiqobjectdetection_resized_logo.gif

[video_eIQObjectDetection]: ../media/demos/eIQObjectDetection/video_eIQObjectDetection_street.gif

[image_eIQObjectDetectionDNN]: ../media/demos/eIQObjectDetectionDNN/image_eiqobjectdetectiondnn_resized_logo.gif

[image_eIQObjectDetectionYOLOv3]: ../media/demos/eIQObjectDetectionYOLOV3/image_eiqobjectdetectionyolov3_resized_logo.gif

[tflite]: https://img.shields.io/badge/TFLite-2.1.0-orange
[opencv]: https://img.shields.io/badge/OpenCV-4.2.0-yellow
[eiq]: https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ
