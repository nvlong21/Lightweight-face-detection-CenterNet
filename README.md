# Center face
# Center face detection model
![img1](https://github.com/nvlong21/CenterFace/blob/master/test/centerface.jpg)
This model is a lightweight facedetection model designed for edge computing devices
## Tested the environment that works
- Ubuntu16.04、Ubuntu18.04、Windows 10（for inference）
- Python3.6
- Pytorch1.0.1
- CUDA10.0 + CUDNN7.6

## Accuracy, speed, model size comparison
Using the cleaned widerface labels provided by [Retinaface](https://github.com/deepinsight/insightface/blob/master/RetinaFace/README.md) 

### Widerface test
- Test accuracy in the WIDER FACE val set (single-scale input resolution: **640*640 or scaling by the maximum side length of 320**)

Model|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
libfacedetection v1（caffe）|0.65 |0.5       |0.233
libfacedetection v2（caffe）|0.714 |0.585       |0.306
Retinaface-Mobilenet-0.25 (Mxnet)   |0.745|0.553|0.232
CenterFace |--     |--       |--


- Test accuracy in the WIDER FACE val set (single-scale input resolution: **VGA 640*480 or scaling by the maximum side length of 640** )

Model|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
libfacedetection v1（caffe）|0.741 |0.683       |0.421
libfacedetection v2（caffe）|0.773 |0.718       |0.485
Retinaface-Mobilenet-0.25 (Mxnet)   |**0.879**|0.807|0.481
CenterFace |--     |--       |--



##  Reference
- [libfacedetection](https://github.com/ShiqiYu/libfacedetection/)
- [RFBNet](https://github.com/ruinmessi/RFBNet)
- [Retinaface](https://github.com/deepinsight/insightface/blob/master/RetinaFace/README.md)
- [CenterNet](https://github.com/xingyizhou/CenterNet)
- [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
