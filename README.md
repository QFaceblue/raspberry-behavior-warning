# raspberry-behavior-warning
在树莓派4B上部署神经网络，检测摄像头视频流中驾驶员的驾驶行为并通过语音设备提示。
# 环境
raspberry pi 4B  
python 3.7  
onnxruntime 1.4  
opencv-python 4.4  
pyttsx3 2.9  
torch 1.6（非必须，使用yolo进行方向盘检测时需要）  
ffmpeg 4.1(非必须，使用ffmpeg实现推流)
# 介绍
该项目旨在利用树莓派检测驾驶员的驾驶行为并通过语音提示，提升驾驶安全。  
硬件设备：树莓派4B 8G内存  
使用轻量级神经网络（MobileNetv2等）对视频流中视频帧进行行为分类。  
使用北斗模块获取GPS信息（位置信息和车速）  
使用pyttsx3模块将文字提示转为语音提示，使用蓝牙音箱播放  
# 关键文件说明
detect_config.py 检测文件包含所有功能，根据配置文件设置检测。实现服务端连接、检测驾驶行为、语音提示、视频推流等。  
detect_draw.py 检测驾驶行为并将结果本地展示
detect_push.py 检测驾驶行为、推流
detect_push_draw.py 检测驾驶行为、展示、推流  
ffmpeg_push.py 使用ffmpeg推流  
predict.py 测试驾驶行为分类  
save_beidou.py 保存北斗行为信息  
test_ffmpeg.py 测试ffmpeg  
test_pyttsx.py 测试文字转语音  
test_util.py 测试工具，检测设备是否连接，如摄像头、北斗、神经加速棒等  
yolo.py 目标检测  
yolo_predict.py 测试方向盘检测
# 模型训练
参考项目[Driving-Behavior-Recognition](https://github.com/QFaceblue/Driving-Behavior-Recognition)
# 模型转换
将pth格式权重转化为通用onnx格式权重，可以使用onnxruntime推理引擎  
若想使用intel神经加速棒，需将onnx格式权重转为openvino指定格式，使用openvino推理引擎