# A Detection and Positioning System on Face-Forged Video by AI
基于 Python 语言编写代码复现了模型框架，分别在 Celeb-DF 数据集上训练 HeadPoses 模型，在 FaceForensics++的 Deepfake 数据集训练 MesoNet 和 Xception 两个基于深度网络的伪脸检测模型，并在此基础上使用 PyQt5 库设计和实现了一个对用户友好的 AI 换脸视频检测定位系统，该系统可直观展示伪脸检测算法的帧定位效果和视频检测结果

#### Environment

- Windows 10
- tqdm 4.28.1
- numpy 1.16.2
- dlib 19.21.1
- opencv-python 4.1.1.26
- Pillow 6.2.0
- Sklearn 0.20.4
- Torch 1.7.1+cu110
