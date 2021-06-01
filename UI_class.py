import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import numpy as np
import argparse, pickle
import cv2, face_recognition
from Headpose_forensic.utils.face_proc import FaceProc
from Headpose_forensic.forensic_test import examine_a_frame
from Headpose_forensic.utils.head_pose_proc import PoseEstimator
from scipy.ndimage.interpolation import zoom

from MesoNet_prc import classifiers
import time

import torch
from Xception_prc.network.models import model_selection
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from torch import nn



class MyUI(QWidget):

    def __init__(self):
        super(MyUI, self).__init__()
        self.frame = []          # 存图片
        self.face_loc = []       # 脸部位置 CSS(top, right, bottom, left)
        self.detectFlag = False  # 检测flag
        self.cap = []
        self.timer_camera = QTimer()  # 定义定时器
        self.mode = ''                # 算法
        self.face = []                # 存剪裁后的人脸
        self.target_size = 256
        self.count = 0
        # 以下是一些测试指标
        # threshold = 0.5      # 阈值
        # real = 0             # 视频内检测为真脸的帧数
        # fake = 0             # 假脸帧数
        # accurate = 0         # 准确率 fake/(fake+real)
        # speed = 0            # 平均检测速率
        self.result = {'threshold': 0.5, 'real': 0, 'fake': 0, 'accurate': 0, 'speed': 0}

        # 外框
        self.resize(1000, 800)
        self.setWindowTitle("Deepfake Detection")

        # 文本框
        self.textEdit = QTextEdit(self)
        self.textEdit.setGeometry(QtCore.QRect(30, 645, 940, 145))
        self.textEdit.setObjectName("textEdit")
        # self.textEdit.setFont(font)
        # 游标移动到末端
        self.textEdit.moveCursor(QTextCursor.End)

        # 图片label
        self.label = QLabel(self)
        self.label.setText("Waiting for video...")
        self.label.setFixedSize(940, 530)  # width height
        self.label.move(30, 60)
        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(100,100,100);font-size:15px;font-weight:bold;font-family:宋体;}"
                                 )

        # 修饰label
        self.label_num = QLabel(self)
        font = QFont()
        font.setPointSize(12)
        self.label_num.setText("Waiting for detecting...")
        self.label_num.setFont(font)
        self.label_num.setFixedSize(300, 30)  # width height
        self.label_num.move(330, 20)
        self.label_num.setStyleSheet("QLabel{background:yellow;}")

        # 开启视频按键
        self.btn = QPushButton(self)
        self.btn.setText("Open")
        self.btn.move(120, 600)
        self.btn.clicked.connect(self.slotOpen)

        # 检测按键
        self.btn_detect = QPushButton(self)
        self.btn_detect.setText("Detect")
        self.btn_detect.move(520, 600)
        self.btn_detect.setStyleSheet("QPushButton{background:red;}")  # 没检测红色，检测绿色
        self.btn_detect.clicked.connect(self.detection)

        # 关闭视频按钮
        self.btn_stop = QPushButton(self)
        self.btn_stop.setText("Stop")
        self.btn_stop.move(720, 600)
        self.btn_stop.clicked.connect(self.slotStop)

        # 检测算法选择列表框
        self.cb = QComboBox(self)
        self.cb.move(320, 600)
        self.cb.addItems(['','Headpose', 'MesoNet', 'Xception'])
        self.cb.resize(120,35)
        self.cb.currentIndexChanged.connect(self.selectAL)

    # 选择伪脸识别算法
    def selectAL(self):
        self.label.adjustSize()
        self.mode =  self.cb.currentText()
        self.face_inst = FaceProc()
        if (self.mode == 'Headpose'):
            # 构造参数
            parser = argparse.ArgumentParser(description="headpose forensics")
            parser.add_argument('--input_dir', type=str, default='Videos')
            parser.add_argument('--markID_c', type=str, default='18-36,49,55',
                                help='landmark ids to estimate CENTRAL face region')
            parser.add_argument('--markID_a', type=str, default='1-36,49,55',
                                help='landmark ids to estimate WHOLE face region')
            parser.add_argument('--classifier_path', type=str, default=
            'Headpose_forensic/models/trained_models/trained_model.p')
            parser.add_argument('--save_file', type=str, default='proba_list.p')
            self.args = parser.parse_args()
            # 加载模型
            # initiate face process class, used to detect face and extract landmarks
            height, width = self.frame.shape[0:2]
            self.pose_estimator = PoseEstimator([height, width])

            # initialize SVM classifier for face forensics
            with open(self.args.classifier_path, 'rb') as f:
                # 此处被修改，添加了 encoding = 'iso-8859-1'，否则无法加载模型
                self.model = pickle.load(f)

        elif (self.mode == 'MesoNet'):
            self.model = classifiers.Meso4()
            self.model.load('MesoNet_prc/weights/weight_1.h5')
            '''
            # 为设定好的网络架构加载权重
            self.model = classifiers.MesoInception4()
            self.model.load('MesoNet_prc/weights/MesoInception_DF.h5')
            '''
        elif (self.mode == 'Xception'):
            model_path = 'Xception_prc/pretrained_model/deepfake_c0_xception.pkl'
            self.transform1 = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
            ])
            self.model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
            self.model.load_state_dict(torch.load(model_path))
            self.model = self.model.cuda()
            self.model.eval()
        else:
            pass

    def slotOpen(self):
        '''Slot function to start the progamme
        '''
        videoName, _ = QFileDialog.getOpenFileName(self, "Open", "./Videos", "*.mp4;;*.avi;;All Files(*)")
        if videoName != "":
            self.cap = cv2.VideoCapture(videoName)
            # 设置定时器间隔ms
            self.timer_camera.start(1)
            self.timer_camera.timeout.connect(self.displayFrame)

    def slotStop(self):
        '''Slot function to stop the programme
        '''

        if self.cap != []:
            self.detectFlag = False
            self.cap.release()
            self.timer_camera.stop()  # 停止计时器
            self.label.setText("This video has been stopped.")
            self.label.setStyleSheet("QLabel{background:white;}"
                                     "QLabel{color:rgb(100,100,100);font-size:15px;"
                                     "font-weight:bold;font-family:宋体;}"
                                     )

            self.textEdit.clear()
            # 部分计数值重新初始化
            self.result = {'threshold': 0.5, 'real': 0, 'fake': 0, 'accurate': 0, 'speed': 0}
            self.count = 0
        else:
            self.label_num.setText("Push the left upper corner button to Quit.")
            Warming = QMessageBox.warning(self, "Warming", "Push the left upper corner button to Quit.",
                                          QMessageBox.Yes)

    def displayFrame(self):
        """ Slot function to display frame on the mainWindow
        """
        if (self.cap.isOpened()):
            ret, self.frame = self.cap.read()
            if ret:
                if(self.detectFlag and self.mode != ''):
                    if self.mode == 'Headpose':
                        time_start = time.time()
                        self.Headpose_forenic()
                        time_end = time.time()
                        self.result['speed'] += time_end - time_start
                    elif self.mode == 'MesoNet':
                        time_start = time.time()
                        self.MesoNet()
                        time_end = time.time()
                        self.result['speed'] += time_end - time_start
                        # print(time_end - time_start)
                    elif self.mode == 'Xception':
                        time_start = time.time()
                        self.Xception()
                        time_end = time.time()
                        self.result['speed'] += time_end - time_start
                        # print(time_end - time_start)
                    else:
                        pass
                frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QImage(frame.data, width, height, bytesPerLine,
                                 QImage.Format_RGB888).scaled(self.label.width(), self.label.height())
                self.label.setPixmap(QPixmap.fromImage(q_image))

            # 视频播放完毕
            else:
                self.cap.release()
                self.timer_camera.stop()  # 停止计时器
                self.textEdit.append("The number of real face: %5d" % (self.result['real']))
                self.textEdit.append("The number of fake face: %5d" % (self.result['fake']))
                self.textEdit.append(
                    "The accuracy: %7.3f" % (self.result['fake'] / (self.result['fake'] + self.result['real'])))
                self.textEdit.append("The average speed: %7.3f s/frame" % (self.result['speed'] / self.count))


    def detection(self):
        self.detectFlag = True

    def rect_to_ltrb(self):
        left = self.face_loc.left()
        top = self.face_loc.top()
        right = self.face_loc.right()
        bottom = self.face_loc.bottom()
        return (left, top, right, bottom)

    def img_process(self, scale):
        # 左上(x0, y0)，右下(x1, y1)
        scale = (scale - 1) / 2
        self.face_loc = self.face_inst.get_all_face_rects(self.frame)[0]
        if self.face_loc == None:
            print("Don't detect the face")
            return 0
        else:
            x_offset = round(scale * (self.face_loc.right() - self.face_loc.left()))
            y_offset = round(scale * (self.face_loc.bottom() - self.face_loc.top()))
            x0 = max(self.face_loc.left() - x_offset, 0)
            y0 = max(self.face_loc.top() - y_offset, 0)
            x1 = min(self.face_loc.right() + x_offset, self.frame.shape[1])
            y1 = min(self.face_loc.bottom() + y_offset, self.frame.shape[0])

        self.face = self.frame[y0:y1, x0:x1]
        if self.mode == 'MesoNet':
            m, n = self.face.shape[:2]
            self.face = zoom(self.face, (self.target_size / m, self.target_size / n, 1))
            # cv2.imshow('awd',self.face)

    def MesoNet(self):
        if self.img_process(1.3) != 0:
            proba = self.model.predict(np.array([self.face]))
            (left, top, right, bottom) = self.rect_to_ltrb()
            if proba[0] > self.result['threshold']:
                color = (0, 255, 0)
                self.result['real'] += 1
            else:
                color = (0, 0, 255)
                self.result['fake'] += 1
            # color = (0, 255, 0) if proba[0] > self.result['threshold'] else (0, 0, 255)
            cv2.rectangle(self.frame, (left, top), (right, bottom), color, 2)
            cv2.putText(
                img=self.frame,
                text='%.4f' % proba[0],
                org=(left, top),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(255, 0, 0),
                thickness=2
            )
            self.count += 1



    def Headpose_forenic(self):
        # proba_list = []
        self.img_process(1.3)
        all_landmarks = self.face_inst.get_landmarks_all_faces(self.frame, self.face_loc)
        proba = examine_a_frame(self.args, self.frame, self.face_loc, all_landmarks,
                                self.model[0], self.model[1], self.pose_estimator)
        # print(proba)
        # self.face_loc = face_recognition.face_locations(self.frame)[0] # 只考虑视频中存在一张人脸的情况
        # self.face_loc = proba[1][0]
        # color = (0,255,0) if proba[0] > self.result['threshold'] else (0, 0, 255)
        if  (1 - proba[0]) > self.result['threshold']:
            color = (0, 255, 0)
            self.result['real'] += 1
        else:
            color = (0, 0, 255)
            self.result['fake'] += 1
        (left, top, right, bottom) = self.rect_to_ltrb()
        cv2.rectangle(self.frame, (left, top),(right, bottom), color, 2)
        cv2.putText(
            img=self.frame,
            text='%.4f' %  (1 - proba[0]),
            org=(left, top),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(255, 0, 0),
            thickness=2
        )
        self.count += 1

        '''
        print('fake_proba: {},   optout: {}'.format(str(proba), optout))
        tmp_dict = dict()
        tmp_dict['file_name'] = f_name
        tmp_dict['probability'] = proba
        proba_list.append(tmp_dict)
        pickle.dump(proba_list, open(args.save_file, 'wb'))
        '''

    # 在Xception里分类为1是假，0是真，如[0.0943, 0.9057],[1]为假，[9.9984e-01, 1.6108e-04]为真
    def Xception(self):
        self.face_loc = self.face_inst.get_all_face_rects(self.frame)[0]
        if self.face_loc == None:
            print("Don't detect the face")
            return 0
        else:
            PIL_image = Image.fromarray(self.frame)
            self.face = self.transform1(PIL_image)
            self.face = Variable(torch.unsqueeze(self.face, dim=0).cuda(), requires_grad=False)
            outputs = self.model(self.face)
            _, preds = torch.max(outputs.data, 1)
            # print(preds)
            smax = nn.Softmax(1)
            proba = smax(outputs)[0]

            (left, top, right, bottom) = self.rect_to_ltrb()
            if proba[0] > self.result['threshold']:
                color = (0, 255, 0)
                self.result['real'] += 1
            else:
                color = (0, 0, 255)
                self.result['fake'] += 1
            # color = (0, 255, 0) if proba[0] > self.result['threshold'] else (0, 0, 255)
            cv2.rectangle(self.frame, (left, top), (right, bottom), color, 2)
            cv2.putText(
                img=self.frame,
                text='%.4f' % proba[0],
                org=(left, top),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(255, 0, 0),
                thickness=2
            )
            self.count += 1