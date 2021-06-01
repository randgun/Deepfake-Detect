from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from Xception_prc.network.models import model_selection
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from torch import nn
from collections import Counter
from MesoNet_prc import classifiers
import cv2
from Headpose_forensic.utils.face_proc import FaceProc
from Headpose_forensic.forensic_test import examine_a_frame
from Headpose_forensic.utils.head_pose_proc import PoseEstimator
import numpy as np
import argparse, pickle
from scipy.ndimage.interpolation import zoom

def detection(filepath, model,transform1):
    img = Image.open(filepath)
    # img = cv2.imread(filepath,1)
    # img = Image.fromarray(img)
    img = transform1(img)
    img = Variable(torch.unsqueeze(img, dim=0).cuda(), requires_grad=False)
    outputs = model(img)
    smax = nn.Softmax(1)
    return smax(outputs)[0][1].item()

def img_process(scale, face_inst, frame):
    # 左上(x0, y0)，右下(x1, y1)
    scale = (scale - 1) / 2
    face_loc = face_inst.get_all_face_rects(frame)[0]
    if face_loc == None:
        print("Don't detect the face")
        return 0
    else:
        x_offset = round(scale * (face_loc.right() - face_loc.left()))
        y_offset = round(scale * (face_loc.bottom() - face_loc.top()))
        x0 = max(face_loc.left() - x_offset, 0)
        y0 = max(face_loc.top() - y_offset, 0)
        x1 = min(face_loc.right() + x_offset, frame.shape[1])
        y1 = min(face_loc.bottom() + y_offset, frame.shape[0])

    face = frame[y0:y1, x0:x1]
    m, n = face.shape[:2]
    return zoom(face, (256 / m, 256 / n, 1))

def load_model():
    parser = argparse.ArgumentParser(description="headpose forensics")
    parser.add_argument('--input_dir', type=str, default='Videos')
    parser.add_argument('--markID_c', type=str, default='18-36,49,55',
                        help='landmark ids to estimate CENTRAL face region')
    parser.add_argument('--markID_a', type=str, default='1-36,49,55',
                        help='landmark ids to estimate WHOLE face region')
    parser.add_argument('--classifier_path', type=str, default=
    'Headpose_forensic/models/trained_models/trained_model.p')
    parser.add_argument('--save_file', type=str, default='proba_list.p')
    args = parser.parse_args()
    # 加载模型
    # initiate face process class, used to detect face and extract landmarks

    # initialize SVM classifier for face forensics
    with open(args.classifier_path, 'rb') as f:
        # 此处被修改，添加了 encoding = 'iso-8859-1'，否则无法加载模型
        model = pickle.load(f)
    return model, args

def roc(scores, y_true):
    Auc = []
    fpr = [1,1,1]
    tpr = [1,1,1]
    for i in range(3):
        x = np.array(scores[i])
        print(x)
        y = np.array(y_true[i])
        fpr[i], tpr[i], thresholds = roc_curve(y, x, drop_intermediate=False)
        Auc.append(auc(fpr[i], tpr[i]))
    return fpr, tpr, Auc



if __name__ == '__main__':
    scores = [[], [], []]
    y = [[], [], []]
    face_inst = FaceProc()

    with torch.no_grad():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        CUDA: 1
        model_path = './Xception_prc/pretrained_model/deepfake_c0_xception.pkl'
        transform1 = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
        model.eval()
        path = "C:/Users/25220/Desktop/Data/Xception_roc"  # 文件夹目录
        files = os.listdir(path)  # 得到文件夹下的所有文件名称
        for file in files:  # 遍历文件夹
            if file == 'Deepfake':
                mydir = path + '/'+file
                images = os.listdir(mydir)
                for image in images:
                    image = mydir + '/' + image
                    print("Detecting image: ", image)
                    score = detection(image, model, transform1)
                    scores[0].append(score)

            else:
                mydir = path + '/' + file
                images = os.listdir(mydir)
                for image in images:
                    image = mydir + '/' + image
                    print("Detecting image: ", image)
                    score = detection(image, model, transform1)
                    scores[0].append(score)
    
    
    model = classifiers.Meso4()
    model.load('MesoNet_prc/weights/weight_1.h5')
    path = "C:/Users/25220/Desktop/Data/Meso_roc"
    files = os.listdir(path)
    for file in files:  # 遍历文件夹
        if file == 'Deepfake':
            mydir = path + '/' + file
            images = os.listdir(mydir)
            for image in images:
                image = mydir + '/' + image
                print("Detecting image: ", image)
                img = cv2.imread(image, 1)
                face = img_process(1.3, face_inst, img)
                proba = 1 - model.predict(np.array([face]))[0]
                scores[1].append(proba[0])
        else:
            mydir = path + '/' + file
            images = os.listdir(mydir)
            for image in images:
                image = mydir + '/' + image
                print("Detecting image: ", image)
                img = cv2.imread(image, 1)
                face = img_process(1.3, face_inst, img)
                proba = 1 - model.predict(np.array([face]))[0]
                scores[1].append(proba[0])


    model, args = load_model()
    path = "C:/Users/25220/Desktop/Data/HeadPoses_roc"
    files = os.listdir(path)
    for file in files:  # 遍历文件夹
        if file == 'Deepfake':
            mydir = path + '/' + file
            images = os.listdir(mydir)
            for image in images:
                image = mydir + '/' + image
                print("Detecting image: ", image)
                img = cv2.imread(image, 1)
                height, width = img.shape[0:2]
                pose_estimator = PoseEstimator([height, width])
                face_loc = face_inst.get_all_face_rects(img)[0]
                all_landmarks = face_inst.get_landmarks_all_faces(img, face_loc)
                proba = examine_a_frame(args, img, face_loc, all_landmarks,
                                        model[0], model[1], pose_estimator)[0]
                scores[2].append(proba)
        else:
            mydir = path + '/' + file
            images = os.listdir(mydir)
            for image in images:
                image = mydir + '/' + image
                print("Detecting image: ", image)
                img = cv2.imread(image, 1)
                height, width = img.shape[0:2]
                pose_estimator = PoseEstimator([height, width])
                face_loc = face_inst.get_all_face_rects(img)[0]
                all_landmarks = face_inst.get_landmarks_all_faces(img, face_loc)
                proba = examine_a_frame(args, img, face_loc, all_landmarks,
                                        model[0], model[1], pose_estimator)[0]
                scores[2].append(proba)


    y[0] = [1 for i in range(335)]
    z = [0 for i in range(335)]
    y[0] = y[0] + z
    print(Counter(y[0]))

    y[1] = [1 for i in range(350)]
    z = [0 for i in range(350)]
    y[1] = y[1] + z
    print(Counter(y[1]))

    y[2] = [1 for i in range(326)]
    z = [0 for i in range(350)]
    y[2] = y[2] + z
    print(Counter(y[2]))

    fpr, tpr, Auc = roc(scores, y)
    print(Auc)
    plt.title("ROC")
    plt.plot(fpr[0], tpr[0], label="Xception(AUC = 0.9997)", color = "green")
    plt.plot(fpr[1], tpr[1], label="Meso-4(AUC = 0.8034)", color="red")
    plt.plot(fpr[2], tpr[2], label="HeadPoses(AUC = 0.9801)", color="blue")
    plt.plot((0, 1), (0, 1),  ls='--', c='k')
    plt.legend()
    plt.xlabel("false presitive rate")
    plt.ylabel("true presitive rate")
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)
    plt.show()

