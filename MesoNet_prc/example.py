import numpy as np
from classifiers import *
# from pipeline import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# MesoNet在训练网络时标注0为假脸，1为真脸，对视频检测时采取采样方式，处理样本帧得到prediction_face来判断整个视频是否真实

# 1 - Load the model and its pretrained weights
classifier = Meso4()

classifier.load('weights/Meso4_DF.h5')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'test_images',
        target_size=(256, 256),
        batch_size=4,
        class_mode='binary',
        save_to_dir=r'E:\Pycharm Projects\MesoNet_prc\train_result',
        subset='training')

# 3 - Predict
X, y = generator.next()

# 对输入图像X进行预测分类
print('Predicted :', classifier.predict(X), '\nReal class :', y)
'''
# 4 - Prediction for a video dataset

classifier.load('weights/Meso4_DF.h5')

predictions = compute_accuracy(classifier, 'test_videos')
for video_name in predictions:
    print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
'''