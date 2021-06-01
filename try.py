# 第一步，import
import tensorflow as tf  # 导入模块
from sklearn import datasets  # 从sklearn中导入数据集
import numpy as np  # 导入科学计算模块


# 第二步，train, test
x_train = datasets.load_iris().data  # 导入iris数据集的输入

y_train = datasets.load_iris().target  # 导入iris数据集的标签

np.random.seed(120)  # 设置随机种子，让每次结果都一样，方便对照

np.random.shuffle(x_train)  # 使用shuffle()方法，让输入x_train乱序

np.random.seed(120)  # 设置随机种子，让每次结果都一样，方便对照

np.random.shuffle(y_train)  # 使用shuffle()方法，让输入y_train乱序

tf.set_random_seed(120)  # 让tensorflow中的种子数设置为120

# 第三步，models.Sequential()
model = tf.keras.models.Sequential([  # 使用models.Sequential()来搭建神经网络
    tf.keras.layers.Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2())
    # 全连接层，三个神经元，激活函数为softmax,使用l2正则化
])

# 第四步，model.compile()
model.compile(  # 使用model.compile()方法来配置训练方法
    optimizer=tf.keras.optimizers.SGD(lr=0.1),  # 使用SGD优化器，学习率为0.1
    loss="sparse_categorical_crossentropy",  # 配置损失函数
    metrics=['sparse_categorical_accuracy']  # 标注网络评价指标
)

# 第五步，model.fit()
model.fit(  # 使用model.fit()方法来执行训练过程，
    x_train, y_train,  # 告知训练集的输入以及标签，
    batch_size=32,  # 每一批batch的大小为32，
    epochs=500,  # 迭代次数epochs为500
    validation_split=0.2  # 从测试集中划分80%给训练集
    # validation_freq=20  # 测试的间隔次数为20
)

# 第六步，model.summary()
model.summary()  # 打印神经网络结构，统计参数数目

model.save('my_model.h5')
del model