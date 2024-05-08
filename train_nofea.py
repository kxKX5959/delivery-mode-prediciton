# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
from scipy.io import loadmat
from models_1 import ATCnet
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import tensorflow as tf
np.random.seed(3)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.set_printoptions(precision=4)
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Add, LSTM, Bidirectional
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, precision_recall_curve, \
    roc_curve, auc, brier_score_loss
from tensorflow import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.layers import BatchNormalization, Conv1D, AveragePooling1D, GlobalAveragePooling1D, Minimum, MaxPooling1D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten, Lambda, multiply
from tensorflow.keras.constraints import max_norm
from models_direct import fhrnet
import itertools
matplotlib.rc('font', family='simsun', weight='bold')  # 'FangSong'
import itertools
import scipy

confidence = 0.95  # Change to your desired confidence level
z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
dataPath = "F:\expert_agreement\第五章对比实验的特征提取\ours混合特征\数据集特征加标签\HUAYI_pH_data-784\segment_pre/"



def Processing(Data):
    SIZE1 = 1
    SIZE2 = Data.shape[1]

    # global data_max
    # global data_min

    data_max = 220
    data_min = 50
    # data_max = np.max(Data)
    # data_min = np.min(Data)
    Data = (Data - data_min) / (data_max - data_min)

    # train_x = Data.reshape([-1, SIZE1, SIZE2, 1])

    train_x = Data.reshape([-1, SIZE1, SIZE2])
    train_x = train_x.transpose([0, 2, 1])

    print('train_x', train_x.shape)
    # print('Scaling data ...-----------------\n')
    # for j in range(train_x.shape[0]):
    #     train_x[j, :, :] = scale(train_x[j, :, :], axis=0)
    return train_x


def Processing2(Data):
    SIZE1 = 1
    SIZE2 = Data.shape[1]

    # global data_max
    # global data_min

    data_max = np.max(Data)
    data_min = np.min(Data)
    # data_max = np.max(Data)
    # data_min = np.min(Data)
    Data = (Data - data_min) / (data_max - data_min)

    # train_x = Data.reshape([-1, SIZE1, SIZE2, 1])

    train_x = Data.reshape([-1, SIZE1, SIZE2])
    train_x = train_x.transpose([0, 2, 1])

    print('train_x', train_x.shape)
    # print('Scaling data ...-----------------\n')
    # for j in range(train_x.shape[0]):
    #     train_x[j, :, :] = scale(train_x[j, :, :], axis=0)
    return train_x


def loadData(dataath):
    FHR = []
    UC = []
    Label = []
    Feature = []
    with open('G:\华医数据临床信息\最终版/剖宫产+顺产-最新筛选.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'filename':
                continue
            filename = str(row[0]).split('.')[0]
            data = loadmat(dataPath + filename + '.mat')

            fhr = data['fhr']
            toco = data['toco']
            FHR.append(fhr)
            UC.append(toco)
            # print(row[3])row[3]为片段2标签，row[2]为片段1标签
            if row[143] == 'TRUE':
                label = 1
            else:
                label = 0
            Label.append(label)
            feature2 = np.array(row[119:123], dtype=float)  # 年龄、产次、孕次、孕周
            feature3 = np.array(row[220:221], dtype=float)  # BMI
            feature4 = np.array(row[223:228], dtype=float)  # 妊娠合并症
            feature5 = np.array(row[238:242], dtype=float)  # 超声信息
            # 假设第238列是包含'OA', 'OT', 'OP'的列
            # 假设第238列（Python中的索引是从0开始的，所以列索引是237）是包含'OA', 'OT', 'OP'的列
            if row[237] == 'OA':
                feature6 = np.array([1.0], ndmin=1)
            elif row[237] == 'OT':
                feature6 = np.array([2.0], ndmin=1)
            else:
                feature6 = np.array([3.0], ndmin=1)
            features = np.concatenate((feature2, feature3, feature4, feature5, feature6))

            Feature.append(features)
        FHR = np.squeeze(np.array(FHR).transpose(0, 2, 1), 2)
        TOCO = np.squeeze(np.array(UC).transpose(0, 2, 1), 2)
        Label = np.array(Label)
        Feature = np.array(Feature)
    return FHR, TOCO, Label, Feature


def lr_schedule(epoch):
    # 训练网络时学习率衰减方案
    lr = 0.01
    if epoch >= 15 and epoch < 90:
        # 15  90    10
        lr = 0.001
    if epoch >= 90:
        lr = 0.0001
    print('Learning rate: ', lr)
    return lr


def plot_sonfusion_matrix(cm, classes, normalize=False, title='混淆矩阵', cmap=plt.cm.Blues):
    plt.title(title, fontsize=10)  # 'Confusion matrix'
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    # plt.tight_layout()
    plt.ylabel('真实标签', fontsize=10)  # 'True label'
    plt.xlabel('预测标签', fontsize=10)  # 'Predict label'
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(shrink=.75).ax.tick_params(labelsize=10)
    plt.savefig('混合s-m.png', dpi=600)
    plt.show()


def BCE_Loss(y_true, y_pred):
    epsilon = 1e-7  # 避免log(0)出现
    N = len(y_true)
    loss = -1 / N * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    return loss


def bce_with_logits_loss(y_true, logits):
    y_true = tf.cast(y_true, dtype=tf.float32)  # 将y_true转换为float32类型
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits)
    return tf.reduce_mean(loss)
def EEGNet_v2(nb_classes, Samples=4800,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Samples, 1))
    # input2 = Input(shape=(15,))

    print("input shape", input1.shape, Samples, kernLength)
    ##################################################################
    block1 = Conv1D(F1, kernLength, padding='same',
                    input_shape=(Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)

    block1_1 = Conv1D(F1, 32, padding='same',
                    input_shape=(Samples, 1),
                    use_bias=False)(input1)
    block1_1 = BatchNormalization()(block1_1)




    block1 = Concatenate(axis=-1)([block1, block1_1])
    print(block1.shape)


    block1 = Conv1D(24, 32, use_bias=False, padding='same')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling1D(4)(block1)
    block1 = dropoutType(dropoutRate)(block1)


    block2 = Conv1D(24, 32, use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling1D(8)(block2)
    block2 = dropoutType(dropoutRate)(block2)


    block2 = Conv1D(24, 32, use_bias=False, padding='same')(block2)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling1D(16)(block2)
    block2 = dropoutType(dropoutRate)(block2)

    bi_block2 = Bidirectional(LSTM(12, return_sequences=True), merge_mode='concat')(block2)  # merge_mode='concat'
    block2 = Add()([block2, bi_block2])
    flatten = Flatten(name='flatten')(block2)


    # dense = Dense(32, name='dense_all', kernel_constraint=max_norm(0.25))(flatten)
    # dense = Activation('elu', name='elu')(dense)
    # dense = dropoutType(0.5)(dense)
    #
    # features = Concatenate(axis=-1)([dense, input2])
    # features = dropoutType(dropoutRate)(features)


    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(0.25))(flatten)
    # kernel_constraint=max_norm(norm_rate) 0.15
    sig = Activation('sigmoid', name='sigmoid')(dense)

    return Model(inputs=[input1], outputs=sig)
    # return Model(inputs=[input1], outputs=sig)

def log_loss(y_true, y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

def cross_entropy_loss(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
    return tf.reduce_mean(loss)


Data, Data2, Label, Feature = loadData(dataPath)
Data = Processing(Data)

Feature = Feature[:, :15]
# Data2 = Processing2(Data2)
cvsacc1 = []
cvssen1 = []
cvsspe1 = []
cvsacc = []
cvssen = []
cvsspe = []
all_acc, all_sen, all_spe, all_qi, all_pre, all_rec, all_f1, all_auc = [], [], [], [], [], [], [], []
index = 1
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
acc_list = []
Sensitivity_list = []
Specificity_list = []
Qi_list = []
TPR_list = []
TNR_list = []
BRIES_list = []
F1_list = []
AUC_list = []
flag = True

for train, test in kfold.split(Data, Label):
    x_res = Data[train]
    y_res = Label[train]
    # val_Data, val_Label
    val_Data = Data[test]
    val_Lable = Label[test]

    print("DATA[train]", x_res.shape)
    print("Label[train]", y_res.shape)
    eng_train = Feature[train]
    eng_test = Feature[test]
    s_scale = StandardScaler()
    eng_trainf = s_scale.fit_transform(eng_train)
    eng_testf = s_scale.transform(eng_test)
    model = EEGNet_v2(nb_classes=1, Samples=7200,
                      dropoutRate=0.25, kernLength=64, F1=8, D=2, F2=16,
                      dropoutType='Dropout')  # 16,32

    opt = optimizers.SGD(learning_rate=lr_schedule(0), momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    lr_scheduler = LearningRateScheduler(lr_schedule)
    model_name = 'net_lead_' + str(index) + '.hdf5'
    MODEL_PATH = './Model2/'
    # set a valid path for your system to record model checkpoints
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH + model_name, verbose=1,
                                                      monitor='loss', mode='min',
                                                      save_best_only=True)

    class_weights = {0: 0.25, 1: 0.75}
    fittedModel = model.fit([x_res,], y_res, batch_size=16, epochs=220,
                            verbose=2,
                            # validation_data=(val_Data, val_Lable),
                            callbacks=[lr_scheduler, checkpointer],
                            class_weight=class_weights)

    # evaluate the model
    model = load_model(MODEL_PATH + model_name)
    pred_vt = model.predict([Data[test],], batch_size=16, verbose=0)

    pred_v = np.round(pred_vt)
    true_v = Label[test]

    # 评估模型的性能 ---------------------------------------------------------------------------------------------
    Conf_Mat = confusion_matrix(true_v, pred_v)  # 利用专用函数得到混淆矩阵

    Acc1 = (Conf_Mat[0][0] + Conf_Mat[1][1]) / (np.sum(Conf_Mat[0]) + np.sum(Conf_Mat[1]))
    specificity1 = Conf_Mat[0][0] / np.sum(Conf_Mat[0])
    sensitivity1 = Conf_Mat[1][1] / np.sum(Conf_Mat[1])
    precision = precision_score(true_v, pred_v,)
    recall = recall_score(true_v, pred_v,)
    f1 = f1_score(true_v, pred_v,)
    brier = brier_score_loss(true_v, pred_v)
    QI = np.sqrt(specificity1 * sensitivity1)
    fpr, tpr, thresholds = roc_curve(true_v, pred_vt, drop_intermediate=False)
    AUC = auc(fpr, tpr)

    print('===============测试集交叉验证的结果=====================')
    print("TEST: acc = %0.15f, sen = %0.15f, spe = %0.15f, QI= %0.15f, F1 = %0.15f,AUC = %0.15f,BRIER = %0.15f" % (
        Acc1, sensitivity1, specificity1, QI, f1, AUC, brier))

    print('\nConfusion Matrix:\n')

    print(Conf_Mat)
    acc_list.append(Acc1)
    Sensitivity_list.append(sensitivity1)
    Specificity_list.append(specificity1)
    Qi_list.append(QI)
    F1_list.append(f1)
    BRIES_list.append(brier)
    AUC_list.append(AUC)
    if flag:
        #  pred_vt = model.predict([X_test], batch_size=16, verbose=0)
        predict_proba = pred_vt
        true = Label[test]
        flag = False
    else:
        predict_proba = np.concatenate((pred_vt, predict_proba), axis=0)
        true = np.concatenate((Label[test], true), axis=0)
    index = index + 1


print("=========5折平均结果============")
print(
    "TEST: acc = %0.15f, sen = %0.15f, spe = %0.15f, QI = %0.15f, F1 = %0.15f,AUC = %0.15f,BRIER = %0.15f" % (
        np.mean(acc_list), np.mean(Sensitivity_list), np.mean(Specificity_list), np.mean(Qi_list),
        np.mean(F1_list), np.mean(AUC_list), np.mean(BRIES_list)))

print("=========5折方差============")
print(
    "TEST: acc = %0.15f, sen = %0.15f, spe = %0.15f, QI = %0.15f, F1 = %0.15f,AUC = %0.15f,BRIER = %0.15f" % (
        np.std(acc_list), np.std(Sensitivity_list), np.std(Specificity_list), np.std(Qi_list),
        np.std(F1_list), np.std(AUC_list), np.std(BRIES_list)))
fpr, tpr, threshold = roc_curve(true, predict_proba, drop_intermediate=False)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6),)
plt.title('Test ROC')
plt.plot(fpr, tpr, 'lightblue', label='Val AUC-CNN-BiLSTM = %0.3f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print('======综合混淆矩阵的AUC = %0.15f' % (roc_auc))
print("tpr::::", )
print(', '.join(map(str, tpr)))
print("fpr::::::")
print(', '.join(map(str, fpr)))
print("true::::::")
print(', '.join(map(str, true)))
print('----------------------------------------------------')

