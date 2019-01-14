#! -*- coding: utf-8 -*-
# 基于序列标注的思路对对联。

import codecs
import numpy as np
import uniout
from keras.models import Model
from keras.layers import *
from keras.callbacks import Callback


min_count = 2
maxlen = 16
batch_size = 64
char_size = 128


def read_data(txtname):
    txt = codecs.open(txtname, encoding='utf-8').read()
    txt = txt.strip().split('\n')
    txt = [l.strip().split(' ') for l in txt]
    txt = [l for l in txt if len(l) <= maxlen] # 删除过长的对联
    return txt


x_train_txt = read_data('couplet/train/in.txt')
y_train_txt = read_data('couplet/train/out.txt')
x_test_txt = read_data('couplet/test/in.txt')
y_test_txt = read_data('couplet/test/out.txt')


chars = {}
for txt in [x_train_txt, y_train_txt, x_test_txt, y_test_txt]:
    for l in txt:
        for w in l:
            chars[w] = chars.get(w, 0) + 1


chars = {i:j for i,j in chars.items() if j >= min_count}
id2char = {i+1:j for i,j in enumerate(chars)}
char2id = {j:i for i,j in id2char.items()}


def string2id(s):
    # 0: <unk>
    return [char2id.get(c, 0) for c in s]

x_train = map(string2id, x_train_txt)
y_train = map(string2id, y_train_txt)
x_test = map(string2id, x_test_txt)
y_test = map(string2id, y_test_txt)


# 按字数分组存放
train_dict = {}
test_dict = {}

for i,x in enumerate(x_train):
    j = len(x)
    if j not in train_dict:
        train_dict[j] = [[], []]
    train_dict[j][0].append(x)
    train_dict[j][1].append(y_train[i])

for i,x in enumerate(x_test):
    j = len(x)
    if j not in test_dict:
        test_dict[j] = [[], []]
    test_dict[j][0].append(x)
    test_dict[j][1].append(y_test[i])

for j in train_dict:
    train_dict[j][0] = np.array(train_dict[j][0])
    train_dict[j][1] = np.array(train_dict[j][1])

for j in test_dict:
    test_dict[j][0] = np.array(test_dict[j][0])
    test_dict[j][1] = np.array(test_dict[j][1])


def data_generator(data):
    data_p = [float(len(i[0])) for i in data.values()]
    data_p = np.array(data_p) / sum(data_p)
    while True: # 随机选一个字数，然后随机选样本，生成字数一样的一个batch
        idx = np.random.choice(len(data_p), p=data_p) + 1
        size = min(batch_size, len(data[idx][0]))
        idxs = np.random.choice(len(data[idx][0]), size=size)
        np.random.shuffle(idxs)
        yield data[idx][0][idxs], np.expand_dims(data[idx][1][idxs], 2)


def gated_resnet(x, ksize=3):
    # 门卷积 + 残差
    x_dim = K.int_shape(x)[-1]
    xo = Conv1D(x_dim*2, ksize, padding='same')(x)
    return Lambda(lambda x: x[0] * K.sigmoid(x[1][..., :x_dim]) \
                            + x[1][..., x_dim:] * K.sigmoid(-x[1][..., :x_dim]))([x, xo])


x_in = Input(shape=(None,))
x = x_in
x = Embedding(len(chars)+1, char_size)(x)
x = Dropout(0.25)(x)

x = gated_resnet(x)
x = gated_resnet(x)
x = gated_resnet(x)
x = gated_resnet(x)
x = gated_resnet(x)
x = gated_resnet(x)

x = Dense(len(chars)+1, activation='softmax')(x)

model = Model(x_in, x)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam')


def couplet_match(s):
    # 输出对联
    # 先验知识：跟上联同一位置的字不能一样
    x = np.array([string2id(s)])
    y = model.predict(x)[0]
    for i,j in enumerate(x[0]):
        y[i, j] = 0.
    y = y[:, 1:].argmax(axis=1) + 1
    r = ''.join([id2char[i] for i in y])
    print u'上联：%s，下联：%s' % (s, r)
    return r


class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10
    def on_epoch_end(self, epoch, logs=None):
        # 训练过程中观察几个例子，显示对联质量提高的过程
        couplet_match(u'晚风摇树树还挺')
        couplet_match(u'今天天气不错')
        couplet_match(u'鱼跃此时海')
        couplet_match(u'只有香如故')
        # 保存最优结果
        if logs['val_loss'] <= self.lowest:
            self.lowest = logs['val_loss']
            model.save_weights('./best_model.weights')


evaluator = Evaluate()

model.fit_generator(data_generator(train_dict),
                    steps_per_epoch=1000,
                    epochs=100,
                    validation_data=data_generator(test_dict),
                    validation_steps=100,
                    callbacks=[evaluator])
