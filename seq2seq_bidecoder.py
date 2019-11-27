#! -*- coding: utf-8 -*-
# seq2seq，双向解码机制的Keras实现

import numpy as np
import pymongo
from tqdm import tqdm
import os,json
import uniout
import tensorflow as tf
import keras
from keras.layers import *
from keras_layer_normalization import LayerNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


min_count = 32
maxlen = 400
batch_size = 64
epochs = 100
char_size = 128
z_dim = 128
db = pymongo.MongoClient().text.thucnews # 我的数据存在mongodb中


if os.path.exists('seq2seq_config.json'):
    chars,id2char,char2id = json.load(open('seq2seq_config.json'))
    id2char = {int(i):j for i,j in id2char.items()}
else:
    chars = {}
    for a in tqdm(db.find()):
        for w in a['content']: # 纯文本，不用分词
            chars[w] = chars.get(w,0) + 1
        for w in a['title']: # 纯文本，不用分词
            chars[w] = chars.get(w,0) + 1
    chars = {i:j for i,j in chars.items() if j >= min_count}
    # 0: mask
    # 1: unk
    # 2: start
    # 3: end
    id2char = {i+4:j for i,j in enumerate(chars)}
    char2id = {j:i for i,j in id2char.items()}
    json.dump([chars,id2char,char2id], open('seq2seq_config.json', 'w'))


def str2id(s, start_end=False):
    # 文字转整数id
    if start_end: # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:maxlen-2]]
        ids = [2] + ids + [3]
    else: # 普通转化
        ids = [char2id.get(c, 1) for c in s[:maxlen]]
    return ids


def id2str(ids):
    # id转文字，找不到的用空字符代替
    return ''.join([id2char.get(i, '') for i in ids])


def padding(x):
    # padding至batch内的最大长度
    ml = max([len(i) for i in x])
    return [i + [0] * (ml-len(i)) for i in x]


def data_generator():
    # 数据生成器
    X, Y1, Y2 = [], [], []
    while True:
        for a in db.find():
            X.append(str2id(a['content']))
            Y1.append(str2id(a['title'], start_end=True))
            Y2.append(str2id(a['title'], start_end=True)[::-1])
            if len(X) == batch_size:
                X = np.array(padding(X))
                Y1 = np.array(padding(Y1))
                Y2 = np.array(padding(Y2))
                yield [X, Y1, Y2], None
                X, Y1, Y2 = [], [], []


class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        if not keras.__version__.startswith('2.3.'):
            for w in layer.trainable_weights:
                if w not in self._trainable_weights:
                    self._trainable_weights.append(w)
            for w in layer.non_trainable_weights:
                if w not in self._non_trainable_weights:
                    self._non_trainable_weights.append(w)
            for u in layer.updates:
                if not hasattr(self, '_updates'):
                    self._updates = []
                if u not in self._updates:
                    self._updates.append(u)
        return outputs


def to_one_hot(x):
    """输出一个词表大小的向量，来标记该词是否在文章出现过
    """
    x, x_mask = x
    x = K.cast(x, 'int32')
    x = K.one_hot(x, len(chars)+4)
    x = K.sum(x_mask * x, 1, keepdims=True)
    x = K.cast(K.greater(x, 0.5), 'float32')
    return x


class ScaleShift(Layer):
    """缩放平移变换层（Scale and shift）
    """
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    def build(self, input_shape):
        kernel_shape = (1,) * (len(input_shape)-1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')
    def call(self, inputs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs


class OurBidirectional(OurLayer):
    """自己封装双向RNN，允许传入mask，保证对齐
    """
    def __init__(self, layer, **args):
        super(OurBidirectional, self).__init__(**args)
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
    def reverse_sequence(self, x, mask):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return tf.reverse_sequence(x, seq_len, seq_dim=1)
    def call(self, inputs):
        x, mask = inputs
        x_forward = self.reuse(self.forward_layer, x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.reuse(self.backward_layer, x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], -1)
        if K.ndim(x) == 3:
            return x * mask
        else:
            return x
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.forward_layer.units * 2,)


def seq_avgpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做avgpooling。
    """
    seq, mask = x
    return K.sum(seq * mask, 1) / (K.sum(mask, 1) + 1e-6)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


class SelfModulatedLayerNormalization(OurLayer):
    """模仿Self-Modulated Batch Normalization，
    只不过降Batch Normalization改为Layer Normalization
    """
    def __init__(self, num_hidden, **kwargs):
        super(SelfModulatedLayerNormalization, self).__init__(**kwargs)
        self.num_hidden = num_hidden
    def build(self, input_shape):
        super(SelfModulatedLayerNormalization, self).build(input_shape)
        output_dim = input_shape[0][-1]
        self.layernorm = LayerNormalization(center=False, scale=False)
        self.beta_dense_1 = Dense(self.num_hidden, activation='relu')
        self.beta_dense_2 = Dense(output_dim)
        self.gamma_dense_1 = Dense(self.num_hidden, activation='relu')
        self.gamma_dense_2 = Dense(output_dim)
    def call(self, inputs):
        inputs, cond = inputs
        inputs = self.reuse(self.layernorm, inputs)
        beta = self.reuse(self.beta_dense_1, cond)
        beta = self.reuse(self.beta_dense_2, beta)
        gamma = self.reuse(self.gamma_dense_1, cond)
        gamma = self.reuse(self.gamma_dense_2, gamma)
        for _ in range(K.ndim(inputs) - K.ndim(cond)):
            beta = K.expand_dims(beta, 1)
            gamma = K.expand_dims(gamma, 1)
        return inputs * (gamma + 1) + beta
    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Attention(OurLayer):
    """多头注意力机制
    """
    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)
    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10
    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = tf.einsum('ijkl,ijml->ijkm', qw, kw) / self.key_size**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        if self.mask_right:
            ones = K.ones_like(a[:1, :1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
            a = a - mask
        a = K.softmax(a)
        # 完成输出
        o = tf.einsum('ijkl,ijlm->ijkm', a, vw)
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


# 搭建seq2seq模型

x_in = Input(shape=(None,))
yl_in = Input(shape=(None,))
yr_in = Input(shape=(None,))
x, yl, yr = x_in, yl_in, yr_in

x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(yl)

x_one_hot = Lambda(to_one_hot)([x, x_mask])
x_prior = ScaleShift()(x_one_hot) # 学习输出的先验分布（标题的字词很可能在文章出现过）

embedding = Embedding(len(chars)+4, char_size)
x = embedding(x)

# encoder，双层双向LSTM
x = LayerNormalization()(x)
x = OurBidirectional(CuDNNLSTM(z_dim // 2, return_sequences=True))([x, x_mask])
x = LayerNormalization()(x)
x = OurBidirectional(CuDNNLSTM(z_dim // 2, return_sequences=True))([x, x_mask])
x_max = Lambda(seq_maxpool)([x, x_mask])

# 正向decoder，单向LSTM
y = embedding(yl)
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])
y = CuDNNLSTM(z_dim, return_sequences=True)(y)
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])
y = CuDNNLSTM(z_dim, return_sequences=True)(y)
yl = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])

# 逆向decoder，单向LSTM
y = embedding(yr)
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])
y = CuDNNLSTM(z_dim, return_sequences=True)(y)
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])
y = CuDNNLSTM(z_dim, return_sequences=True)(y)
yr = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])

# 对齐attention + 检索attention
yl_ = Attention(8, 16, mask_right=True)([yl, yr, yr])
ylx = Attention(8, 16)([yl, x, x, x_mask])
yl = Concatenate()([yl, yl_, ylx])
# 对齐attention + 检索attention
yr_ = Attention(8, 16, mask_right=True)([yr, yl, yl])
yrx = Attention(8, 16)([yr, x, x, x_mask])
yr = Concatenate()([yr, yr_, yrx])

# 最后的输出分类（左右共享权重）
classifier = Dense(len(chars)+4)

yl = Dense(char_size)(yl)
yl = LeakyReLU(0.2)(yl)
yl = classifier(yl)
yl = Lambda(lambda x: (x[0]+x[1])/2)([yl, x_prior]) # 与先验结果平均
yl = Activation('softmax')(yl)

yr = Dense(char_size)(yr)
yr = LeakyReLU(0.2)(yr)
yr = classifier(yr)
yr = Lambda(lambda x: (x[0]+x[1])/2)([yr, x_prior]) # 与先验结果平均
yr = Activation('softmax')(yr)

# 交叉熵作为loss，但mask掉padding部分
cross_entropy_1 = K.sparse_categorical_crossentropy(yl_in[:, 1:], yl[:, :-1])
cross_entropy_1 = K.sum(cross_entropy_1 * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])
cross_entropy_2 = K.sparse_categorical_crossentropy(yr_in[:, 1:], yr[:, :-1])
cross_entropy_2 = K.sum(cross_entropy_2 * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])
cross_entropy = (cross_entropy_1 + cross_entropy_2) / 2

model = Model([x_in, yl_in, yr_in], [yl, yr])
model.add_loss(cross_entropy)
model.compile(optimizer=Adam(1e-3))


def gen_sent(s, topk=3, maxlen=64):
    """双向beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    xid = np.array([str2id(s)] * topk**2) # 输入转id
    yl_id = np.array([[2]] * topk) # L2R解码均以<start>开头，这里<start>的id为2
    yr_id = np.array([[3]] * topk) # R2L解码均以<end>开头，这里<end>的id为3
    l_scores, r_scores = [0] * topk, [0] * topk # 候选答案分数
    l_order, r_order = [], [] # 组合顺序
    for i in range(topk):
        for j in range(topk):
            l_order.append(i)
            r_order.append(j)
    for i in range(maxlen): # 强制要求输出不超过maxlen字
        l_proba, r_proba = model.predict([xid, yl_id[l_order], yr_id[r_order]]) # 计算左右解码概率
        l_proba = l_proba[:, i, 3:] # 直接忽略<padding>、<unk>、<start>
        r_proba = np.concatenate([r_proba[:, i, 2: 3], r_proba[:, i, 4:]], 1) # 直接忽略<padding>、<unk>、<end>
        l_proba = l_proba.reshape((topk, topk, -1)).mean(1) # 对所有候选R2L序列求平均，得到当前L2R方向的预测结果
        r_proba = r_proba.reshape((topk, topk, -1)).mean(0) # 对所有候选L2R序列求平均，得到当前R2L方向的预测结果
        l_log_proba = np.log(l_proba + 1e-6) # 取对数方便计算
        r_log_proba = np.log(r_proba + 1e-6) # 取对数方便计算
        l_arg_topk = l_log_proba.argsort(axis=1)[:, -topk:] # 每一项选出topk
        r_arg_topk = r_log_proba.argsort(axis=1)[:, -topk:] # 每一项选出topk
        _yl_id, _yr_id = [], [] # 暂存的候选目标序列
        _l_scores, _r_scores = [], [] # 暂存的候选目标序列得分
        if i == 0:
            for j in range(topk):
                _yl_id.append(list(yl_id[j]) + [l_arg_topk[0][j]+3])
                _l_scores.append(l_log_proba[0][l_arg_topk[0][j]])
                _yr_id.append(list(yr_id[j]) + [r_arg_topk[0][j]+3])
                _r_scores.append(r_log_proba[0][r_arg_topk[0][j]])
        else:
            for j in range(topk):
                for k in range(topk): # 遍历topk*topk的组合
                    _yl_id.append(list(yl_id[j]) + [l_arg_topk[j][k]+3])
                    _l_scores.append(l_scores[j] + l_log_proba[j][l_arg_topk[j][k]])
                    _yr_id.append(list(yr_id[j]) + [r_arg_topk[j][k]+3])
                    _r_scores.append(r_scores[j] + r_log_proba[j][r_arg_topk[j][k]])
            _l_arg_topk = np.argsort(_l_scores)[-topk:] # 从中选出新的topk
            _r_arg_topk = np.argsort(_r_scores)[-topk:] # 从中选出新的topk
            _yl_id = [_yl_id[k] for k in _l_arg_topk]
            _l_scores = [_l_scores[k] for k in _l_arg_topk]
            _yr_id = [_yr_id[k] for k in _r_arg_topk]
            _r_scores = [_r_scores[k] for k in _r_arg_topk]
        yl_id = np.array(_yl_id)
        yr_id = np.array(_yr_id)
        l_scores = np.array(_l_scores)
        r_scores = np.array(_r_scores)
        l_best_one = l_scores.argmax()
        r_best_one = r_scores.argmax()
        if yl_id[l_best_one][-1] == 3 and l_scores[l_best_one] >= r_scores[r_best_one]:
            return id2str(yl_id[l_best_one])
        if yr_id[r_best_one][-1] == 3 and r_scores[r_best_one] >= l_scores[l_best_one]:
            return id2str(yr_id[r_best_one][::-1)
    # 如果maxlen字都找不到<end>，直接返回
    l_best_one = l_scores.argmax()
    r_best_one = r_scores.argmax()
    if l_scores[l_best_one] >= r_scores[r_best_one]:
        return id2str(yl_id[l_best_one])
    else:
        return id2str(yr_id[r_best_one][::-1])


s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医 。'
s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'

class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10
    def on_epoch_end(self, epoch, logs=None):
        # 训练过程中观察一两个例子，显示标题质量提高的过程
        print gen_sent(s1)
        print gen_sent(s2)
        # 保存最优结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')


evaluator = Evaluate()

model.fit_generator(data_generator(),
                    steps_per_epoch=1000,
                    epochs=epochs,
                    callbacks=[evaluator])
