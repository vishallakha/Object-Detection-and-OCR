#OCR Training
import time
from pandas import DataFrame, unique
import json
import os
from PIL import Image
#import Image
from sys import exit
#from sklearn.cross_validation import train_test_split
#import keras
import tensorflow as tf
import logging
import os
from os.path import join
import json
import random
import itertools
import re
import datetime
#import cairocffi as cairo
#import editdistance
import numpy as np
from scipy import ndimage
#import pylab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, CuDNNGRU,CuDNNLSTM
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import GRU,LSTM
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import cv2
import h5py
from collections import Counter
from keras.models import load_model
from keras.models import model_from_json
python_log_path="ITM_auto_challan_streaming.log"

# Define the logger function
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create a file handler
logger.info('created directory {}'.format(python_log_path))
handler = logging.FileHandler(python_log_path)
handler.setLevel(logging.ERROR)
# create a logging format
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s [%(module)s,%(funcName)s,%(lineno)d] %(process)d ---: %(message)s')
handler.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(handler)


def image_slicer(image_name):
    im=Image.open(image_name)
    left=row["plateXvalue"]
    up=row["plateYvalue"]-12
    right=left+row["plateWidth"]+4
    down=up+row["plateHeight"]+16
    box=(left,up,right,down)
    sliced_image=im.crop(box)
    return sliced_image

def json_object_creator(row,tag):
    obj={}
    obj["description"]=row["number_plate"]
    obj["tags"]=[tag]
    obj["size"]={"height":row["plateHeight"], "width":row["plateWidth"]}
    obj["objects"]=[]
    return obj


def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def is_valid_str(s):
    for ch in s:
        if not ch in letters:
            return False
    return True


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)



def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret
    


letters=[u'0',
         u'1',
         u'2',
         u'3',
         u'4',
         u'5',
         u'6',
         u'7',
         u'8',
         u'9',
         u'A',
         u'B',
         u'C',
         u'D',
         u'E',
         u'F',
         u'G',
         u'H',
         u'I',
         u'J',
         u'K',
         u'L',
         u'M',
         u'N',
         u'O',
         u'P',
         u'Q',
         u'R',
         u'S',
         u'T',
         u'U',
         u'V',
         u'W',
         u'X',
         u'Y',
         u'Z']

def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def is_valid_str(s):
    for ch in s:
        if not ch in letters:
            return False
    return True


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def train(img_w, load=False):
    # Input Parameters
    img_h = 64

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 256
    rnn_size = 256

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)
        
    batch_size = 120
    downsample_factor = pool_size ** 2

        
    data_train = TextImageGenerator('train', 'train', img_w, img_h, batch_size, downsample_factor)
    data_train.build_data()
    data_val = TextImageGenerator('train', 'val', img_w, img_h, batch_size, downsample_factor)
    data_val.build_data()


    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(32, (3,3), padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(64, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
    inner = Conv2D(128, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv3')(inner)
    """inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max3')(inner)    
    inner = Conv2D(256, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv4')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max4')(inner)"""

    conv_to_rnn_dims = (32,2048)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    lstm1 = CuDNNGRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
    lstm1_b = CuDNNGRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
    lstm1_merged = add([lstm1, lstm1_b])
    lstm2 = CuDNNGRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    lstm2_b = CuDNNGRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)

    # transforms RNN output to character activations:
    inner = Dense(data_train.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([lstm2, lstm2_b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[data_train.max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    if load:
        model = load_model('./tmp_model.h5', compile=False)
    else:
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    
    if not load:
        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([input_data], [y_pred])

        model.fit_generator(generator=data_train.next_batch(), 
                            steps_per_epoch=data_train.n,
                            epochs=1, 
                            validation_data=data_val.next_batch(), 
                            validation_steps=data_val.n)

    return model
    
os.chdir("/home/smart/Desktop/itm_analytics/new_ocr/")

class TextImageGenerator:

    def __init__(self, 
                 dirpath,
                 tag,
                 img_w, img_h, 
                 batch_size, 
                 downsample_factor,
                 max_text_len=10):

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor

        img_dirpath = join(dirpath, 'img')
        ann_dirpath = join(dirpath, 'ann')
        self.samples = []
        for filename in os.listdir(img_dirpath):
            name, ext = os.path.splitext(filename)
            if ext in ['.png', '.jpg']:
                img_filepath = join(img_dirpath, filename)
                json_filepath = join(ann_dirpath, name + '.json')
                ann = json.load(open(json_filepath, 'r'))
                description = ann['description']
                tags = ann['tags']
                if tag not in tags:
                    continue
                if is_valid_str(description):
                    self.samples.append([img_filepath, description])

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0

    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):
            img = cv2.imread(img_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img /= 255
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :] = img
            self.texts.append(text)

    def get_output_size(self):
        return len(letters) + 1

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = text_to_labels(text)
                source_str.append(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                #'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)
print( "model training has started.")

model = train(128, load=False)

model.save_weights("model_synt_img_1.h5")
model_json=model.to_json()
with open("model_synt_img_1.json","w") as json_file:
    json_file.write(model_json)