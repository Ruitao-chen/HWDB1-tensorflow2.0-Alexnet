from cProfile import label
from msilib.schema import Feature
import os
import numpy as np
import tensorflow as tf

def preprocess_image(image):
    image_size=28
    img_tensor = tf.image.decode_jpeg(image, channels=1)
    img_tensor = tf.image.resize(img_tensor, [image_size, image_size])
    img_tensor /= 255.0 # normalize to [0,1] range
    return img_tensor

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

def change_range(image,label):
    return 2*image-1, label

def get_HWDBdataset(dset):
    root_path = './DATA/HWDB1'
    BATCH_SIZE = 32
    images = []
    labels = []
    with open(os.path.join(root_path, dset+'.txt'), 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line is not '':
                stringArray = line.split('\\')
                imgpath = stringArray[0] + '/' + stringArray[1] + '/' + stringArray[2]
                label = stringArray[1]
                images.append(imgpath)
                labels.append(int(label))
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    image_label_ds = ds.map(load_and_preprocess_from_path_label)
    if dset == 'train':
        ds = image_label_ds.shuffle(buffer_size= len(images))
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        keras_ds = ds.map(change_range)
    else:
        ds = image_label_ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        keras_ds = ds.map(change_range)
    feature = []
    labels = []
    for index, lines in keras_ds:
        for i in index:
            feature.append(i)
        for j in lines:
            labels.append(j)
    feature = np.array(feature)
    labels = np.array(labels)
    return (feature, labels), int(np.ceil(len(images)/BATCH_SIZE))
