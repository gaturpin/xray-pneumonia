import os
import numpy as np
import cv2
import tensorflow as tf
import json

def LoadData(data_dir, labels_list_str, img_size = 150):
    """Loads xray image data
    Default img_size = 150
    Returns: numpy arrays of images and labels"""
    images = []
    labels = []
    
    for label_name in labels_list_str:
        label = 0 if label_name == 'normal' else 1
        image_path = os.path.join(data_dir, label_name)
        print(image_path)
        
        for file_name in os.listdir(image_path):
            if file_name.endswith('.jpeg'):
                try:
                    full_path = os.path.join(image_path, file_name)
                    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (img_size, img_size))
                    img = np.expand_dims(img, axis=-1) 
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(e)
    return np.array(images), np.array(labels)

def OpenHistory(filename):
    """Opens history json file"""
    with open(filename, 'r') as f:
        history = json.load(f)
    return history


def yPred(predictions, thresh = 0.5):
    """Calculates predicitions
    If the probability of 1 is > thresh, predicts 1
    Default: thresh = 0.5"""
    return (predictions > thresh).flatten().astype(int)

def Accuracy(predictions, y_test, thresh = 0.5):
    """Calculates accuracy from predicitions"""
    y_pred = yPred(predictions, thresh)
    accuracy = np.mean(y_pred == y_test)
    return accuracy

#Using tensorflow:
def AccuracyTF(predictions, y_test):
    """Calculates accuracy using tensorflow"""
    y_pred = tf.cast(predictions > 0.5, tf.int32)
    y_pred = tf.reshape(y_pred, [-1]) #Reshape to use with y_test
    tf_accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_test), tf.float32))
    return tf_accuracy

def MetricsCalc(y_pred, y_test):
    """Takes 1D y_pred, y_test
    Returns: (precision, recall, f1 score, false positive rate) as tuple"""

    true_pos = sum((y_test == 1) & (y_pred == 1))
    true_neg = sum((y_test == 0) & (y_pred == 0))
    false_pos = sum((y_test == 0) & (y_pred == 1))
    false_neg = sum((y_test == 1) & (y_pred == 0))
    
    precision = true_pos/(true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos/(true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
    fpr = false_pos/(false_pos + true_neg) if (false_pos + true_neg) > 0 else 0
    return (precision, recall, f1, fpr)

    
