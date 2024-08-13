import os
import numpy as np
import cv2

def LoadData(data_dir, labels_list_str, img_size = 150):
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