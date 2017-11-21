import pickle,pprint
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.misc import imread
import csv


def readPhoto(file_dir):
    #IMG = []
    len_num = 14
    IMG = np.ones((14, 512,512,3))
    for root, dirs, files in os.walk(file_dir):
        print(root)
        print(dirs)
        print(files)
        idx = 0
        for file in files:
            apple_path = os.path.join(file_dir, file)
            img = imread(apple_path).astype(np.float32)
            #print(type(img))
            img = img - np.mean(img)
            IMG[idx] = img
            idx = idx+1
            #print(img.size)
            #IMG.append(img)
            #np.append(IMG,img,axis=0)
    #IMG = np.ndarray(IMG)
    return IMG


ttttt = readPhoto('apple')
print(type(ttttt))
print(type(ttttt[1]))
#print(ttttt)
with open("apple_data.csv", "w") as csvFile:
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['features', ttttt])
    csvFile.close()


fffff = [['features',ttttt],['labels',np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1])]]

with open('data.pickle', 'wb') as f:
    # Pickle the 'features' dictionary using the highest protocol available.
    pickle.dump(fffff, f, pickle.HIGHEST_PROTOCOL)


# with tf.Session() as sess:
#     img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)  #decode the image to RGB
#     img_data_jpg_unit8 = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)  # change to unit8
#     print(type(image_raw_data_jpg))
#     print(type(img_data_jpg))
#     print(type(img_data_jpg_unit8))
