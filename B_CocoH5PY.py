import h5py
import numpy as np
import os
import cv2
import PIL.Image as Image

# d1 = np.random.random(size = (1000,20))
# d2 = np.random.random(size = (1000,200))
#
#
# hf = h5py.File('./TestData.h5', 'w')
#
# hf.create_dataset('dataset_1', data=d1)
# hf.create_dataset('dataset_2', data=d2)
# hf.close()
#
# hfR = h5py.File("./TestData.h5","r")
# for k,v in hfR.items():
#     print(k)
#     print(np.array(v))

coco_train_path = "E:\COCO\\train2014\\train2014"
hf = h5py.File('./cocoTrain.h5py', 'w')
d2 = np.random.random(size = (1000,200))
files = os.listdir(coco_train_path)
k = 0
print(len(files))
for file in files:
    path = os.path.join(coco_train_path,file)
    if k % 1000 == 0:
        print(file)
        print(k)
    img2 = np.array(cv2.imread(path))
    hf.create_dataset(file, data=img2)
    k += 1
hf.close()
hfR = h5py.File("./cocoTrain.h5py","r")
for k,v in hfR.items():
    print(k)
    print(np.array(v).shape)
    break

