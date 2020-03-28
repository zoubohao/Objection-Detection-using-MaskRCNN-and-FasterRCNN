import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import dataset
import h5py


class COCO_Train_Data_Set(dataset.Dataset):

    def __init__(self, h5pyFilePath, transforms,imageID2info, imageID2fileName):
        self.dataset = h5py.File(h5pyFilePath,mode="r")
        self.transforms = transforms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imageID2FileName = {}
        self.fileName2Info = {}
        with open(imageID2fileName,"r") as rh:
            for line in rh:
                oneLine = line.strip("\n").split("\t")
                self.imageID2FileName[oneLine[0]] = oneLine[1]
        with open(imageID2info,"r") as rh:
            for line in rh:
                oneLine = line.strip("\n").split("\t")
                fileName = self.imageID2FileName[oneLine[0]]
                bbox = np.array([float(v) for v in oneLine[1].split(",")],dtype=np.float32)
                category_id = int(oneLine[2])
                iscrowd = int(oneLine[3])
                if fileName not in self.fileName2Info:
                    newMap = {"category_id": list(), "iscorwd": list(), "bbox": list()}
                    newMap["bbox"].append(bbox)
                    newMap["category_id"].append(category_id)
                    newMap["iscorwd"].append(iscrowd)
                    self.fileName2Info[fileName] = newMap
                else:
                    thisMap = self.fileName2Info[fileName]
                    thisMap["bbox"].append(bbox)
                    thisMap["category_id"].append(category_id)
                    thisMap["iscorwd"].append(iscrowd)
                    self.fileName2Info[fileName] = thisMap
        self.fileNames = list(self.fileName2Info.keys())
        #print(len(self.fileNames))
        #print(len(self.fileName2Info))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = np.array(self.dataset[fileName])
        inforMap = self.fileName2Info[fileName]
        bboxes = torch.from_numpy(np.array(inforMap["bbox"])).to(self.device)
        labels = torch.as_tensor(inforMap["category_id"],dtype=torch.int64).to(self.device)
        # suppose all instances are not crowd
        iscrowd = torch.as_tensor(inforMap["iscorwd"], dtype=torch.int64).to(self.device)
        target = {"boxes": bboxes, "labels": labels,"iscrowd": iscrowd}
        if self.transforms is not None:
            img = self.transforms(img)
        return img.to(self.device), target


if __name__ == "__main__":
    import torchvision
    testDataSet = COCO_Train_Data_Set("./cocoTrain.h5py",torchvision.transforms.ToTensor(),
                                      "./imageID2infoTrain.txt","./imageID2fileNameTrain.txt")
    print(testDataSet.__getitem__(11200)[0].shape)
