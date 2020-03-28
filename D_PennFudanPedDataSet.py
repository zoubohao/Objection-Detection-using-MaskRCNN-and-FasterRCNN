import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import dataset

class PennFudanDataset(dataset.Dataset):
    def __init__(self, root, transforms,device):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        self.device = device

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        ### This will judge how many instances in this mask.
        ### For example, if you have 2 people in this image, people is the
        # instance which we want to detect. The obj_ids is [0,1,2]
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        ### Only preserves the instances in image. Background is 0
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        ### This means that splitting instances which are in this mask in to a binary mask.
        ### For example, if the obj_ids is [0,1,2], than, the masks is [2,H,W] which elements only
        ### contain True and False.
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        # boxes (FloatTensor[N, 4])
        boxes = torch.as_tensor(boxes, dtype=torch.float32).to(self.device)
        # there is only one class
        # labels (Int64Tensor[N]): the label for each bounding box
        labels = torch.ones((num_objs,), dtype=torch.int64).to(self.device)
        # masks (UInt8Tensor[N, H, W])
        masks = torch.as_tensor(masks, dtype=torch.uint8).to(self.device)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64).to(self.device)

        target = {"boxes": boxes, "labels": labels, "masks": masks,"iscrowd": iscrowd}
        if self.transforms is not None:

            img = self.transforms(img)

        return img.to(self.device), target

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    import torchvision
    dataSet = PennFudanDataset('./PennFudanPed/',torchvision.transforms.ToTensor() , torch.device("cpu"))
    print(dataSet.__getitem__(4))

