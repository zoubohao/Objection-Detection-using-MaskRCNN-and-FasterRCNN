import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
import torch
from torchvision.models.detection import MaskRCNN
from SelfBackBone import ResNet

batchSize = 1
weightDecay = 5e-4
trainOrTest = "Train"
num_classes = 2 # include the background
trainingEpoch = 5
trainingTimesInOneEpoch = 5000
lr = 1e-4
displayTimes = 25
testWeightLoad = 4
ifLoadWeightForTraining = False
loadWeightFile = 0
scoreThreshold = 0.6

# load a pre-trained model for classification and return
# only the features
### The last feature map which size is [B, C, H, W]

# backbone (nn.Module): the network used to compute the features for the model.
# It should contain a out_channels attribute, which indicates the number of output
# channels that each feature map has (and it should be the same for all feature maps).
# The backbone should return a single Tensor or and OrderedDict[Tensor].
# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone = ResNet(layers=[8, 12, 24, 5])

# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
# backbone.out_channels = 1280
backbone.out_channels = 512

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512),
                                   aspect_ratios=(0.5, 1.0, 2.0))


# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.

# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
# Features are assumed to have all the same number of channels, but they can have different sizes.

#         features = self.backbone(images.tensors)
#         if isinstance(features, torch.Tensor):
#             features = OrderedDict([('0', features)])

### if we will use FPN (Feature pyramid network), we should set featmap_name as ['0','1','2','3','4']
### however, we do not use FPN, so only set ['0'] is ok.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0','1','2'],
                                                output_size=[7,7],
                                                sampling_ratio=2)

mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0','1','2'],
                                                     output_size=[14,14],
                                                     sampling_ratio=2)

# put the pieces together inside a FasterRCNN model

# rpn_head – module that computes the objectness and regression deltas from the RPN
from HeadANDPredictionModules import RPNHead
rpn_head = RPNHead(backbone.out_channels,anchor_generator)

# box_roi_pool – the module which crops and resizes
# the feature maps in the locations indicated by the bounding boxes

# box_head – module that takes the cropped feature maps as input
from HeadANDPredictionModules import BoxHead
representSize = 512
box_head = BoxHead(backbone.out_channels,representation_size=representSize)

# box_predictor – module that takes the output of box_head and
# returns the classification logits and box regression deltas.

# mask_roi_pool – the module which crops and resizes the feature maps
# in the locations indicated by the bounding boxes, which will be used for the mask head.

# mask_head – module that takes the cropped feature maps as input
from HeadANDPredictionModules import MaskRCNNHeads
mask_heads_out_channels = 512
mask_head = MaskRCNNHeads(backbone.out_channels,(256,128,mask_heads_out_channels),dilation=1)

# mask_predictor – module that takes the output of the mask_head
# and returns the segmentation mask logits

import torchvision.models.detection.faster_rcnn as fasterRCNN
import torchvision.models.detection.mask_rcnn as maskRCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MaskRCNN(  backbone=backbone,
                   rpn_anchor_generator=anchor_generator,
                   rpn_head = rpn_head,
                   box_roi_pool=roi_pooler,
                   box_head = box_head,
                   box_predictor=fasterRCNN.FastRCNNPredictor(representSize, num_classes),
                   mask_roi_pool= mask_roi_pooler,
                   mask_head=mask_head,
                   mask_predictor= maskRCNN.MaskRCNNPredictor(mask_heads_out_channels,dim_reduced=256,num_classes = num_classes)
                   ).to(device)

if ifLoadWeightForTraining :
    model.load_state_dict(torch.load("./Model_" + str(loadWeightFile) + ".pth"))


import torchvision.transforms as T
from D_PennFudanPedDataSet import PennFudanDataset
transformC = T.Compose([T.RandomApply([T.ColorJitter()],p=0.3),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        T.RandomErasing(p = 0.3)])
dataSet = PennFudanDataset('./PennFudanPed/',transformC,device)

def dataGenerator(oneDataSet):
    while True:
        for _ ,(imageG, targetG) in enumerate(oneDataSet):
            yield imageG, targetG

if trainOrTest.lower() == "train":
    import torch.optim.rmsprop as rmsprop
    from LearningRateSch import CosineDecaySchedule
    trainingDataGenerator = dataGenerator(dataSet)
    optimizer = rmsprop.RMSprop(model.parameters(),lr=lr,alpha=0.95, weight_decay=weightDecay,momentum=0.9)
    scheduler = CosineDecaySchedule(lrMin=5e-6,lrMax=lr,tMaxIni=1250,factor=1.25,lrDecayRate=0.94)
    # For Training
    model.train()
    # images (list[Tensor]): images to be processed
    # targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
    trainingTimes = 0
    for e in range(trainingEpoch):
        for t in range(trainingTimesInOneEpoch):
            images = []
            targets = []
            for b in range(batchSize):
                img , target = trainingDataGenerator.__next__()
                images.append(img)
                targets.append(target)
            # {'loss_classifier': tensor(0.6214, grad_fn=<NllLossBackward>),
            # 'loss_box_reg': tensor(0.0002, grad_fn=<DivBackward0>),
            # 'loss_mask': tensor(5.7685, grad_fn=<BinaryCrossEntropyWithLogitsBackward>),
            # 'loss_objectness': tensor(0.7188, grad_fn=<BinaryCrossEntropyWithLogitsBackward>),
            # 'loss_rpn_box_reg': tensor(4.0351, grad_fn=<DivBackward0>)}
            optimizer.zero_grad()
            losses = model(images,targets)
            addedLosses = losses["loss_classifier"] + losses["loss_box_reg"] + losses["loss_mask"] + losses["loss_objectness"] + losses["loss_rpn_box_reg"]
            if trainingTimes % displayTimes == 0:
                print("#######")
                print("Epoch : ",e)
                print("Current Training Times : ",t)
                print("Current learning rate : ",optimizer.state_dict()["param_groups"][0]["lr"])
                print("loss_classifier : ", losses["loss_classifier"])
                print("loss_box_reg : ", losses["loss_box_reg"])
                print("loss_mask : ", losses["loss_mask"])
                print("loss_objectness : ",losses["loss_objectness"])
                print("loss_rpn_box_reg : ",losses["loss_rpn_box_reg"])
            addedLosses.backward()
            optimizer.step()
            learning_rate = scheduler.calculateLearningRate()
            state_dic = optimizer.state_dict()
            state_dic["param_groups"][0]["lr"] = learning_rate
            optimizer.load_state_dict(state_dic)
            scheduler.step()
            trainingTimes += 1
        torch.save(model.state_dict(), "./Model_" + str(e) + ".pth")
else:
    # For inference
    model.load_state_dict(torch.load("./Model_" + str(testWeightLoad) + ".pth"))
    model.eval()
    import PIL.Image as Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    testImagePath = "./test7.png"
    maskSavePath = "./test7Masks.png"
    testImg = Image.open(testImagePath).convert("RGB")
    testImgTensor = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(T.ToTensor()(testImg)).to(device)
    ### This prediction is a dict. It contains boxes, labels, scores, masks
    prediction = model([testImgTensor],None)
    bboxes = prediction[0]["boxes"]
    labels = prediction[0]["labels"]
    scores = prediction[0]["scores"]
    masks = prediction[0]["masks"]
    img = np.array(Image.open(testImagePath), dtype=np.int32)
    indices = torchvision.ops.nms(bboxes,scores,iou_threshold=0.2)
    finalIndices = []
    print(scores)
    for index in indices:
        if scores[index] >= scoreThreshold:
            finalIndices.append(index.reshape([1]))
    print(finalIndices)
    finalIndices = torch.cat(finalIndices,dim=0)
    bboxes = bboxes[finalIndices].cpu().detach().numpy()
    labels = labels[finalIndices].cpu().detach().numpy()
    scores = scores[finalIndices].cpu().detach().numpy()
    masks = masks[finalIndices].cpu().detach().numpy()
    # Create figure and axes
    print(scores)
    print(bboxes)
    fig, ax = plt.subplots(1)
    if len(masks) > 0:
        import cv2
        c,h,w = masks[0].shape[0],masks[0].shape[1],masks[0].shape[2]
        zeros = np.zeros(shape=[h,w],dtype=np.float32)
        for mask in masks:
            zeros += np.reshape(mask,newshape=[h,w])
        cv2.imwrite(maskSavePath,zeros * 255.)
    for box in bboxes:
        # Create a Rectangle patch
        xMin = box[0]
        yMin = box[1]
        height = box[3] - yMin
        width = box[2] - xMin
        rect = patches.Rectangle((xMin, yMin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    # Display the image
    ax.imshow(img)
    plt.show()


