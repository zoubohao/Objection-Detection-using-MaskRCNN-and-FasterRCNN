import torchvision
import torch
import torchvision.models.detection.faster_rcnn as fasterRCNN


m = torchvision.ops.MultiScaleRoIAlign(["1", "3"], [4,4], 2)
i = fasterRCNN.OrderedDict()
### The is like FPN (Feature pyramid network)
i["1"] = torch.rand(3, 16, 64, 64)
i["2"] = torch.rand(3, 5, 32, 32)  # this feature won't be used in the pooling
i["3"] = torch.rand(3, 16, 16, 16)
# create some random bounding boxes
boxes = torch.rand(6, 4) * 256
boxes[:, 2:] += boxes[:, :2]
# original image size, before computing the feature maps
image_sizes = [(512, 512),(512, 512),(512, 512)]
print(boxes)

# x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
# all the same number of channels, but they can have different sizes.

# boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
# (x1, y1, x2, y2) format and in the image reference size, not the feature map reference.

# image_shapes (List[Tuple[height, width]]): the sizes of each image before they
# have been fed to a CNN to obtain feature maps. This allows us to infer the
# scale factor for each one of the levels to be pooled.
output = m(i, [boxes,boxes,boxes], image_sizes)
print(output.shape)


a = {1:2,3:4}
print(list(a.values()))

