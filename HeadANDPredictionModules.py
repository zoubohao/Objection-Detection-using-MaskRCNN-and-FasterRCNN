import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.ops import misc as misc_nn_ops



class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Arguments:
        in_channels (int): number of channels of the input feature. It is out channels of backbone network.
        anchorsGeneModule (nn.Module): It is the anchor generation module.
        (AnchorGenerator(sizes=(32, 64, 128, 256, 512),aspect_ratios=(0.5, 1.0, 2.0)))
    """

    def __init__(self, in_channels, anchorsGeneModule):
        super(RPNHead, self).__init__()
        num_anchors = anchorsGeneModule.num_anchors_per_location()[0]
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

    def forward(self, x):
        """
        :param x: (List[Tensor])
        :return:
        """
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


### This is the box head.
# resolution = box_roi_pool.output_size[0]
# representation_size = 1024
# box_head = TwoMLPHead(
#     out_channels * resolution ** 2,
#     representation_size)  The out_channels is the backbone out channels.
class BoxHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels.The in_channels is the backbone out channels.
        representation_size (int): size of the intermediate representation. You can set any value.
    """

    def __init__(self, in_channels, representation_size):
        super(BoxHead, self).__init__()
        self.conv = misc_nn_ops.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc6 = nn.Linear(in_channels, representation_size)

    def forward(self, x):
        #x = x.flatten(start_dim=1)
        conv = self.conv(x)
        relu = F.relu(conv)
        act = self.pooling(relu).flatten(start_dim=1)
        result = self.fc6(act)
        return result

### This is the mask head
#        if mask_head is None:
#            mask_layers = (256, 256, 256, 256)
#            mask_dilation = 1
#            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation) The out_channels is the backbone out channels.



class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Arguments:
            in_channels (int): it is backbone out channels.
            layers (): mask layers. Like (256,256,256)
            dilation (int): mask dilation
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d["mask_fcn{}".format(layer_idx)] = misc_nn_ops.Conv2d(
                next_feature, layer_features, kernel_size=3,
                stride=1, padding=dilation, dilation=dilation)
            d["BatchNor{}".format(layer_idx)] = misc_nn_ops.BatchNorm2d(layer_features)
            d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super(MaskRCNNHeads, self).__init__(d)
