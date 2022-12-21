import torchvision
from torchsummary import summary
import torch.nn as nn
def get_model():
    Res_Deeplab = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    Res_Deeplab.classifier[4] = nn.Conv2d(256,9,kernel_size=(1, 1), stride=(1, 1))
    return Res_Deeplab