import torch
import torch.nn as nn
import torchvision.models as models
from . import DA_network

class Deeplabv3plusMobilenetFeatureExtractor(nn.Module):
    def __init__(self, layers_to_extract):
        super(Deeplabv3plusMobilenetFeatureExtractor, self).__init__()
        self.deeplabv3plus_mobilenet = DA_network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19)
        self.deeplabv3plus_mobilenet.load_state_dict( torch.load('/workspace/pado-internal/models/checkpoint/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')['model_state']  )

        for param in self.deeplabv3plus_mobilenet.parameters():
            param.requires_grad = False
        
        self.layers_to_extract = layers_to_extract if layers_to_extract is not None else []


    def forward(self, x):
        features = {}

        x = self.deeplabv3plus_mobilenet.backbone.low_level_features(x)

        return x
    
class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, layers_to_extract):
        super(ResNet18FeatureExtractor, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        
        # Disable gradients for the feature extractor
        for param in self.resnet18.parameters():
            param.requires_grad = False
        
        
        self.layers_to_extract = layers_to_extract if layers_to_extract is not None else []

        self.selected_layers = {
            'conv1': self.resnet18.conv1,
            'bn1': self.resnet18.bn1,
            'relu': self.resnet18.relu,
            'maxpool': self.resnet18.maxpool,
            'layer1': self.resnet18.layer1,
            'layer2': self.resnet18.layer2,
            'layer3': self.resnet18.layer3,
            'layer4': self.resnet18.layer4,
        }

    def forward(self, x):
        features = {}
        for name, layer in self.selected_layers.items():
            x = layer(x)
            if name in self.layers_to_extract:
                features[name] = x
        return features
    
def DA_loss(pred, gt, args, feature='resnet'):
    """
    Domain adaptation loss function
    Args:
        feature: 'segmentation' or 'resnet'
    """
    if pred.shape[1]==1:
        pred = pred.repeat(1, 3, 1, 1)
    if gt.shape[1]==1:
        gt = gt.repeat(1, 3, 1, 1)
    if feature=='resnet':
        layers_to_extract = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        feature_extractor = ResNet18FeatureExtractor(layers_to_extract).to(args.device)
        image_far_features = feature_extractor(gt.float())
        img_conv_features = feature_extractor(pred.float())
        da_loss = 0    
        for layer in layers_to_extract:
            feat_img = image_far_features[layer]
            feat_conv = img_conv_features[layer]
            da_loss += args.da_loss_weight * args.l1_criterion(feat_conv, feat_img)
        return da_loss 
    else:
        layers_to_extract = ['low_level_features', 'high_level_features']
        feature_extractor = Deeplabv3plusMobilenetFeatureExtractor(layers_to_extract).to(args.device)
        image_far_features = feature_extractor(gt.float())
        img_conv_features = feature_extractor(pred.float())
        da_loss = 0    
        for layer in layers_to_extract:
            feat_img = image_far_features
            feat_conv = img_conv_features
            da_loss += args.da_loss_weight * args.l1_criterion(feat_conv, feat_img)
        return da_loss

