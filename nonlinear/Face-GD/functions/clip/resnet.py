import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50
from torchvision import transforms
class Resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = resnet50(pretrained=True).cuda()
        self.classifier.requires_grad = True
        self.preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop((224,224)),
                # transforms.ToTensor(),
                transforms.Lambda(lambda x: (x+1)/2),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def get_residual(self, image, cl):
        image = self.preprocess(image)
        logits = self.classifier(image)
        return self.criterion(logits,cl)