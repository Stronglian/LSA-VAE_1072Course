# -*- coding: utf-8 -*-
"""
特徵擷取測試
練習: https://zhuanlan.zhihu.com/p/30315331
"""



import torchvision.models as models


vgg19 = models.vgg19(pretrained=True)

for parma in vgg19.parameters():
    parma.requires_grad = False
    
vgg19.features()