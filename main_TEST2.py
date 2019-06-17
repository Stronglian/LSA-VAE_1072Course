# -*- coding: utf-8 -*-
"""
FROM: https://github.com/mitmedialab/3D-VAE/blob/master/vq_vae/auto_encoder.py
https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html
"""
import numpy as np
#import logging
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
#import torch.distributions as tdist
import torchvision.models as models
from torch import optim
#from lossess import PerceptualLoss
#from data import CelebA

from torch.utils.data import DataLoader

#from .nearest_embed import NearestEmbed

class PerceptualLoss():	
    def contentFunc(self):
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        return model
		
    def __init__(self):
        print("create PerceptualLoss")
    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        loss = torch.sqrt((f_fake - f_real)**2)
        return loss

class ResBlock(nn.Module):
    def __init__(self, in_channels, channels, bn=False):
        super(ResBlock, self).__init__()

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

class LSA_VAE(nn.Module):
    def __init__(self, d, kl_coef=0.1, **kwargs):
        super(LSA_VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.f = 8
        self.d = d
        self.fc11 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
        self.fc12 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
        self.kl_coef = kl_coef
        self.kl_loss = 0
        self.mse = 0
        
    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(-1, self.d * self.f ** 2)
        return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return F.tanh(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, size):
        sample = Variable(torch.randn(size, self.d * self.f ** 2), requires_grad=False)
        if self.cuda():
            sample = sample.cuda()
        return self.decode(sample).cpu()

    def loss_function(self, x, recon_x, attribute, mu, logvar):
#        self.mse = F.mse_loss(recon_x, x)
#        batch_size = x.size(0)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl_loss = F.kl_div(mu, torch.Tensor(np.random.normal(0, 1, mu.size())))
#        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.attr_real_loss = F.cross_entropy(logvar, attribute)
        # Normalise by same number of elements as in reconstruction
#        self.kl_loss /= batch_size * 3 * 1024

        # return mse
        return self.attr_real_loss, self.kl_loss
    

    def latest_losses(self):
        return {'mse': self.mse, 'kl': self.kl_loss}
    
    
def train(model_e_d, opt, train_loader):
    model_e_d.train()
    for idx, (img, attr) in enumerate(train_loader):
        
        
        opt.zero_grad()
        
        recon_x, mu, logvar = model_e_d(img)
        
        attr_real_loss, kl_real_loss = model_e_d.loss_function(img, recon_x, attr, mu, logvar)
        rec_loss = PLL.get_loss(recon_x, img)
        
        ir_loss = attr_real_loss - kl_real_loss + rec_loss
        
        ir_loss.backward()
        opt.step()
        
        print("Train :: ir_loss: %0.5f, attr_real_loss: %0.5f, kl_loss: %0.5f, rec_loss: %0.5f" %(ir_loss, attr_real_loss, kl_real_loss, rec_loss))
        
        
    
    return
    
def test():
    
    
    return

if __name__=="__main__":
    
    attr_path = "data/list_attr_celeba.txt"
    data_path = "data/img_align_celeba"
    img_size=128
    num_workers = 0
    attrs_default = [
    'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
]
    attrs_list = attrs_default.copy()
    batch_size = 32
    n_samples = 16
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    PLL = PerceptualLoss()
    model_e_d = LSA_VAE(img_size)
    
    
    from data import CelebA
    train_dataset = CelebA(data_path, attr_path, img_size, 'train', attrs_list)
    valid_dataset = CelebA(data_path, attr_path, img_size, 'valid', attrs_list)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, drop_last=True
        )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=n_samples, num_workers=num_workers,
        shuffle=False, drop_last=False
        )
    
#    SGD_optimizer = optim.SGD(model_e_d.parameters(), lr=0.0005, momentum=0.5, weight_decay=1e-4)
    ADAM_optimizer = optim.Adam(model_e_d.parameters(), lr=0.0005, betas = (0.5, 0.99))
    
    train(model_e_d, ADAM_optimizer, train_dataloader)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    main()