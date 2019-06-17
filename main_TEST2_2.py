# -*- coding: utf-8 -*-
"""
FROM: https://github.com/mitmedialab/3D-VAE/blob/master/vq_vae/auto_encoder.py
https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html
2:https://github.com/bhpfelix/Variational-Autoencoder-PyTorch/blob/master/src/vanila_vae.py
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

#class LSA_VAE(nn.Module):
#    def __init__(self, d, att_len, kl_coef=0.1, **kwargs):
#        super(LSA_VAE, self).__init__()
        
#        self.att_len = att_len
#        self.encoder = nn.Sequential(
#            nn.Conv2d(3, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
#            nn.BatchNorm2d(d // 2),
#            nn.ReLU(inplace=True),
#            
#            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
#            nn.BatchNorm2d(d),
#            nn.ReLU(inplace=True),
#            
#            ResBlock(d, d, bn=True),
#            nn.BatchNorm2d(d),
#            
#            ResBlock(d, d, bn=True),
#            
##            nn.linear(self.f**2 * self.d, )
#        )
#        self.decoder = nn.Sequential(
#            ResBlock(d, d, bn=True),
#            nn.BatchNorm2d(d),
#            ResBlock(d, d, bn=True),
#            nn.BatchNorm2d(d),
#
#            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
#            nn.BatchNorm2d(d//2),
#            nn.ReLU(inplace=True),
#            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
#        )
#        
#        
#        self.f = 32
#        self.d = d
##        self.fc11 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
#        self.fc11 = nn.Linear(self.f ** 2 * self.d, self.att_len)
##        self.fc12 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
#        self.fc12 = nn.Linear(self.f ** 2 * self.d, self.att_len)
#        self.kl_coef = kl_coef
#        self.kl_loss = 0
#        self.mse = 0
#        
#        
#    def encode(self, x):
#        h1 = self.encoder(x)
#        print("h1.size():", h1.size())
#        self.tmp_h1_size = h1.size()
##        h1 = h1.view(-1, self.d * self.f ** 2) #reshape
#        h1 = h1.view(h1.size(0), -1) #reshape
#        print("h1.size():", h1.size())
#        return self.fc11(h1), self.fc12(h1)
#
#    def reparameterize(self, mu, logvar):
#        if self.training:
#            std = logvar.mul(0.5).exp_()
#            eps = Variable(std.new(std.size()).normal_())
#            return eps.mul(std).add_(mu)
#        else:
#            return mu
#
#    def decode(self, z):
##        z = z.view(-1, self.d, self.f, self.f)
#        z = z.view()##########################################################
#        h3 = self.decoder(z)
#        return F.tanh(h3)
#    def forward(self, x):
#        mu, logvar = self.encode(x)
#        z = self.reparameterize(mu, logvar)
#        print(z.size())
#        return self.decode(z), mu, logvar
        
class LSA_VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(LSA_VAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

#        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
#        self.bn4 = nn.BatchNorm2d(ndf*8)
#
#        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
#        self.bn5 = nn.BatchNorm2d(ndf*8)

#        self.fc1 = nn.Linear(ndf*8*4*4, latent_variable_size)
#        self.fc2 = nn.Linear(ndf*8*4*4, latent_variable_size)
        self.fc1 = nn.Linear(ndf*4*4, latent_variable_size) #m2?
        self.fc2 = nn.Linear(ndf*4*4, latent_variable_size)

        # decoder
#        self.d1 = nn.Linear(latent_variable_size, ngf*8*2*4*4)
        self.d1 = nn.Linear(latent_variable_size, ngf*4*4)

#        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
#        self.pd1 = nn.ReplicationPad2d(1)
#        self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
#        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

#        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
#        self.pd2 = nn.ReplicationPad2d(1)
#        self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
#        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h5 = self.leakyrelu(self.bn3(self.e3(h2)))
#        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
#        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
#        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
#        h5 = h5.view(-1, self.ndf*8*4*4)
        h5 = h5.view(-1, self.ndf*4*4)

        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if  torch.cuda.is_available() and False:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*4*4, 4, 4)
#        h1 = h1.view(-1, self.ngf*8*2, 4, 4)
#        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
#        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
#        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h1)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

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
        print("size:", logvar.size(), attribute.size())
        self.attr_real_loss = F.cross_entropy(logvar, attribute)
        # Normalise by same number of elements as in reconstruction
#        self.kl_loss /= batch_size * 3 * 1024

        # return mse
        return self.attr_real_loss, self.kl_loss
    

    def latest_losses(self):
        return {'attr_real_loss': self.attr_real_loss, 'kl_loss': self.kl_loss}
    
    
def train(model_e_d, opt, train_loader):
    model_e_d.train()
    for idx, (img, attr) in enumerate(train_loader):
        
#        img = img.cuda() if torch.cuda.is_available() else img
#        attr = attr.cuda() if torch.cuda.is_available() else attr
        
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
    attrs__all = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", 
                  "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", 
                  "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", 
                  "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", 
                  "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", 
                  "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", 
                  "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", 
                  "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", 
                  "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", 
                  "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", 
                  "Wearing_Necktie", "Young"] # 40
    attrs_default = [
    'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
]
    attrs_list = attrs_default.copy()
    batch_size = 32
    n_samples = 16
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    PLL = PerceptualLoss()
#    model_e_d = LSA_VAE(8, len(attrs_list))
    model_e_d = LSA_VAE(nc=3, ngf=128, ndf=128, latent_variable_size=256)
    
    
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