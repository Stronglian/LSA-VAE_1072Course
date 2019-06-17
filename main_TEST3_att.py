# -*- coding: utf-8 -*-
"""
FROM: https://github.com/mitmedialab/3D-VAE/blob/master/vq_vae/auto_encoder.py
https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html
https://github.com/elvisyjlin/AttGAN-PyTorch/blob/master/attgan.py
"""
global ta
ta = None
import numpy as np
#import logging
import torch
import torch.utils.data
from torch import nn
from nn import Conv2dBlock, ConvTranspose2dBlock
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

MAX_DIM = 64 * 16  # 1024

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
    def __init__(self, d, att_len, kl_coef=0.1, 
                 enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=13, shortcut_layers=1, inject_layers=0, img_size=128, **kwargs):
        super(LSA_VAE, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2**enc_layers  # f_size = 4 for 128x128
        
        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn
            )]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)
        
        layers = []
        n_in = n_in + n_attrs  # 1024 + 13
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2**(dec_layers-i-1), MAX_DIM)
                layers += [ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                )]
                n_in = n_out
                n_in = n_in + n_in//2 if self.shortcut_layers > i else n_in
                n_in = n_in + n_attrs if self.inject_layers > i else n_in
            else:
                layers += [ConvTranspose2dBlock(
                    n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh'
                )]
        self.dec_layers = nn.ModuleList(layers)
    
    def encode(self, x):
        z = x
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        return zs
    
    def decode(self, zs, a):
        a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, 1, self.f_size, self.f_size)
        z = torch.cat([zs[-1], a_tile], dim=1)
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            if self.shortcut_layers > i:  # Concat 1024 with 512
                z = torch.cat([z, zs[len(self.dec_layers) - 2 - i]], dim=1)
            if self.inject_layers > i:
                a_tile = a.view(a.size(0), -1, 1, 1) \
                          .repeat(1, 1, self.f_size * 2**(i+1), self.f_size * 2**(i+1))
                z = torch.cat([z, a_tile], dim=1)
        return z


    def forward(self, x, a=None, mode='enc-dec'):
        if mode == 'enc-dec':
            assert a is not None, 'No given attribute.'
            return self.decode(self.encode(x), a)
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a)
        raise Exception('Unrecognized mode: ' + mode)
        
#    def forward(self, x):
#        mu, logvar = self.encode(x)
#        z = self.reparameterize(mu, logvar)
#        return self.decode(z), mu, logvar

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
        
        opt.zero_grad()
        
        recon_x, mu, logvar = model_e_d(img, attr)
        
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
    model_e_d = LSA_VAE(8, len(attrs_list))
    
    
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