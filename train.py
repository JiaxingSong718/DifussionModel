from config import *
from torch.utils.data import DataLoader
from dataset.dataset import train_dataset
from Unet import UNet
from diffusion import forward_difussion
import torch
from torch import nn
import os

epochs = 200
batch_size = 600
img_channel = 1

dataloader = DataLoader(train_dataset,batch_size=batch_size,num_workers=4,persistent_workers=True,shuffle=True)

try:
    model = torch.load('./checkpoints/model.pt')
except:
    model = UNet(img_channel=img_channel).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
loss_fn = nn.L1Loss() #绝对值误差均值

if __name__ == '__main__':
    model.train()
    for epoch in range(epochs):
        last_loss = 0
        for batch_x,_ in dataloader:
            #像素值调整到[-1,1]之间，以便与高斯噪声值匹配
            batch_x = batch_x.to(DEVICE)*2 - 1
            batch_t = torch.randint(0,T,size=(batch_x.size(0),)).to(DEVICE) # 每张图片生成随机步数
            # 生成t时刻的加噪图片和对应噪声
            batch_x_t, batch_noise_t = forward_difussion(batch_x, batch_t)
            # 模型预测t时刻噪声
            batch_predict_t = model(batch_x_t,batch_t)
            # 求损失
            loss = loss_fn(batch_predict_t,batch_noise_t)
            #优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
        print('epoch:{} loss:{}'.format(epoch,last_loss))
        torch.save(model,'./checkpoints/model.pt.tmp')
        os.replace('./checkpoints/model.pt.tmp','./checkpoints/model.pt')