from config import *
import torch
from diffusion import *
import matplotlib.pyplot as plt
from dataset.dataset import tensor_to_pil

def backward_senoise(model, batch_x_t):
    step = [batch_x_t,]

    global alphas,alphas_cumprod,variance

    model = model.to(DEVICE)
    batch_x_t = batch_x_t.to(DEVICE)
    alphas = alphas.to(DEVICE)
    alphas_cumprod = alphas_cumprod.to(DEVICE)
    variance = variance.to(DEVICE)

    with torch.no_grad():
        for t in range(T-1,-1,-1):
            batch_t = torch.full((batch_x_t.size(0),),t) #[999,999,999,999,999,999,999,999,999,999]
            batch_t = batch_t.to(DEVICE)
            # 预测x_t时刻的噪音
            batch_predict_noise_t = model(batch_x_t, batch_t)
            # 生成t-1时刻的图像
            shape = (batch_x_t.size(0),batch_x_t.size(1),1,1)
            batch_mean_t = 1 / torch.sqrt(alphas[batch_t].view(*shape)) * \
                (
                    batch_x_t -
                    ((1-alphas[batch_t].view(*shape)) / torch.sqrt(1-alphas_cumprod[batch_t].view(*shape))) * batch_predict_noise_t
                )
            if t != 0:
                batch_x_t = batch_mean_t + torch.randn_like(batch_x_t) * torch.sqrt(variance[batch_t].view(*shape))
            else:
                batch_x_t = batch_mean_t 
            batch_x_t = torch.clamp(batch_x_t, -1.0, 1.0).detach() #将像素点调整到-1，1之间, detach()将数据从GPU拉到CPU
            step.append(batch_x_t)

    return step

if __name__ == '__main__': 
    #加载模型
    difussion_model = torch.load('./checkpoints/model.pt')
    # 生成噪音图
    batch_size = 10
    img_channel = 1
    batch_x_t = torch.randn(size=(batch_size,img_channel,IMG_SIZE,IMG_SIZE))
    # 逐步得到去噪原图
    steps = backward_senoise(difussion_model, batch_x_t)
    # 绘制数量
    num_imgs = 40

    # #绘制还原过程
    # plt.figure(figsize=(20,20))
    # for i in range(batch_size):
    #     for j in range(0,num_imgs):
    #         idx = int(T/num_imgs)*(j+1)
    #         #像素值还原到0，1
    #         final_img = (steps[idx][i].to('cpu')+1)/2
    #         final_img = tensor_to_pil(final_img)
    #         plt.subplot(batch_size,num_imgs,i*num_imgs+j+1)
    #         plt.imshow(final_img)
    # plt.show()

    plt.figure(figsize=(10,10))
    for i in range(10):
        #像素值还原到0，1
        final_img = (steps[-1][i].to('cpu')+1)/2
        final_img = tensor_to_pil(final_img)
        plt.subplot(int(batch_size/5),int(batch_size/2),i+1)
        plt.axis('off')
        plt.imshow(final_img)
    plt.show()