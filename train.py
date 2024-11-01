import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from dataloader import train_dataloader
from gan import Discriminator, Generator


def to_img(x):  # 将结果的-0.5~0.5变为0~1保存图片
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  #限制范围至01
    out = out.view(-1, 1, 28, 28)
    return out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    D = Discriminator().to(device)
    G = Generator().to(device)
    optimizerD = optim.Adam(D.parameters(), lr=0.0003)
    optimizerG = optim.Adam(G.parameters(), lr=0.0003)
    epochs = 50

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(train_dataloader):
            batch_size = imgs.size(0)
            real_imgs = imgs.view(batch_size, -1).to(device)
            real_labels = torch.ones(batch_size).reshape(batch_size, -1).to(device)
            fake_labels = torch.zeros(batch_size).reshape(batch_size, -1).to(device)

            # 先训练鉴别器
            optimizerD.zero_grad()
            real_pred = D(real_imgs)
            lossD_real = criterion(real_pred, real_labels)

            z = torch.randn(batch_size, 100).to(device)
            fake_imgs = G(z).detach()  # 防止计算G的梯度
            fake_pred = D(fake_imgs)
            lossD_fake = criterion(fake_pred, fake_labels)
            lossD = (lossD_real + lossD_fake).to(device)
            lossD.backward()
            optimizerD.step()

            # 开始训练生成器
            optimizerG.zero_grad()
            z = torch.randn(batch_size, 100).to(device)
            gen_imgs = G(z)
            d_pred = D(gen_imgs)
            lossG = criterion(d_pred, real_labels)
            lossG.backward()
            optimizerG.step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                      'D real: {:.6f}, D fake: {:.6f}'.format(
                    epoch, epochs, lossD.cpu().item(), lossG.cpu().item(),
                    real_pred.cpu().mean().item(), fake_pred.cpu().mean().item()))
            if epoch == epochs - 1:
                grid_imgs = make_grid(to_img(gen_imgs).cpu(), nrow=8, padding=2)
                save_image(grid_imgs, 'results/gird_imgs.png')

                torch.save(G.state_dict(), 'results/generator.pth')
                torch.save(D.state_dict(), 'results/discriminator.pth')
