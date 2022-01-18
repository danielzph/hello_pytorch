import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
from torchvision.datasets import MNIST
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import os
from torchvision import transforms as T
# import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

transform = T.Compose([
    T.Resize((28, 28)),
    T.Grayscale(),
    T.ToTensor(),
])


# 定义数据对象
class MNISTDataset(Dataset):
    def __init__(self, images, transform):
        self.dataset = images.data.to(torch.float32)
        self.transform = transform

    def __getitem__(self, index):
        data = self.dataset[index]
        data = np.asarray(data)
        data = Image.fromarray(data)
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.dataset)


##加载数据
def load_mnist():
    trainMnist = MNIST('', download=True, train=True)
    testMnist = MNIST('', download=True, train=False)
    trainMnist = MNISTDataset(trainMnist, transform=transform)
    testMnist = MNISTDataset(testMnist, transform=transform)
    trainLoader = DataLoader(trainMnist, shuffle=True, batch_size=256)
    testLoader = DataLoader(testMnist, shuffle=False, batch_size=24)
    return (trainLoader, testLoader, trainMnist, testMnist)


# 定义网络模型(卷积自编码器)
class EncodeModel(nn.Module):
    def __init__(self, judge=True):
        super(EncodeModel, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7),  # 16*22*22
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16*11*11

            nn.Conv2d(16, 4, kernel_size=3),  # 4*9*9
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4*4*4

            nn.Conv2d(4, 4, kernel_size=2),  # 4*3*3
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(4, 1, kernel_size=5, stride=2),  # 1*9*9
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(1, 1, kernel_size=7, stride=2),  # 1*23*23
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(1, 1, kernel_size=6, stride=1),  # 1*28*28
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, judge):
        enOutputs = self.encode(x)
        outputs = self.decode(enOutputs)
        if judge:
            return outputs
        else:
            return enOutputs


def train(trainLoader, testLoader):
    model = EncodeModel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
    criterion = nn.MSELoss().cuda()
    epochs = 50
    for epoch in range(epochs):
        for (i, trainData) in enumerate(trainLoader):
            trainData = trainData.cuda()
            outputs = model(trainData, True).cuda()
            optimizer.zero_grad()
            loss = criterion(outputs, trainData)
            loss.backward()
            optimizer.step()
            torch.save(model.state_dict(), 'EncodeModel.pth')
        print('epoch:{} loss:{:7f}'.format(epoch, loss.item()))
        scheduler.step()
        model.train(False)
        for (i, testData) in enumerate(testLoader):
            testData = testData.cuda()
            outputs = model(testData, True)
            plt.figure(1)
            testData = testData.to('cpu')
            outputs = outputs.to('cpu')
            plt.imshow((torchvision.utils.make_grid(outputs).permute((1, 2, 0))).detach().numpy())
            plt.show()
            break
        model.train(True)
    return model


#数据压缩
def datacompress(testMnist,input):
    model = EncodeModel()
    model.load_state_dict(torch.load('EncodeModel.pth'))
    model.train(False)
    criterion = nn.MSELoss()




# 以图搜图函数
def search_by_image(testMnist, inputImage, K=5):
    model = EncodeModel()
    model.load_state_dict(torch.load('EncodeModel.pth'))
    model.train(False)
    criterion = nn.MSELoss()
    testLoader = DataLoader(testMnist, batch_size=1, shuffle=False)
    inputImage = inputImage.unsqueeze(0)
    inputEncode = model(inputImage, False)
    lossList = []
    for (i, testImage) in enumerate(testLoader):
        testEncode = model(testImage, False)
        enLoss = criterion(inputEncode, testEncode)
        lossList.append((i, enLoss.item()))
    lossList = sorted(lossList, key=lambda x: x[1], reverse=False)[:K]
    plt.figure(1)
    trueImage = inputImage.squeeze(0).squeeze(0)
    plt.imshow(trueImage, cmap='gray', shape=(28, 28))
    plt.title('true')
    plt.show()
    for j in range(K):
        showImage = testMnist[lossList[j][0]]
        showImage = showImage.squeeze(0)
        showImage = np.array(showImage)
        plt.subplot(1, 5, j + 1)
        plt.imshow(showImage, cmap='gray')
    plt.title('sim')
    plt.show()


if __name__ == '__main__':
    trainLoader, testLoader, trainMnist, testMnist = load_mnist()
    # model=train(trainLoader,testLoader)
    i = 0
    for inputImage in testMnist:
        search_by_image(testMnist, inputImage)
        i += 1
        if i > 200:
            break
