import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
import torchvision as tv
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

model_path = './weights/VGG16_epoch1.pth'
BATCH_SIZE = 32
LR = 0.01
train_EPOCH = 3


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):  # 构造函数
        super(VGG, self).__init__()
        # 网络结构（仅包含卷积层和池化层，不包含分类器）
        self.features = features
        self.classifier = nn.Sequential(  # 分类器结构
            # fc6
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),

            # fc7
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            # fc8
            nn.Linear(4096, num_classes))
        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


# 生成网络每层的信息
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 设定卷积层的输出数量
            conv2d = nn.Conv2d(in_channels, v, 3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  # 返回一个包含了网络结构的时序容器


def vgg16(**kwargs):
    model = VGG(make_layers(cfg, batch_norm=True), **kwargs)
    # model.load_state_dict(torch.load(model_path))
    return model

def save_checkpoint(net,epoch=10):
    torch.save(net.state_dict(), f'./weights/VGG16_epoch{epoch}.pth')

def getData():  # 定义数据预处理
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()])
    trainset = tv.datasets.mnist.MNIST(root='/data/', train=True, transform=transform, download=True)
    testset = tv.datasets.mnist.MNIST(root='/data/', train=False, transform=transform, download=True)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    return train_loader, test_loader, classes


def train():
    trainset_loader, testset_loader, _ = getData()
    net = vgg16()
    net.cuda()
    cudnn.benchmark = True
    net.train()
    print(net)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)

    # Train the model
    for epoch in range(train_EPOCH):
        for step, (inputs, labels) in enumerate(trainset_loader):
            optimizer.zero_grad()  # 梯度清零
            inputs=inputs.cuda()
            labels=labels.cuda()
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            #print(step)
            if step % 20 == 0:
                save_checkpoint(net,epoch)
                print('Epoch', epoch, '|step ', step, '/', len(trainset_loader), 'loss: %.4f' % loss.item())
    print('Finished Training')
    save_checkpoint(net, epoch + 1)
    acc = test(net, testset_loader)
    print('Epoch', epoch, '|step ', step, 'loss: %.4f' % loss.item(), 'test accuracy:%.4f' % acc)
    return net


def test(net, testdata):
    with torch.no_grad():
        correct, total = .0, .0
        print(len(testdata))
        for inputs, labels in testdata:
            net.eval()
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        # net.train()
    return float(correct) / total


if __name__ == '__main__':
    net = train()
    '''net = vgg16()
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    cudnn.benchmark = True
    trainset_loader, testset_loader, _ = getData()
    acc=test(net,testset_loader)
    print('test accuracy:%.4f' % acc)'''