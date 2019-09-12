import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
import torchvision as tv
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from ECDataLoader import ECDatas

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


model_path = './weights/AlexNet_epoch50.pth'
BATCH_SIZE = 100
LR = 5e-5
train_EPOCH = 25

def alex(**kwargs):
    model = AlexNet(num_classes=4,**kwargs)
    # model.load_state_dict(torch.load(model_path))
    return model

def save_checkpoint(net,epoch=10):
    torch.save(net.state_dict(), f'./weights/AlexNet_epoch{epoch}.pth')

def getData():  # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        #transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()])
    #trainset = tv.datasets.mnist.MNIST(root='/data/', train=True, transform=transform, download=True)
    #testset = tv.datasets.mnist.MNIST(root='/data/', train=False, transform=transform, download=True)

    trainset = ECDatas(root='./data/', transform=transform)
    testset = ECDatas(root=r'G:\FileRecv\PublicTest', transform=transform)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    return train_loader, test_loader, classes


def train():
    trainset_loader, testset_loader, _ = getData()
    net = alex()
    net.cuda()
    cudnn.benchmark = True
    net.train()
    print(net)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

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
        acc = test(net, testset_loader)
        print('Epoch', epoch, 'test accuracy:%.4f' % acc)
    print('Finished Training')
    save_checkpoint(net, epoch + 1)
    acc = test(net, testset_loader)
    print('Epoch', epoch, '|step ', step, 'loss: %.4f' % loss.item(), 'test accuracy:%.4f' % acc)
    return net


def test(net, testdata):
    with torch.no_grad():
        net.eval()
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

def demo(net, testdata):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()])

    with torch.no_grad():
        net.eval()
        inputs=torch.stack([transform(x) for x in testdata],0)
        inputs = inputs.cuda()
        outputs = net(inputs)
        #print(outputs[0])
        _, predicted = torch.max(outputs, 1)
        if predicted==2 and outputs[0][2]-outputs[0][0]<0.3:
            return 0
        return predicted

def load_build_net():
    net = alex()
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    cudnn.benchmark = True
    return net


if __name__ == '__main__':
    net = train()
    '''net = vgg16()
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    cudnn.benchmark = True
    trainset_loader, testset_loader, _ = getData()
    acc=test(net,testset_loader)
    print('test accuracy:%.4f' % acc)'''