# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from ECDataLoader import ECDatas
from torch.utils.data import DataLoader
from sklearn import metrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 50
num_classes = 5
batch_size = 50
learning_rate = 0.001

class LetNet5(nn.Module):
    def __init__(self, num_clases=10):
        super(LetNet5, self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5),
            nn.BatchNorm2d(120),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(84, num_classes),
            nn.LogSoftmax()
        )

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def getData():
    # MNIST dataset
    '''train_dataset = torchvision.datasets.MNIST(root='/data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='/data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)'''

    transform = transforms.Compose([
        transforms.Resize((28,28)),
        #transforms.RandomHorizontalFlip(),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()])
    # trainset = tv.datasets.mnist.MNIST(root='/data/', train=True, transform=transform, download=True)
    # testset = tv.datasets.mnist.MNIST(root='/data/', train=False, transform=transform, download=True)

    trainset = ECDatas(root='./data/', transform=transform)
    testset = ECDatas(root='./data/', transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=10000, shuffle=False)

    return train_loader,test_loader

def lenet():
    return LetNet5(num_classes).to(device)

def save_checkpoint(net,epoch=10):
    torch.save(net.state_dict(), f'./weights/LeNet_epoch{epoch}.pth')

def train(train_loader):
    model = lenet()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    print(total_step)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 3 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        save_checkpoint(model,epoch)
        test(model,test_loader)
    return model


def test(model,test_loader):
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #matrix = metrics.confusion_matrix(labels.cpu(), predicted.cpu())
            #print(matrix)

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

def demo(model,demodata):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()])
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        images=torch.stack([transform(x) for x in demodata],0)
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        return predicted

def load_build_net(model_path):
    net = lenet()
    net.load_state_dict(torch.load(model_path))
    return net

if __name__ == '__main__':
    train_loader,test_loader=getData()
    net=train(train_loader)
    test(net,test_loader)

