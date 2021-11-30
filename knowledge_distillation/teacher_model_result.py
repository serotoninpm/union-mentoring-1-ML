# model import
# DataSet prepare
# teather model 불러오기
# teacher model 결과 출력


import torch
import torch.nn as nn
import torchvision.models as models
from utils import progress_bar, format_time
import json
import torchvision.transforms as transforms
import torchvision





# search gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainset = torchvision.datasets.ImageNet('../low_precision/datasets/imagenet', split='train', download=None, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=14, pin_memory=False, drop_last=True)

testset = torchvision.datasets.ImageNet('../low_precision/datasets/imagenet', split='val', download=None, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=14, pin_memory=False)


# pretrained Teacher model
# Model
net = models.resnet50(pretrained=True)

if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)

criterion = nn.CrossEntropyLoss()






def trainset(epoch_i):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    acc = 100. * correct / total
    return acc

def testset(epoch_i):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    acc = 100. * correct / total
    return acc


trainset_acc = trainset(1)
testset_acc = testset(1)


#json파일로 저장
file_path = "./result/" + 'resnet_50_teacher_result.json'
data = {}
data['result'] = []
data['result'].append({
    "trainset_acc": trainset_acc,
    "testset_acc": testset_acc
})


with open(file_path, 'w') as outfile:
    json.dump(data, outfile)