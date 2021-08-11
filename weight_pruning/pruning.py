import torch
from models.net import Net
import torch.nn as nn
from utils import progress_bar
import argparse
import json
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='pruning_weight')
parser.add_argument('--percent',  type=float, help='epoch')
parser.add_argument('--batch',  type=int, help='epoch')
parser.add_argument('--save', type=str, help='save_path')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = "./checkpoint/cifar10.pth"

batch_size = args.batch

def cut_node(torch_layer, percent):
    #정렬된 weight(절댓값 기준)
    abs_sort_result = torch.abs(torch_layer).view(-1).sort()
    cut_off_index = int(len(abs_sort_result.values) * percent)
    cut_off_value = abs_sort_result.values[cut_off_index-1]
    return torch.where(torch.abs(torch_layer) > cut_off_value, torch_layer, torch.zeros(torch_layer.size(), dtype=torch.float32).cuda())


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=14, pin_memory=False)

testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=14, pin_memory=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
net = Net()
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)
checkpoint = torch.load(PATH)
net.load_state_dict((checkpoint['net']))

criterion = nn.CrossEntropyLoss()

percent = args.percent
for name, param in net.named_parameters():
    if "weight" in name:
        cut = cut_node(param, percent)
        net.state_dict()[name].data.copy_(cut)


global_parameters_count = 0
global_zero_parameters_count = 0
for name, param in net.named_parameters():
    if "weight" in name:
        loclal_parameters_count = param.nelement()
        local_zero_parameters_count = torch.sum(param == 0)
        global_parameters_count += loclal_parameters_count
        global_zero_parameters_count += local_zero_parameters_count

global_zero_parameters_count = int(global_zero_parameters_count)


def trainset():
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

    return 100. * correct / total


def testset():
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
    return 100. * correct / total


trainset_acc = trainset()
testset_acc = testset()

#json파일로 저장
file_path = args.save + "/" + str(percent)+'.json'
data = {}
data['result'] = []
data['result'].append({
    "percent": args.percent,
    "trainset_acc": trainset_acc,
    "testset_acc": testset_acc,
    "zero_params": global_zero_parameters_count,
    "all_params": global_parameters_count
})


with open(file_path, 'w') as outfile:
    json.dump(data, outfile)


# 저장 해야하는 내용
'''
1. pruning percent
2. trainset_acc
3. testset_acc
4. all_param
5. zero_param

'''