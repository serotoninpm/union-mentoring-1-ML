import torch
from models.net import Net
import torch.nn as nn
from utils import progress_bar, format_time
import argparse
import json
import torchvision
import torchvision.transforms as transforms
import time
import torch.optim as optim
import os


parser = argparse.ArgumentParser(description='basic')
parser.add_argument('--epochs',  type=int, help='epoch')
parser.add_argument('--batch',  type=int, help='batch')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = args.epochs
batch_size = args.batch
best_acc = 0  # best test accuracy

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
net = Net()
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


#training
def trainset(epoch_i):
    print('\nEpoch: %d' % epoch_i)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_start_time = time.time()
    pre_precessing_time_once = 0
    pre_precessing_time_total = 0
    cpu_to_gpu_time_total = 0
    forward_time_total = 0
    backward_time_total = 0
    list_iter =[]
    iter_end = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx == 0:
            pre_precessing_time_once = time.time() - train_start_time
            list_iter.append(pre_precessing_time_once)
        if batch_idx != 0:
            iter_time = time.time() - iter_end
            list_iter.append(iter_time)



        # cpu to gpu 시간측정
        cpu_to_gpu_time_start = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        cpu_to_gpu_time_end = time.time()
        cpu_to_gpu_time_total += (cpu_to_gpu_time_end-cpu_to_gpu_time_start)

        # forward time
        forward_start = time.time()
        outputs = net(inputs)
        forward_end = time.time()
        forward_time_total += (forward_end-forward_start)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)

        # backward time
        backward_time_start = time.time()
        loss.backward()
        optimizer.step()
        backward_time_end = time.time()
        backward_time_total += (backward_time_end - backward_time_start)


        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        iter_end = time.time()

    train_time_total = time.time() - train_start_time
    return train_time_total, sum(list_iter), cpu_to_gpu_time_total, forward_time_total, backward_time_total


def testset(epoch_i):
    global best_acc
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

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch_i,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/cifar10.pth')
        best_acc = acc


train_time_total_list =[]
pre_precessing_time_total_list =[]
cpu_to_gpu_time_total_list =[]
forward_time_total_list =[]
backward_time_total_list =[]

for epoch_i in range(epochs):
    train_time_total, pre_precessing_time_total, cpu_to_gpu_time_total, forward_time_total, backward_time_total = trainset(epoch_i)
    testset(epoch_i)
    train_time_total_list.append(train_time_total)
    pre_precessing_time_total_list.append(pre_precessing_time_total)
    cpu_to_gpu_time_total_list.append(cpu_to_gpu_time_total)
    forward_time_total_list.append(forward_time_total)
    backward_time_total_list.append(backward_time_total)




#json파일로 저장
file_path = "./result/" + 'basic_num_worker_16_pin_True.json'
data = {}
data['result'] = []
data['result'].append({
    "all_training_time": format_time(sum(train_time_total_list)),
    "pre_processing_time": format_time(sum(pre_precessing_time_total_list)),
    "cpu_to_gpu_time": format_time(sum(cpu_to_gpu_time_total_list)),
    "forward_time": format_time(sum(forward_time_total_list)),
    "backward_time": format_time(sum(backward_time_total_list)),
    "testset_acc": best_acc
})


with open(file_path, 'w') as outfile:
    json.dump(data, outfile)


# 저장 해야하는 내용
'''
1. all_training time
2. pre-processing time
3. cpu to gpu time
4. forward time
5. back-ward time
6. test_acc
특이사항, traing 할 때의 값이다, test로 acc구할 때 시간은 포함하지 않는다.
'''