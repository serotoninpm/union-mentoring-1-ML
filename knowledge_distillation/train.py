# model import
# DataSet prepare
# teather model 불러오기
# student model 선언
# student model 학습
# 학습방법 지정


import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from utils import progress_bar, format_time
import json
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import os
import torch.optim as optim






# argparse
parser = argparse.ArgumentParser(description='Knowledge Distillation')
parser.add_argument('--epochs',  type=int, help='epoch')
parser.add_argument('--batch',  type=int, help='batch')
parser.add_argument('--student',  type=str, help='student model')
parser.add_argument('--loss',  type=str, help='loss function')
args = parser.parse_args()

# search gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = args.epochs
batch_size = args.batch
T = 2
alpha = 0.5
best_acc = 0  # best test accuracy

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

# teacher model
teacher_net = models.resnet50(pretrained=True)

# student_net
if args.student == "resnet18":
    student_net = models.resnet18(pretrained=False)
elif args.student == "resnet34":
    student_net = models.resnet34(pretrained=False)
elif args.student == "resnet50":
    student_net = models.resnet50(pretrained=False)


if torch.cuda.device_count() > 1:
    teacher_net = nn.DataParallel(teacher_net)
    student_net = nn.DataParallel(student_net)

teacher_net.to(device)
student_net.to(device)

student_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student_net.parameters(), lr=0.001)

# if args.loss == "euclidean":
#     teacher_criterion = nn.MSELoss()
# elif args.loss == "kl":
#     teacher_criterion = torch.nn.KLDivLoss(reduction='batchmean')
# elif args.loss == "kl_ce":
#     teacher_criterion = nn.MSELoss()


def loss_fn(outputs, labels, teacher_outputs):
    #student output, label, teacher_output
    if args.loss == "kl":
        loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                                 F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
                  F.cross_entropy(outputs, labels) * (1. - alpha)
    elif args.loss == "euclidean":
        loss = torch.nn.MSELoss()(outputs, teacher_outputs) * alpha + \
                  F.cross_entropy(outputs, labels) * (1. - alpha)
    elif args.loss == "kl_ce":
        loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)*0.5 + \
                    F.cross_entropy(outputs, teacher_outputs) * alpha * 0.5 + F.cross_entropy(outputs, labels) * (1. - alpha)
    return loss

# training
def trainset(epoch_i):
    print('\nEpoch: %d' % epoch_i)
    student_net.train()
    teacher_net.eval()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            teacher_outputs = teacher_net(inputs)

        student_outputs = student_net(inputs)

        optimizer.zero_grad()

        loss = loss_fn(student_outputs, targets, teacher_outputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = student_outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    acc = 100. * correct / total
    return acc



def testset(epoch_i):
    global best_acc
    student_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student_net(inputs)
            loss = student_criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    # Save checkpoint.
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': student_net.state_dict(),
            'acc': acc,
            'epoch': epoch_i,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/resnet.pth')
        best_acc = acc

    return acc


trainset_acc_list = []
testset_acc_list = []

for epoch_i in range(epochs):
    trainset_acc = trainset(epoch_i)
    testset_acc = testset(epoch_i)
    trainset_acc_list.append(trainset_acc)
    testset_acc_list.append(testset_acc)


#json파일로 저장
file_path = "./result/" + args.student + "_" + args.loss + '_' + str(best_acc)+'.json'
data = {}
data['result'] = []
data['result'].append({
    "trainset_acc_list": trainset_acc_list,
    "testset_acc_list": testset_acc_list,
    "best_testset_acc": best_acc
})


with open(file_path, 'w') as outfile:
    json.dump(data, outfile)
