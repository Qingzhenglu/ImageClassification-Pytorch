from dataloader import LoadData
import time
import torch
from torch import nn, optim
from torchvision import models
from torch.utils.data import DataLoader
import shutil

from tensorboardX import SummaryWriter


def accuracy(output, target, topk=(1,)):
    """
        计算 准确率（模型正确预测的样本数与总样本数之比）
    """
    with torch.no_grad():   # 禁用梯度计算
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        class_to = pred[0].cpu().numpy()

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, class_to


def save_checkpoint(state, is_best, filename='checkpoint_4.pth.tar'):
    """
    保存模型策略： 根据is_best，保存valid acc 最好的模型
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_' + filename)


def train(train_dataloader, model, loss_fn, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 训练模式
    model.train()

    end = time.time()
    # input ： img , target : label
    for i, (input, target) in enumerate(train_dataloader):
        # 计算数据加载时间
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # 预测label , 计算loss
        output = model(input)
        loss = loss_fn(output, target)

        # 计算准确率 and 记录loss
        [prec1, prec5], class_to = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        optimizer.zero_grad()   # 梯度清零
        loss.backward()  # 计算损失函数的梯度
        optimizer.step()

        # 计算一个阶段经历的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_dataloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    writer.add_scalar('loss/train_loss', losses.val, global_step=epoch)


def validate(val_dataloader, model, loss_fn, epoch, writer, phase="VAL"):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 评估模式
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_dataloader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input)
            loss = loss_fn(output, target)

            # measure accuracy and record loss
            [prec1, prec5], class_to = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test-{0}: [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    phase, i, len(val_dataloader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1, top5=top5))

        print(' * {} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(phase, top1=top1, top5=top5))
    writer.add_scalar('loss/valid_loss', losses.val, global_step=epoch)
    return top1.avg, top5.avg


class AverageMeter(object):
    """
        计算单个变量的算术平均值
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    batch_size = 32
    epochs = 30
    num_classes = 214

    # 给训练集和测试集分别创建一个数据集加载器
    train_data = LoadData("dataSet/train.txt", True)
    valid_data = LoadData("dataSet/val.txt", False)
    train_dataloader = DataLoader(dataset=train_data, num_workers=4, pin_memory=True, batch_size=batch_size,
                                  shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_data, num_workers=4, pin_memory=True, batch_size=batch_size)
    train_data_size = len(train_data)
    print('训练集数量：%d' % train_data_size)
    valid_data_size = len(valid_data)
    print('验证集数量：%d' % valid_data_size)

    # 定义网络
    model = models.resnet50(weights='DEFAULT')
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, num_classes)

    if torch.cuda.is_available():
        model = model.cuda()

    # 定义损失函数，计算相差多少，交叉熵，
    loss_fn = nn.CrossEntropyLoss().cuda()

    # 定义优化器，用来训练时候优化模型参数，优化器选择 Adam，并采用 StepLR 进行学习率衰减。
    lr_init = 0.0001  # 初始学习率
    lr_stepsize = 5
    weight_decay = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)

    writer = SummaryWriter('runs/resnet50_4')
    # 训练
    best_prec1 = 0
    for epoch in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer, epoch, writer)
        # 在验证集上测试效果
        valid_prec1, valid_prec5 = validate(valid_dataloader, model, loss_fn, epoch, writer, phase="VAL")
        scheduler.step()
        is_best = valid_prec1 > best_prec1
        best_prec1 = max(valid_prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet50',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best,
            filename='checkpoint_resnet50_4.pth.tar')
    writer.close()
