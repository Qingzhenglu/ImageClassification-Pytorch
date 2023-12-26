import time

import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from torch.utils.data import DataLoader

from utils import LoadData

# model = resnet34(weights=None, num_classes=12).to(device)


def test(dataloader, model):
    size = len(dataloader.dataset)
    # 将模型转为验证模式
    model.eval()
    # 初始化test_loss 和 correct， 用来统计每次的误差
    test_loss, correct = 0, 0
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in dataloader:
            # 将数据转到GPU
            X, y = X.to(device), y.to(device)
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            # 计算预测值pred和真实值y的差距
            test_loss += loss_fn(pred, y).item()
            # 统计预测正确的个数
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 返回相应维度的最大值的索引
    test_loss /= size
    correct /= size
    print(f"correct = {correct}, Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    batch_size = 10

    # # 给训练集和测试集分别创建一个数据集加载器
    train_data = LoadData("train.text", True)
    valid_data = LoadData("test.text", False)

    train_dataloader = DataLoader(dataset=train_data, num_workers=4, pin_memory=True, batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=valid_data, num_workers=4, pin_memory=True, batch_size=batch_size)

    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device='cpu'
    print(f"Using {device} device")

    '''# 加载模型
    temp = torch.load(
        "C:\\Users\\motong\\PycharmProjects\\Pytorch\\resnet34-333f7ec4.pth")
    # 加载模型，如果只有数值就只会加载模型数据，如果有字典，则会加载模型数据和字典数据
    model.load_state_dict(temp)'''
    model = resnet34(weights=None, num_classes=12).to(device)
    print(model)

    # 定义损失函数，计算相差多少，交叉熵，
    loss_fn = nn.CrossEntropyLoss()

    # 定义优化器，用来训练时候优化模型参数，随机梯度下降法
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 初始学习率

    # 一共训练1次
    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        """ time_start = time.time()
        train(train_dataloader, model, loss_fn, optimizer)
        time_end = time.time() """
        # print(f"train time: {(time_end-time_start)}")
        time_start = time.time()
        test(test_dataloader, model)
        time_end = time.time()
        print(f"train time: {(time_end - time_start)}")
    print("Done!")
