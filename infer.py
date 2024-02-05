from dataloader import LoadData
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, 0)
    return softmax_x


with open('dir_label.txt', 'r', encoding='utf-8') as f:
    labels = f.readlines()
    labels = list(map(lambda x: x.strip().split('\t'), labels))

if __name__ == "__main__":
    test_list = 'test.txt'
    test_data = LoadData(test_list, train_flag=False)
    test_loader = DataLoader(dataset=test_data, num_workers=1, pin_memory=True, batch_size=1)

    model = models.resnet50(weights=None)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, 214)
    model = model.cuda()

    # 加载训练好的模型
    checkpoint = torch.load('model_best_checkpoint_resnet50_4.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])  # 加载模型参数
    model.eval()

    for i, (image, label) in enumerate(test_loader):
        if i == 27:
            src = image.numpy()
            src = src.reshape(3, 224, 224)
            src = np.transpose(src, (1, 2, 0))
            image = image.cuda()
            label = label.cuda()
            pred = model(image)  # 输出张量
            pred = pred.data.cpu().numpy()[0]  # 张量转化为NumPy数组
            score = softmax(pred)   # 将模型的输出转换为概率分布
            pred_id = np.argmax(score)  # 具有最高概率的类别的索引
            plt.imshow(src)
            print('预测结果：', labels[pred_id][0])
            plt.show()
            break
