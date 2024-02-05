# Pytorch
try to train the model of Garbage classification...

1. 我使用的垃圾数据集有5万张左右，214个类别
2. 根据数据集的文件目录生成train、val和test三个txt文件，每行有每张图片的路径和类别id
3. 对图像数据进行预处理，例如归一化，标准化，将图片resize相同的尺寸。还有简单的数据增强，例如随机水平或竖直翻转。然后将数据转换成Tensor格式。最后使用torch.utils.data.DataLoader设置一些参数进行数据加载生成我们需要用到的dataset
4. 加载与训练模型（我用的是resnet50,weights='DEFAULT'），定义损失函数，使用Adam优化器，并采用 StepLR 进行学习率衰减。在训练模式训练，评估模式验证，通过模型保存策略保存正确率最高的模型。
5. 使用保存的模型。使用评估模式推断并输出推测的类别
