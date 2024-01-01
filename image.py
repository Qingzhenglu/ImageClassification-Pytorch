import os
import random

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

rootdata = r"垃圾图片库"

class_flag = -1

train_list, val_list, test_list = [], [], []
data_list = []

for a, b, c in os.walk(rootdata):
    for i in range(len(c)):
        data_list.append(os.path.join(a, c[i]))

    for i in range(0, int(len(c) * train_ratio)):
        train_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'  # class_flag 表示分类的类
        train_list.append(train_data)

    for i in range(0, int(len(c) * val_ratio)):
        val_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'
        val_list.append(val_data)

    for i in range(0, int(len(c) * test_ratio)):
        test_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'
        test_list.append(test_data)

    class_flag += 1

random.shuffle(train_list)
random.shuffle(val_list)
random.shuffle(test_list)

with open('train.txt', 'w', encoding='utf-8') as f:
    for train_img in train_list:
        f.write(str(train_img))

with open('val.txt', 'w', encoding='utf-8') as f:
    for val_img in val_list:
        f.write(str(val_img))

with open('test.txt', 'w', encoding='utf-8') as f:
    for test_img in test_list:
        f.write(str(test_img))
