import os
import random

test_size = 100

path = "data/selected/gt.txt"
assert os.path.exists(path)
with open(path, "r") as annotations:
    train = open("MS_LSTM/data/train_dataset.txt", 'w')
    test = open("MS_LSTM/data/test_dataset.txt", 'w')
    lines = list(annotations)
    random.shuffle(lines)

    for i, line in enumerate(lines):
        name, mark = line.strip().split(',')
        print(name.split('-')[1], mark, sep=',', file=test if i < test_size else train)
