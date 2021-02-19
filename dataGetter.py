import numpy as np
import random
import args

class DataGetter:

    def __init__(self, mode='train'):
        self.mode = mode
        args1 = args.args()
        self.datasetPath = args1.datasetPath

    def getSinalData(self, path):
        img = np.load(self.datasetPath + path).squeeze()
        if self.mode == 'train':
            seed = random.randint(0, 9)
            img = self.randFlip(img, seed)
        img = self.limitToLenght(img, 80)
        img = self.norm(img)
        return img

    def norm(self, img, max=1000, min=0):
        img[img > max] = max
        img[img < min] = min
        img = (img - min) / (max - min)
        return img

    def limitToLenght(self, data, lenghtTarget):
        lenght = data.shape[0]
        outData = np.zeros((lenghtTarget, data.shape[1], data.shape[2]))
        if lenght <= lenghtTarget:
            outData[0:lenght, :, :] = data
        else:
            outData = data[0:lenghtTarget, :, :]
        return outData

    def randFlip(self, data, seed):
        if seed <= 3:
            return data
        else:
            data1 = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
            for i in range(data.shape[0]):
                data1[i] = data[data.shape[0] - 1 - i]
            return data1

    def getPathsAndLables(self):
        labels = []
        if self.mode == 'train':
            f = open(self.datasetPath + '/train.txt', 'r')
        elif self.mode == 'test':
            f = open(self.datasetPath + '/test.txt', 'r')
        elif self.mode == 'val':
            f = open(self.datasetPath + '/val.txt', 'r')
        paths = f.readlines()

        for i in range(len(paths)):
            paths[i] = paths[i][:-1]
            label = paths[i].split('/')[2]
            if label == 'fracture':
                labels.append(0)
            if label == 'noFracture':
                labels.append(1)
        return paths, labels




