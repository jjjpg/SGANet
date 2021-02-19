import torch
import SGANet as model
import dataload
import numpy as np
import os
import args
import sys

args1 = args.args()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def singleTest(path, mode='test'):
    model1 = model.getNet(in_channel=1).cuda()
    model1.load_state_dict(torch.load(path))
    model1.eval()
    accNum_f = 0
    accNum_nf = 0
    if mode == 'test':
        dataloader = dataload.test_loader
        sumN = 88
        sumP = 164
        batchSize = args1.bactsizeTest
    elif mode =='val':
        dataloader = dataload.val_loader
        sumN = 117
        sumP = 227
        batchSize = args1.bactsizeVal
    else:
        print("please choose test or val")
        return

    for i, (inputs, labels, _) in enumerate(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model1(inputs)
        outputs = outputs.data
        outputs = outputs.permute(1, 0)
        outputs = outputs[0].reshape(batchSize, 1)
        labels = labels.reshape(batchSize, 1)
        labels = labels.cpu().data.numpy().astype(np.int32)
        outputs = outputs.cpu().data.numpy()

        for j in range(batchSize):
            print(labels[j], outputs[j][0])
            if labels[j] == 1 and outputs[j][0] >= 0.5:
                accNum_nf += 1
            elif labels[j] == 0 and outputs[j][0] < 0.5:
                accNum_f += 1

    # print(accNum_f, accNum_nf)
    acc = (accNum_nf + accNum_f) / (sumP + sumN)
    precision = accNum_f * 1.0 / (accNum_f + sumP - accNum_nf)
    specificity = accNum_nf * 1.0 / sumP
    sensitivity = accNum_f * 1.0 / sumN
    f1_score = (2 * precision * sensitivity) / (precision + sensitivity)
    print("accuracy:", acc, ", precision:", precision, ", specificity:", specificity,
          ", sensitivity:", sensitivity, ", f1-score:", f1_score)


def multiTest(mode='test'):   # test or val
    rootPath = args1.modelSavePath
    modelFile = sorted(os.listdir(rootPath))
    for mmm in range(len(modelFile)):
        which = modelFile[mmm].split(" ")[0]
        if int(which) < 100:
            continue
        path = rootPath + modelFile[mmm]
        singleTest(path, mode)


singleTest(sys.path[0] + "/model_weight_best.pkl", mode='test')