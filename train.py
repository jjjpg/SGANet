import torch
import SGANet as module
import os
import dataload
import tools
import time
import args

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
model = module.getNet(in_channel=1).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0006)
args1 = args.args()
batchSize = args1.bactsizeTrain

start = 0
for epoch in range(150):
    loss_sum = 0
    model.train()
    timeStart = time.time()
    for i, (inputs, labels, _) in enumerate(dataload.train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = tools.clip_by_tensor(outputs, batchSize)

        fractureSample = 0
        noFractureSample = 0
        for j in range(labels.size()[0]):  # each batch
            if labels[j] == 0:
                fractureSample += 1
        noFractureSample = labels.size()[0] - fractureSample
        bf = (labels.size()[0] + 0.018) / (fractureSample + 0.018)
        bnf = (labels.size()[0] + 0.018) / (noFractureSample + 0.018)
        outputs = outputs.permute(1, 0)
        outputs = outputs[0].reshape(batchSize, 1)
        labels = labels.reshape(batchSize, 1)
        loss = 0

        for j in range(labels.size()[0]):  # each batch
            if labels[j] == 0:
                loss = loss - bf * torch.log(1 - outputs[j][0])
            else:
                loss = loss - bnf * torch.log(outputs[j][0])
        # print(loss)
        loss.backward()
        optimizer.step()
        loss_sum += loss
    timeEnd = time.time()
    print("nowModel:", epoch + start + 1, "loss_sum:", loss_sum, "runTime:", (timeEnd - timeStart))

    # save
    print('Saving..')
    torch.save(model.state_dict(),
               args1.modelSavePath + "/%d model.pkl" % (epoch + start + 1))
