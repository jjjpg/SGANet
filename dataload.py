import torch
from torch.utils.data import Dataset
import dataGetter
import numpy as np
import args

class mydata(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.imgsGetter = dataGetter.DataGetter(mode=self.mode)
        self.imgsPath, self.labels = self.imgsGetter.getPathsAndLables()

    def __getitem__(self, index):
        imgPath = self.imgsPath[index]
        label = self.labels[index]
        img = self.imgsGetter.getSinalData(imgPath)
        imgF = torch.zeros((1, 80, 256, 256))
        imgF[0] = torch.from_numpy(img)
        return imgF, label, imgPath

    def __len__(self):
        return len(self.labels)



seed = 10000001
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
args1 = args.args()

train_dataset = mydata(mode='train')
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args1.bactsizeTrain,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )

test_dataset = mydata(mode='test')
test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args1.bactsizeTest,
        num_workers=8,
        drop_last=False
    )

val_dataset = mydata(mode='val')
val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args1.bactsizeVal,
        num_workers=8,
        drop_last=False
    )
