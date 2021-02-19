# for code 
Please modify datasetPath and modelSavePath according your own path in args.py.

Train through train.py.
Validate and test through test.py.



# for dataset 
The dataset can be downloaded in A.

In order to protect personal privacy, all data is desensitized. Each case contains a category label and a numpy array. The label can be obtained from the folder name or txt file. Each number in the numpy array represents the HU (Hounsfield Unit) value in the CT image.
Because the image is too large for training, all CT image slices have been reduced to 256*256, which is consistent with the paper.

Due to different collection batches, the dimensions of the test set numpy array are slightly different. Specifically, all numpy data in the data set has a dimension of size 1, while the position of the dimension in the test set is different.
For all numpy arrays in the data set (including training set, validation set, and test set), please squeeze the dimension of size 1 when you used. This operation will not affect the model effect.


