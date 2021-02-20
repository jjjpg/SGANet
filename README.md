# For code 
Please modify datasetPath and modelSavePath according your own path in args.py.

Train through train.py.
Validate and test through test.py.



# For dataset 
The dataset can be downloaded in https://pan.baidu.com/s/1L3P_AZHjplyZJHNw1XLlmw. Passward is o7tw

In order to protect personal privacy, all data is desensitized. Each case contains a category label and a numpy array taken from the dicom file. The label can be obtained from the folder name or txt file. Each number in the numpy array represents the HU (Hounsfield Unit) value in the CT image.
Because the image is too large for training, all CT image slices have been reduced to 256*256, which is consistent with the paper.

