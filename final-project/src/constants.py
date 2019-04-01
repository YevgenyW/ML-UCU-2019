from src.imports import *

path_to_train_labels='data/train_labels.csv'
path_to_sample_submission='data/sample_submission.csv'
train_path='data/train/'
test_path='data/test/'

CROP_SIZE = 90          # final size after crop
arch = densenet169                  # specify model architecture, densenet169 seems to perform well for this data but you could experiment
BATCH_SIZE = 128                    # specify batch size, hardware restrics this one. Large batch sizes may run out of GPU memory
sz = CROP_SIZE                      # input size is the crop size
MODEL_PATH = str(arch).split()[1]   # this will extrat the model name as the model file name e.g. 'resnet50'