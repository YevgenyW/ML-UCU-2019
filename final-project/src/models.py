from src.constants import *
from src.imports import *
from src.utils import *
from src.preprocessing import *

# arch = densenet169                  # specify model architecture, densenet169 seems to perform well for this data but you could experiment
# BATCH_SIZE = 128                    # specify batch size, hardware restrics this one. Large batch sizes may run out of GPU memory
# sz = CROP_SIZE                      # input size is the crop size
# MODEL_PATH = str(arch).split()[1]   # this will extrat the model name as the model file name e.g. 'resnet50'


def getLearner(imgDataBunch):
    return create_cnn(imgDataBunch, arch, pretrained=True, path='.', metrics=accuracy, ps=0.5, callback_fns=ShowGraph)

def run_one_cycle_policy(learner, imgDataBunch):
	# We can use lr_find with different weight decays and record all losses so that we can plot them on the same graph
	# Number of iterations is by default 100, but at this low number of itrations, there might be too much variance
	# from random sampling that makes it difficult to compare WD's. I recommend using an iteration count of at least 300 for more consistent results.
	lrs = []
	losses = []
	wds = []
	iter_count = 600

	# WEIGHT DECAY = 1e-6
	learner.lr_find(wd=1e-6, num_it=iter_count)
	lrs.append(learner.recorder.lrs)
	losses.append(learner.recorder.losses)
	wds.append('1e-6')
	learner = getLearner(imgDataBunch) #reset learner - this gets more consistent starting conditions

	# WEIGHT DECAY = 1e-4
	learner.lr_find(wd=1e-4, num_it=iter_count)
	lrs.append(learner.recorder.lrs)
	losses.append(learner.recorder.losses)
	wds.append('1e-4')
	learner = getLearner(imgDataBunch) #reset learner - this gets more consistent starting conditions

	# WEIGHT DECAY = 1e-2
	learner.lr_find(wd=1e-2, num_it=iter_count)
	lrs.append(learner.recorder.lrs)
	losses.append(learner.recorder.losses)
	wds.append('1e-2') 

	return [lrs, losses, wds]
