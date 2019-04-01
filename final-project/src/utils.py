from src.constants import *
from src.imports import *

def readImage(path):
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    return rgb_img


def printRandomImages(data):

	fig, ax = plt.subplots(2,5, figsize=(20,8))
	fig.suptitle('Histopathologic scans of lymph node sections',fontsize=20)
	# Negatives
	for i, idx in enumerate(data[data['label'] == 0]['id'][:5]):
	    path = os.path.join('data/train/', idx)
	    ax[0,i].imshow(readImage(path + '.tif'))
	    # Create a Rectangle patch
	    box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='b',facecolor='none', linestyle=':', capstyle='round')
	    ax[0,i].add_patch(box)
	ax[0,0].set_ylabel('Negative samples', size='large')
	# Positives
	for i, idx in enumerate(data[data['label'] == 1]['id'][:5]):
	    path = os.path.join(train_path, idx)
	    ax[1,i].imshow(readImage(path + '.tif'))
	    # Create a Rectangle patch
	    box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='r',facecolor='none', linestyle=':', capstyle='round')
	    ax[1,i].add_patch(box)
	ax[1,0].set_ylabel('Tumor tissue samples', size='large')


def plotWeightDecays(lrs, losses, wds):
	# Plot weight decays
	_, ax = plt.subplots(1,1)
	min_y = 0.5
	max_y = 0.55
	for i in range(len(losses)):
		ax.plot(lrs[i], losses[i])
		min_y = min(np.asarray(losses[i]).min(), min_y)

	ax.set_ylabel("Loss")
	ax.set_xlabel("Learning Rate")
	ax.set_xscale('log')
	#ax ranges may need some tuning with different model architectures 
	ax.set_xlim((1e-3,3e-1))
	ax.set_ylim((min_y - 0.02,max_y))
	ax.legend(wds)
	ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))


def plotOverview():
	# top losses will return all validation losses and indexes sorted by the largest first
    tl_val,tl_idx = interp.top_losses()
    #classes = interp.data.classes
    fig, ax = plt.subplots(3,4, figsize=(16,12))
    fig.suptitle('Predicted / Actual / Loss / Probability',fontsize=20)
    # Random
    for i in range(4):
        random_index = randint(0,len(tl_idx))
        idx = tl_idx[random_index]
        im,cl = interp.data.dl(DatasetType.Valid).dataset[idx]
        im = image2np(im.data)
        cl = int(cl)
        ax[0,i].imshow(im)
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        ax[0,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')
    ax[0,0].set_ylabel('Random samples', fontsize=16, rotation=0, labelpad=80)
    # Most incorrect or top losses
    for i in range(4):
        idx = tl_idx[i]
        im,cl = interp.data.dl(DatasetType.Valid).dataset[idx]
        cl = int(cl)
        im = image2np(im.data)
        ax[1,i].imshow(im)
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])
        ax[1,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')
    ax[1,0].set_ylabel('Most incorrect\nsamples', fontsize=16, rotation=0, labelpad=80)
    # Most correct or least losses
    for i in range(4):
        idx = tl_idx[len(tl_idx) - i - 1]
        im,cl = interp.data.dl(DatasetType.Valid).dataset[idx]
        cl = int(cl)
        im = image2np(im.data)
        ax[2,i].imshow(im)
        ax[2,i].set_xticks([])
        ax[2,i].set_yticks([])
        ax[2,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')
    ax[2,0].set_ylabel('Most correct\nsamples', fontsize=16, rotation=0, labelpad=80)



