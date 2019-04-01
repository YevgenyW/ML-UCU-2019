from src.constants import *
from src.imports import *
from src.utils import *

def augmentationImage(path, augmentations = True):
	ORIGINAL_SIZE = 96      # original size of the images - do not change

	# AUGMENTATION VARIABLES
	RANDOM_ROTATION = 3    # range (0-180), 180 allows all rotation variations, 0=no change
	RANDOM_SHIFT = 2        # center crop shift in x and y axes, 0=no change. This cannot be more than (ORIGINAL_SIZE - CROP_SIZE)//2 
	RANDOM_BRIGHTNESS = 7  # range (0-100), 0=no change
	RANDOM_CONTRAST = 5    # range (0-100), 0=no change
	RANDOM_90_DEG_TURN = 1  # 0 or 1= random turn to left or right

    # augmentations parameter is included for counting statistics from images, where we don't want augmentations

	rgb_img = readImage(path)
    
	if(not augmentations):
		return rgb_img / 255
    
	#random rotation
	rotation = random.randint(-RANDOM_ROTATION,RANDOM_ROTATION)
	if(RANDOM_90_DEG_TURN == 1):
	    rotation += random.randint(-1,1) * 90
	M = cv2.getRotationMatrix2D((48,48),rotation,1)   # the center point is the rotation anchor
	rgb_img = cv2.warpAffine(rgb_img,M,(96,96))

	#random x,y-shift
	x = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)
	y = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)

	# crop to center and normalize to 0-1 range
	start_crop = (ORIGINAL_SIZE - CROP_SIZE) // 2
	end_crop = start_crop + CROP_SIZE
	rgb_img = rgb_img[(start_crop + x):(end_crop + x), (start_crop + y):(end_crop + y)] / 255

	# Random flip
	flip_hor = bool(random.getrandbits(1))
	flip_ver = bool(random.getrandbits(1))
	if(flip_hor):
	    rgb_img = rgb_img[:, ::-1]
	if(flip_ver):
	    rgb_img = rgb_img[::-1, :]
	    
	# Random brightness
	br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.
	rgb_img = rgb_img + br

	# Random contrast
	cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.
	rgb_img = rgb_img * cr

	# clip values to 0-1 range
	rgb_img = np.clip(rgb_img, 0, 1.0)

	return rgb_img

def printRandomCroppedImages(data):
	fig, ax = plt.subplots(2,5, figsize=(20,8))
	fig.suptitle('Cropped histopathologic scans of lymph node sections',fontsize=20)
	# Negatives
	for i, idx in enumerate(data[data['label'] == 0]['id'][:5]):
	    path = os.path.join(train_path, idx)
	    ax[0,i].imshow(augmentationImage(path + '.tif'))
	ax[0,0].set_ylabel('Negative samples', size='large')
	# Positives
	for i, idx in enumerate(data[data['label'] == 1]['id'][:5]):
	    path = os.path.join(train_path, idx)
	    ax[1,i].imshow(augmentationImage(path + '.tif'))
	ax[1,0].set_ylabel('Tumor tissue samples', size='large')

# As we count the statistics, we can check if there are any completely black or white images
def findOutlierImages(data):
	dark_th = 10 / 255      # If no pixel reaches this threshold, image is considered too dark
	bright_th = 245 / 255   # If no pixel is under this threshold, image is considerd too bright
	too_dark_idx = []
	too_bright_idx = []

	x_tot = np.zeros(3)
	x2_tot = np.zeros(3)
	counted_ones = 0
	for i, idx in tqdm_notebook(enumerate(data['id']), 'computing statistics...(220025 it total)'):
		path = os.path.join(train_path, idx)
		imagearray = augmentationImage(path + '.tif', augmentations = False).reshape(-1,3)
		# is this too dark
		if(imagearray.max() < dark_th):
		    too_dark_idx.append(idx)
		    continue # do not include in statistics
		# is this too bright
		if(imagearray.min() > bright_th):
		    too_bright_idx.append(idx)
		    continue # do not include in statistics
		x_tot += imagearray.mean(axis=0)
		x2_tot += (imagearray**2).mean(axis=0)
		counted_ones += 1
	    
	channel_avr = x_tot/counted_ones
	channel_std = np.sqrt(x2_tot/counted_ones - channel_avr**2)
	channel_avr,channel_std

	print('There was {0} extremely dark image'.format(len(too_dark_idx)))
	print('and {0} extremely bright images'.format(len(too_bright_idx)))
	print('Dark one:')
	print(too_dark_idx)
	print('Bright ones:')
	print(too_bright_idx)

	return [too_bright_idx, too_dark_idx]


def printOutlierStatistics(data, too_bright_idx, too_dark_idx): 
	fig, ax = plt.subplots(2,6, figsize=(25,9))
	fig.suptitle('Almost completely black or white images',fontsize=20)
	# Too dark
	i = 0
	for idx in np.asarray(too_dark_idx)[:min(6, len(too_dark_idx))]:
	    lbl = data[data['id'] == idx]['label'].values[0]
	    path = os.path.join(train_path, idx)
	    ax[0,i].imshow(augmentationImage(path + '.tif', augmentations = False))
	    ax[0,i].set_title(idx + '\n label=' + str(lbl), fontsize = 8)
	    i += 1
	ax[0,0].set_ylabel('Extremely dark images', size='large')
	for j in range(min(6, len(too_dark_idx)), 6):
	    ax[0,j].axis('off') # hide axes if there are less than 6
	# Too bright
	i = 0
	for idx in np.asarray(too_bright_idx)[:min(6, len(too_bright_idx))]:
	    lbl = data[data['id'] == idx]['label'].values[0]
	    path = os.path.join(train_path, idx)
	    ax[1,i].imshow(augmentationImage(path + '.tif', augmentations = False))
	    ax[1,i].set_title(idx + '\n label=' + str(lbl), fontsize = 8)
	    i += 1
	ax[1,0].set_ylabel('Extremely bright images', size='large')
	for j in range(min(6, len(too_bright_idx)), 6):
	    ax[1,j].axis('off') # hide axes if there are less than 6


def split(data, test_size=0.1) :
	# we read the csv file earlier to pandas dataframe, now we set index to id so we can perform
	train_df = data.set_index('id')

	#If removing outliers, uncomment the four lines below
	#print('Before removing outliers we had {0} training samples.'.format(train_df.shape[0]))
	#train_df = train_df.drop(labels=too_dark_idx, axis=0)
	#train_df = train_df.drop(labels=too_bright_idx, axis=0)
	#print('After removing outliers we have {0} training samples.'.format(train_df.shape[0]))

	train_names = train_df.index.values
	train_labels = np.asarray(train_df['label'].values)

	# split, this function returns more than we need as we only need the validation indexes for fastai
	tr_n, tr_idx, val_n, val_idx = train_test_split(train_names, range(len(train_names)), test_size=test_size, stratify=train_labels, random_state=123)

	return val_idx

def getDataBunchForFastAI(data):

	arch = densenet169                  # specify model architecture, densenet169 seems to perform well for this data but you could experiment
	BATCH_SIZE = 128                    # specify batch size, hardware restrics this one. Large batch sizes may run out of GPU memory
	sz = CROP_SIZE                      # input size is the crop size
	MODEL_PATH = str(arch).split()[1]   # this will extrat the model name as the model file name e.g. 'resnet50'

	# we read the csv file earlier to pandas dataframe, now we set index to id so we can perform
	train_df = data.set_index('id')

	#If removing outliers, uncomment the four lines below
	#print('Before removing outliers we had {0} training samples.'.format(train_df.shape[0]))
	#train_df = train_df.drop(labels=too_dark_idx, axis=0)
	#train_df = train_df.drop(labels=too_bright_idx, axis=0)
	#print('After removing outliers we have {0} training samples.'.format(train_df.shape[0]))

	train_names = train_df.index.values
	train_labels = np.asarray(train_df['label'].values)

	# split, this function returns more than we need as we only need the validation indexes for fastai
	tr_n, tr_idx, val_n, val_idx = train_test_split(train_names, range(len(train_names)), test_size=0.1, stratify=train_labels, random_state=123)

	# create dataframe for the fastai loader
	train_dict = {'name': train_path + train_names, 'label': train_labels}
	df = pd.DataFrame(data=train_dict)
	# create test dataframe
	test_names = []
	for f in os.listdir(test_path):
	    test_names.append(test_path + f)
	df_test = pd.DataFrame(np.asarray(test_names), columns=['name'])

	# Subclass ImageList to use our own image opening function
	class MyImageItemList(ImageList):
		def open(self, fn:PathOrStr)->Image:
			img = augmentationImage(fn.replace('/./','').replace('//','/'))
			# This ndarray image has to be converted to tensor before passing on as fastai Image, we can use pil2tensor
			return vision.Image(px=pil2tensor(img, np.float32))

	imgDataBunch = (MyImageItemList.from_df(path='./', df=df, suffix='.tif')
        #Where to find the data?
        .split_by_idx(val_idx)
        #How to split in train/valid?
        .label_from_df(cols='label')
        #Where are the labels?
        .add_test(MyImageItemList.from_df(path='./', df=df_test))
        #dataframe pointing to the test set?
        .transform(tfms=[[],[]], size=sz)
        # We have our custom transformations implemented in the image loader but we could apply transformations also here
        # Even though we don't apply transformations here, we set two empty lists to tfms. Train and Validation augmentations
        .databunch(bs=BATCH_SIZE)
        # convert to databunch
        .normalize([tensor([0.702447, 0.546243, 0.696453]), tensor([0.238893, 0.282094, 0.216251])])
        # Normalize with training set stats. These are means and std's of each three channel and we calculated these previously in the stats step.
       )

	return imgDataBunch
    
    


