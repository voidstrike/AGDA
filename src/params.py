"""Params for AGDA."""

# params for dataset and data loader
data_root = "data"
# TODO
# batch_size must equal fusion_size for early fusion model
batch_size = 128
fusion_size = 256
image_size = 64
dim_set = [-2, -1, 100, 400, 500, 800]

# params for target dataset
tgt_dataset = "USPS"
tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
tgt_model_trained = True

# params for data set selection
source_data_set = "dslr"
target_data_set = "dslr"
input_img_size = 28

# params for model training
clf_train_iter = 1          # Number of iterations to train the source domain AE + CLF
tag_train_iter = 1          # Number of iterations to perform the distribution fusion steps (One fusion each iteration)

fusion_steps = 10           # Parameter for early fusion version -- SUSPENDED

g_steps = 1                 # Number of training step performed to Generator
g_learning_rate = 1e-4      # Learning rate of each G update
d_steps = 1                 # Number of training step performed to Discriminator
d_learning_rate = 1e-4      # Learning rate of each D update
clf_learning_rate = 1e-3    # Learning rate of source classifier

num_disable_layer = 0       # Number of layer of target generator that is not trainable (share weight)

source_ae_weight = 1.       # The weight of ae loss during source training process
source_clf_weight = 1.      # The weight of clf loss during source training process
target_ae_weight = 1.       # The weight of ae loss during target training process
target_fusion_weight = 1.   # The weight of domain fusion loss during target training process

# Default Discriminator Network Setting
DEFAULT_DIS_800 = [800, 500, 500]
DEFAULT_DIS_500 = [500, 200]
DEFAULT_DIS_400 = [400, 400, 200]
DEFAULT_DIS_100 = [100, 50, 16]

DEFAULT_IMG_DIS = [256 * 6 * 6, 4096, 2048]

# Default Classifier Network Setting
DEFAULT_CLF_800 = [800, 500]
DEFAULT_CLF_500 = [500, 500]
DEFAULT_CLF_400 = [400, 200, 100]
DEFAULT_CLF_100 = [100, 64, 32, 16]

DEFAULT_IMG_CLF = [256 * 6 * 6, 4096, 4096]


