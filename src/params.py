"""Params for AGDA."""

# params for dataset and data loader
data_root = "data"
# TODO
# batch_size must equal fusion_size for early fusion model
batch_size = 128
fusion_size = 256
image_size = 64
dim_set = [100, 400, 800]

# params for target dataset
tgt_dataset = "USPS"
tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
tgt_model_trained = True

# params for data set selection
source_data_set = "mnist"
target_data_set = "svhn"

# params for model training
clf_train_iter = 1          # Number of iterations to train the source domain AE + CLF
tag_train_iter = 1          # Number of iterations to perform the distribution fusion steps (One fusion each iteration)

fusion_steps = 10           # Parameter for early fusion version -- SUSPENDED

g_steps = 1                 # Number of training step performed to Generator
g_learning_rate = 1e-4      # Learning rate of each G update
d_steps = 2                 # Number of training step performed to Discriminator
d_learning_rate = 1e-4      # Learning rate of each D update
clf_learning_rate = 1e-3    # Learning rate of source classifier

num_disable_layer = 1       # Number of layer of target generator that is not trainable (share weight)

source_ae_weight = 1.       # The weight of ae loss during source training process
source_clf_weight = 1.      # The weight of clf loss during source training process
target_ae_weight = 1.       # The weight of ae loss during target training process
target_fusion_weight = 1.   # The weight of domain fusion loss during target training process


