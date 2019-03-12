"""Params for AGDA."""

# params for dataset and data loader
data_root = "data"
# TODO
# batch_size must equal fusion_size for early fusion model
batch_size = 128
fusion_size = 128
image_size = 64

# params for target dataset
tgt_dataset = "USPS"
tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
tgt_model_trained = True

# params for data set selection
source_data_set = "mnist"
target_data_set = "usps"

# params for model training
clf_train_iter = 200
tag_train_iter = 200

fusion_steps = 1

g_steps = 1
d_steps = 1

num_disable_layer = 0


