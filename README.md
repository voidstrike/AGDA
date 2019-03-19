# Auto Encoder & GAN Domain Adaptation

1. There are two parameters you need to specify when you run the program. They are `source_flag` and `hidden_dim`.
`source_flag` is a boolean parameter s.t. set it to `True` will train the source domain auto encoder & classifier from 
scratch and set it to `False` will let you load the pre-trained model from previous training process. `hidden_dim` 
controls the dimension of hidden code generated, currently it can only be 100 or 400.

2. Run the program via `python3 AGDA.py source_flag hidden_dim`

3. Trained model will be saved in folder 'modeinfo'. For example, if the source domain is `mnist`,
it will create files called `source_ae_mnist.pt` & `source_clf_mnist.pt`

