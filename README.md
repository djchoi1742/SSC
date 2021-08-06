# Multi-modal network to evaluate subscapularis tendon tears
- data
   1. data/setup1.py       - Setting pre-training dataset
   2. data/setup2.py       - Setting main dataset
   3. models/model1.py     - Dense-block based convolutional neural network to train images
- models
   1. models/model2.py     - Multi-modal neural network models (with pre-trained weight, without pre-trained weight) for images and clinical information
   2. models/multi_nn.R    - Fitting Logistic regression to obtain pre-trained weights in multi-modal neural network
- runs
   1. runs/main1.py        - Training images for pre-training dataset 
   2. runs/main2.py        - Training images and clinical information with pre-trained weights
   3. runs/main3.py        - Training main dataset without pre-trained weights
- tf_utils
   1. tf_utils/tboard.py   - Related to Tensorboard output

### Reference: Kang, Y., Choi, D., Lee, K., Oh, J., Kim, B.R., & Ahn, J. (2021). Evaluating subscapularis tendon tears on axillary lateral radiographs using deep learning. European Radiology, 1 - 10.
