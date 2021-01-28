# Multi-modal network to evaluate subscapularis tendon tears
 - Convolutional neural network & Logistic regression for diagnosing mastoiditis
   1. data/setup1.py       - Setting pre-training dataset
    2. data/setup2.py       - Setting main dataset
    3. models/model1.py     - Dense-block based convolutional neural network to train images
   4. models/model2.py     - Multi-modal neural network models (with pre-trained weight, without pre-trained weight) for images and clinical information
   5. models/multi_nn.R    - Fitting Logistic regression to obtain pre-trained weights in multi-modal neural network
   6. runs/main1.py        - Training images for pre-training dataset 
   7. runs/main2.py        - Training images and clinical information with pre-trained weights
   8. runs/main3.py        - Training main dataset without pre-trained weights
   9. tf_utils/tboard.py   - Related to Tensorboard output
