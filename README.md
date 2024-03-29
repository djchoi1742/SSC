# Multi-modal network to evaluate subscapularis tendon tears

- Objective
   - To develop a multi-modal deep learning algorithm for prediction of subscapularis tendon tears on shoulder X-ray and clinical information

- Network Architecture
   <p align="center">
      <img src="https://user-images.githubusercontent.com/49828672/128472304-1b46338d-985b-4ada-9d7f-28e0bfa3fb46.png" width=80% height=80% img align="center"> 
   </p>

- Grad-CAM (positive & negative sample)
   <table>
      <tr>
         <p align="center">
         <img src="https://user-images.githubusercontent.com/49828672/128472829-80de179f-f8c3-4620-b0a7-fa3b18cca997.png" width=40% height=40%,
              title="Positive sample"> </td>
         <img src="https://user-images.githubusercontent.com/49828672/128472837-fb6b8d8d-b7d0-47f3-9a6d-1f7ea184588d.png" width=40% height=40%,
              title="Negative sample">
         </p>
      </tr>
   </table>


- python scripts
   - enviroment: python 3.6, Tensorflow 1.12
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

###### Reference: Kang, Y., Choi, D., Lee, K., Oh, J., Kim, B.R., & Ahn, J. (2021). Evaluating subscapularis tendon tears on axillary lateral radiographs using deep learning. European Radiology 31(12):9408-9417.
