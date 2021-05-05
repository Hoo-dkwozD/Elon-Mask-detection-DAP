# Project Elon Mask

In this project done under SMU BIA's Data Associates Programme (DAP), my team mates and I trained several Convolutional Neural Network (CNN) models for the purpose of mask detection using this [dataset][kaggle_1] and this [dataset][kaggle_2] found on Kaggle. 

| Class | Number of Images |
| :--- | :--- |
| __With Mask__ | 2544 |
| __Without Mask__ | 2543 |

*Overview of our dataset*

After building and training our own CNNs, we also experimented with the use of Transfer Learning with the pre-trained VGG19 model. 

We mainly prioritize the recall rate on the class without masks as we aim to catch people without mask to the fullest extent possible. 

We then selected the best performing models and using OpenCV, we deployed the model as a real-time video stream mask detection model through a local webcam and also as a web app using TensorflowJS. 

Initially, we experimented with the idea of using an [ensemble method][code_stack] that gathers a few of our best performing models as the final model to be deployed, but the reduction in speed does not make up for the slight improvement in accuracy and thus we decided to just use our best performing model. 

Our models final metric scores are as below: 

| Metrics | Score |
| --- | :--- |
| __Accuracy__ | 97.6 |
| __Recall__ | 99.6 |
| __Precision__ | 95.9 |
| __F1__ | 97.7 |

This repository contains the files for the Development and Deployment process: 

___

## Development

This folder contains the [Jupyter notebook][own_ipynb] and [Python script][own_py] for our own architecture, the [Jupyter notebook][tl_ipynb] and [Python script][tl_py] for our Transfer Learning architecture and the [Jupyter notebook][code_stack] for a code stack storing additonal codes that can either help with deployment or forming the ensemble model. 

Different members had varying implementations of each particular model but we shared and consulted on each others process and progress

___

## Deployment 

This folder contains the [script][local_script] for local deployment and its trained model stored in HDF5 format under the local deployment folder and the [code stack][web] for [web deployment][site] using the ExpressJS framework. 

Different members had varying implementations of each particular deployment but we shared and consulted on each others process and progress

___

[kaggle_1]: https://www.kaggle.com/rakshana0802/face-mask-detection-data
[kaggle_2]: https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset
[code_stack]: ./Development/Back_end_Webcam_Integration_Code_Stack.ipynb
[own_ipynb]: ./Development/Own_Architecture.ipynb
[own_py]: ./Development/Own_Architecture.py
[tl_ipynb]: ./Development/Transfer_Learning.ipynb
[tl_py]: ./Development/Transfer_Learning.py
[local_script]: ./Deployment/Local_deployment/model_compiled.py
[web]: ../../tree/master/Deployment/Web_app_deployment
[site]: https://maskdetection-532df.web.app/
