# CancerNet
Backend service for breast cancer detection; specifically, Invasive Ductal
Carcinoma which is the most prevalent type of breast cancer. The tensorflow model has been
trained on 277,524 patches of 50x50 pixel images. The dataset was obtained from
Kaggle [here](https://www.kaggle.com/paultimothymooney/breast-histopathology-images/download).

The dataset distribution is as follows:
* 198,738 negative examples (i.e., no cancer)
* 78,786 positive examples (i.e., indicating breast cancer was found in the
  patch)

The model was trained using a Convuloutional Neural Network, and have reached an
accuracy of %79 after 40 epochs.

![][classification_report]

[classification_report]:https://github.com/moeabdol/cancernet-backend/blob/master/classification_report.png
