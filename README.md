# Explainable AI for Breast Cancer

This repository contains a Jupyter notebook that demonstrate the implementation of Explainable Artificial Intelligence (XAI) in breast cancer diagnosis. The script and notebook use the Breast Histology Pictures dataset from Kaggle to build a machine learning model that can accurately classify breast cancer diagnoses as either malignant or benign. This dataset, which was collected from Andrew Janowczyk's website and utilized for a data science class at Epidemium, comprises 5547 breast histology photos with a size of 50 × 50 x 3. Classifying malignant pictures (IDC: invasive ductal carcinoma) vs non-IDC images is the objective. 

Through this project, I wanted to build a model that could contribute to the healthcare field using Artificial Intelligence. However, as health is a crucial and delicate matter, healthcare professionals hesitate in using AI to make diagnosis, as AI acts as a black box, providing no explanation for it's predictions. A recent development solves this issue, called Explanable AI (xAI). It is an artificial intelligence in which humans can understand the decisions or predictions made by the AI. It contrasts with the "black box" concept in machine learning where even its designers cannot explain why an AI arrived at a specific decision.

xAI techniques are used to interpret the model's predictions, providing insight into how the model makes its decisions. The XAI techniques used in this project include Local Interpretable Model-Agnostic Explanations (LIME). In this project, I made use of the pretrained ResNet-50 model, a convolutional neural network with 50 layers, pre-trained on more than a million photos from the ImageNet collection. This method of using a pre-trained model to solve a new problem is called Transfer Learning. Transfer Learning's key benefits are reduced training time, improved neural network performance, and a lack of big dataset requirements. I used RMSprop optimizer and categorical cross entropy loss function to train the model.

## Getting Started

To run the implementations, you will need to have Python 3 installed on your machine. You will also need to install the following libraries:

* NumPy
* Pandas
* Matplotlib
* TensorFlow
* Keras
* LIME

You can install these libraries using pip. For example, to install NumPy, you can run the following command:
```
pip install keras
```

Once you have installed the required libraries, you can clone this repository to your local machine using Git. To do this, run the following command:
```
git clone https://github.com/reeba212/xAI_breast_cancer
```

To run the notebook, run the following command:
```
jupyter notebook xai_breast_cancer.ipynb
```

This will open the notebook in your default web browser. You can then run each cell in the notebook to reproduce the results.

## Result

The modified ResNet-50 model achieved training accuracy of 93.8021% and validation accuracy of 81.4414%. This verifies that although the model works better on trained data, it performs decently well on validation data too and can provide explanations for the prediction it makes by highlighting specific areas in the images.

## Conclusion

In the context of diagnosing breast cancer, this project offers a useful illustration of how XAI techniques might be applied to understand machine learning models. You can obtain a greater comprehension of how XAI functions and how it might be used to solve issues in the real world by reading the script and notebook. Now that you know how it works, you can expand on it or use it as a springboard for your own initiatives.
