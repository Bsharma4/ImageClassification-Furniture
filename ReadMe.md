Automatic Classification of Furniture and Home Goods Images using Machine Learning
By:- Brijesh Sharma, 001193186

### Introduction
As more people shift to online shopping, there is an increasing need to have products in photos classified automatically. However, automatic product recognition is challenging because a single product can have different pictures taken in varying lighting, angles, backgrounds, and levels of occlusion. Meanwhile, different fine-grained categories can look similar, making it difficult for general-purpose recognition machines to perceive subtle differences between photos, even though these differences could be important for shopping decisions. Therefore, the objective of this project is to develop a machine learning model that can accurately classify furniture and home goods images.

We plan to use a deep learning-based approach to tackle this problem. Specifically, we will implement a convolutional neural network (CNN) that can learn features from the images and classify them into relevant categories. This project aims to build a computer vision model to classify images of various categories. The images dataset is located in the "images" folder, and the goal is to classify each image into one of the categories represented by its label.
Data description
The dataset we will use for this project is the iMaterialist Furniture and Home Goods dataset[2] available on Kaggle[1]. The dataset includes images from 128 furniture and home goods classes with one ground truth label for each image. The training dataset available online includes a total of 194,828 images, while the validation set has 6,400 images, and the testing set has 12,800 images. The data is provided in JSON format, and for each image, only the URL is provided. Therefore, we will download the images ourselves in a directory and preprocess them by resizing them to a standard size and normalizing them to have zero mean and unit variance.
For this project we use the first seven labels and use pandas to load & normalize the JSON dataset, and then create a new DataFrame with the normalized train annotations and image URLs. It selects a sample of 100 images for each of the first seven unique label values and downloads, resizes, and saves the images to a specified directory. We iterate the downloading images code in order to achieve the desired number of samples in the directory.The images are processed in parallel using a thread pool executor with the number of available CPUs. The resulting directory has images available related to each label which can then be used for training a computer vision model.  

### Approach
This project aims to build a computer vision model to classify images of various categories using Dagster. The images dataset is located in the "images" folder, and the goal is to classify each image into one of the categories represented by its label.

### Data Downloading & Preprocessing

 Dataset Description

All the data described below are txt files in JSON format.

Overview

train.json: training data with image urls and labels

validation.json: validation data with the same format as train.json

test.json: images of which the participants need to generate predictions. Only image URLs are provided.

sample_submission_randomlabel.csv: example submission file with random predictions to illustrate the submission file format

Training Data

The training dataset includes images from 128 furniture and home goods classes with one ground truth label for each image. It includes a total of 194,828 images for training and 6,400 images for validation and 12,800 images for testing.
Train and validation sets have the same format as shown below:

{

"images" : [image],

"annotations" : [annotation],

}

image{

"image_id" : int,

"url": [string]

}

annotation{

"image_id" : int,

"label_id" : int

}

Note that for each image, we only provide URL instead of the image content. Users need to download the images by themselves. Note that the image urls may become unavailable over time. Therefore we suggest that the participants start downloading the images as early as possible. We are considering image hosting service in order to handle unavailable URLs. We'll update here if that could be finalized.

This year, we omit the names of the labels to avoid hand labeling the test images.

Testing data and submissions

The testing data only has images as shown below:

{

"images" : [image],

}

image {

"image_id" : int,

"url" : [string],

}

In this section, the code reads the 'train.json' file, which contains metadata about furniture images. It then normalizes the annotations and images URL data and selects the first seven unique label IDs. Next, the code creates an empty list and loops over each unique label value. Within the loop, it filters the original DataFrame for the given label value, selects random 100 rows to download images using the sample() method, and appends the resulting DataFrame to the list. Finally, it concatenates all the DataFrames in the list vertically to create the final DataFrame and resets its index. The resulting DataFrame is saved as sample_train_df.[3]

Next, a directory is created to save the images, and a function is defined to download, resize, and save the images. The function downloads the image from the specified URL, resizes it to 255x255 pixels using the Python Imaging Library (PIL), and saves it to the specified path in JPEG format. The process_image_parallel() function is defined to parallel process the process_image() function on each row in the DataFrame using the concurrent.futures module.

In this section, the code defines a function to load the images and their corresponding labels from the specified folder path. It initializes empty lists to hold the image data and corresponding labels, a dictionary to hold the count of each label, and a counter for each label. Then, it loops over all the files in the folder, extracts the label ID and image ID from the file name, checks if the label has already reached the limit of 300 images, loads the image file, and resizes it to a fixed size. If the label has not reached its limit, it appends the image data and label to the lists and increments the counter for the label. Finally, it converts the lists to numpy arrays and returns them.

### Model Architecture
After loading the images and their corresponding labels, the code splits the data into training and testing sets using the train_test_split() function from the sklearn.model_selection module. It then converts the labels to categorical format using the to_categorical() function from the keras.utils module. Next, it initializes a sequential model and adds a convolutional layer with 32 filters, a kernel size of 3x3, and a rectified linear unit (ReLU) activation function. It then adds a max pooling layer with a pool size of 2x2 and a dropout layer with a rate of 0.25 to reduce overfitting. The process is repeated, increasing the number of filters to 64 and adding another convolutional, max pooling, and dropout layer. Next, it flattens the output of the previous layer and adds two dense layers with 128 units, a ReLU activation function, and a dropout rate of 0.5 to further reduce overfitting. Finally, it adds a dense output layer with a softmax activation function, which outputs a probability distribution over the seven possible classes. The model is then compiled with a categorical cross-entropy loss function, an adaptive moment estimation (Adam) optimizer, and accuracy metrics.

The model is trained on the training data for 20 epochs with a batch size of 32 using the fit() method. After training, the model is evaluated on the testing data using the evaluate() method. Finally, the code plots the training and validation accuracy and loss curves using the matplotlib library.

### Results
The model was evaluated on a test dataset, resulting in a test loss of 3.0312907695770264 and a test accuracy of 0.5809524059295654. This means that the model correctly classified 58.10% of the test samples, which is moderately better than random guessing. In this case, the test loss value is relatively high, indicating that there is still room for improvement in the model. While the accuracy value is moderate, it is still below the desired level of performance for many applications. Therefore, further improvements to the model's architecture, hyperparameters, or training process may be necessary to achieve better accuracy.

### Discussion
The results demonstrate the effectiveness of the CNN architecture in accurately classifying images.[8] 

Based on the analysis above, it can be concluded that the model performed moderately well on the test dataset, with a test accuracy of 0.58 and a test loss of 3.03.

One possible reason for the moderate performance could be the limited size of the training dataset, which may have prevented the model from fully learning the underlying patterns and features of the data. Another possible reason could be the choice of hyperparameters, which may not have been optimized for the specific dataset and task.

Despite these limitations, the model still demonstrates promise for predicting the target variable with reasonable accuracy. Further improvements could be made by increasing the size of the training dataset, experimenting with different hyperparameters, and possibly using more advanced neural network architectures.

The current results provide a starting point for further investigation and improvement of the model's accuracy.While the current model did not perform well, it is important to remember that machine learning is an iterative process, and that there are many different techniques and strategies that can be used to improve model performance.

### Resources

[1] Kaggle competition:- iMaterialist Challenge (Furniture) at FGVC5. (2018). Kaggle. Retrieved from https://www.kaggle.com/c/imaterialist-challenge-furniture-2018
[2] Json DataSet Files:- iMaterialist Challenge (Furniture) at FGVC5: Dataset. (2018). Kaggle. Retrieved from https://www.kaggle.com/competitions/imaterialist-challenge-furniture-2018/data
[3] Python Jupyter Notebook:- Brijesh Sharma. (n.d.). Image Classification with CNNs using Keras and TensorFlow. [Jupyter notebook]. Google Drive. Retrieved from https://drive.google.com/file/d/1XqKOz_CcCIy26EEIVU3X-N_vrIZZT9-C/view?usp=sharing
[4] OpenCV:- OpenCV. (2021). OpenCV-Python Tutorials. Retrieved from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_intro/py_intro.html
[5] Keras:- Keras. (2021). Getting started with the Keras Sequential model. Retrieved from https://keras.io/getting-started/sequential-model-guide/
[6] Scikit-learn:- scikit-learn. (2021). Model selection. Retrieved from https://scikit-learn.org/stable/model_selection.html
[7] NumPy:- NumPy. (2021). NumPy User Guide. Retrieved from https://numpy.org/doc/stable/user/index.html
[8] Matplotlib:- Matplotlib. (2021). Pyplot tutorial. Retrieved from https://matplotlib.org/stable/tutorials/introductory/pyplot.html



