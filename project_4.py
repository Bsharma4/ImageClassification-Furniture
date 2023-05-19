import pandas as pd
import os
import numpy as np
import requests
from PIL import Image
import concurrent.futures
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from dagster import asset
from pandas import DataFrame
import matplotlib.pyplot as plt


@asset
def load_json_folder() -> DataFrame:
    """
    Load and normalize all JSON files in a given folder.

    Args:
        folder_path (str): The path to the folder containing JSON files.

    Returns:
        pandas.DataFrame: The concatenated and normalized JSON dataset.
    """
    folder_path = 'imaterialist-challenge-furniture-2018'
    # Get a list of all the JSON files in the folder
    json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]

    # Load and normalize each JSON file, then concatenate them into a single DataFrame
    dfs = []
    for file_path in json_files:
        df = pd.read_json(file_path)
        dfs.append(df)
    train_df = pd.concat(dfs, axis=0, ignore_index=True)
    print(train_df.info())
    return train_df

@asset
def normalize_data(load_json_folder: DataFrame):
    """
    Normalize the annotations in the given train dataframe and add image URLs to create a new dataframe.

    Args:
        train_df (pandas.DataFrame): The input train dataframe with a 'annotations' and 'images' column.

    Returns:
        pandas.DataFrame: A new dataframe with the normalized annotations and image URLs.
    """
    train = load_json_folder
    # Normalize train annotations and add image URLs to create a new dataframe
    train_df = pd.json_normalize(train['annotations'])
    train_df['url'] = pd.json_normalize(train['images'])['url']
    train_df['url'] = train_df['url'].str[0].str.strip('[]')
    return train_df


@asset
def create_sample_train_df(normalize_data: DataFrame) -> DataFrame :
    """
    Create a DataFrame with a random sample of images from each of the first 'num_labels' unique labels.

    Args:
        train_df (DataFrame): The training dataset DataFrame.
        num_labels (int): The number of unique label IDs to sample from.
        num_samples (int): The number of images to sample from each label.
        random_state (int): The random seed used for sampling.

    Returns:
        DataFrame: The resulting DataFrame with the sampled images.
    """
    num_labels=7 
    num_samples=10
    random_state=42
    train_df = normalize_data
    unique_labels = sorted(train_df['label_id'].unique())
    unique_labels = unique_labels[:num_labels]

    dfs = []
    for label in unique_labels:
        label_df = train_df.loc[train_df['label_id'] == label]
        sample_df = label_df.sample(n=num_samples, random_state=random_state)
        dfs.append(sample_df)

    result_df = pd.concat(dfs, axis=0)
    sample_train_df = result_df.reset_index(drop=True)
    return sample_train_df

def process_image(url, save_path):
    """
    Download an image from a given URL, resize it to 255x255 pixels using PIL library and save it in the specified
    directory with the specified name.

    Args:
        url: A string representing the URL of the image to be downloaded.
        save_path: A string representing the path where the resized image will be saved.

    Returns:
        None
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Open the image using PIL
            img = Image.open(response.raw)
            # Resize the image to 224x224
            img = img.resize((255, 255))
            # Save the image to the specified path
            img.save(save_path, 'JPEG')
            print('Saved: ', save_path)
        else:
            pass
    except Exception as e:
        pass

def process_image_parallel(row):
    """
    A function that takes a row from a pandas DataFrame containing image URLs, label IDs and image IDs as input and
    processes the image using process_image() function in parallel.

    Args:
        row: A pandas Series representing a single row from the DataFrame.

    Returns:
        None
    """
    url = row['url']
    label_id = row['label_id']
    Image_id = row['image_id']
    process_image(url, f'images/{label_id}_{Image_id}.jpg')


@asset
def download_images(create_sample_train_df : DataFrame) -> int:
    """
    A function that downloads, resizes and saves images from the URLs provided in a pandas DataFrame in parallel
    using thread pool executor.

    Args:
        create_sample_train_df: A pandas DataFrame containing image URLs, label IDs and image IDs.

    Returns:
        1 if the function completes successfully.
    """
    sample_train_df = create_sample_train_df
    # Create a directory to save the images
    os.makedirs('images', exist_ok=True)
    # Create a thread pool executor with the number of available CPUs
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the processing function to the executor for each row in the dataframe
        futures = [executor.submit(process_image_parallel, row) for _, row in sample_train_df.iterrows()]
        # Wait for all futures to complete
        concurrent.futures.wait(futures)
    
    return 1

@asset
def load_images_and_labels(download_images : int) -> tuple:
    """
    Load images and corresponding labels from a folder path.

    Parameters:
        folder_path (str): Path to folder containing image files.

    Returns:
        image_data (numpy array): Array containing image data.
        labels (numpy array): Array containing corresponding labels.
    """
    folder_path = 'images/'
    # Initialize empty lists to hold the image data and corresponding labels
    image_data = []
    labels = []

    # Initialize a counter for each label
    label_counts = {}

    # Loop over all the files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is an image
        if file_name.endswith(".jpg"):
            # Extract the label ID and image ID from the file name
            label_id, image_id = file_name.split("_")
            label_id = int(label_id)
            
            # Check if the label has already reached the limit of 200 images
            if label_counts.get(label_id, 0) >= 100:
                continue
            
            # Load the image file and resize it to a fixed size
            image_path = os.path.join(folder_path, file_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (255, 255))
            
            # Append the image data and label to the lists
            image_data.append(image)
            labels.append(label_id)
            
            # Increment the counter for the label
            label_counts[label_id] = label_counts.get(label_id, 0) + 1

    # Convert the lists to numpy arrays
    image_data = np.array(image_data)
    labels = np.array(labels)

    # Print the shapes of the arrays
    print("Image data shape:", image_data.shape)
    print("Labels shape:", labels.shape)
    return image_data, labels

@asset
def Prepare_Data(load_images_and_labels : tuple) -> tuple:
    """
    Train and return a convolutional neural network model for image classification.

    Returns:
        X_train, X_test, y_train, y_test: Tuple containing data for the train & test set.
    """
    # Define the folder path containing the training images
    image_data, labels = load_images_and_labels
    unique_labels = np.unique(labels)
    print("Number of unique labels:", len(unique_labels))
    max_label = np.max(labels)
    print("Maximum label value:", max_label)
    # Convert the labels to categorical format
    num_classes = max_label + 1
    test_size=0.2 
    X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=test_size, random_state=42)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return X_train, X_test, y_train, y_test    

@asset    
def keras_neural_model(Prepare_Data : tuple, load_images_and_labels: tuple) -> DataFrame:
    """
    Train a convolutional neural network model on a set of image data and labels.
    Check accuracy of model
    Make sample predictions on model
    
    Parameters:
        image_data (numpy array): Array containing image data.
        labels (numpy array): Array containing corresponding labels.
        num_classes (int): Number of classes in the classification task.
        test_size (float): Fraction of data to use for testing (default: 0.2).
        epochs (int): Number of epochs to train the model for (default: 10).
        batch_size (int): Batch size to use for training (default: 32).
        image_size (tuple): Size to which the input images will be resized (default: (255, 255)).

    Returns:
        y_pred (pandas DataFrame): DataFrame containing predicted labels for the test set.
    """
    X_train, X_test, y_train, y_test = Prepare_Data
    num_classes = 8
    epochs=5
    batch_size=32 
    image_size=(255, 255)
    model = create_model((image_size[0], image_size[1], 3), num_classes)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    check_accuracy_and_loss(model, X_test, y_test)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    predict_sample_images(load_images_and_labels, model, num_samples=5)
    return pd.DataFrame(y_pred)

def create_model(image_shape, num_classes):
    """
    Create a convolutional neural network model for image classification.

    Parameters:
        image_shape (tuple): Shape of the input image data.
        num_classes (int): Number of classes in the classification task.

    Returns:
        model (Keras model): The created Keras model.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=image_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def predict_sample_images(load_images_and_labels, model, num_samples=5):
    """
    Predicts the labels of a given number of random sample images using the provided model and displays the predicted labels alongside the true labels.

    Args:
    model (tensorflow.keras.Model): A trained neural network model to use for prediction.
    num_samples (int): Number of random sample images to predict. Default is 5.

    Returns:
    None. Displays the predicted labels and corresponding images using matplotlib.

    Raises:
    None.
    """
    # Load the test images and labels
    images, labels = load_images_and_labels

    # Get the sample image indices
    sample_indices = np.random.choice(len(images), size=num_samples, replace=False)

    # Make predictions for each sample image
    for i in sample_indices:
        # Get the sample image and label
        image = images[i]
        label = labels[i]

        # Reshape the image for prediction
        image = image.reshape(1, 255, 255, 3)

        # Make the prediction using the model
        prediction = model.predict(image)

        # Get the predicted label
        predicted_label = np.argmax(prediction)

        # Display the image and predicted/true labels
        fig, ax = plt.subplots()
        ax.imshow(image.squeeze())
        ax.set_title(f"Predicted: {predicted_label}, True: {label}")
        plt.show()

def check_accuracy_and_loss(model, test_images, test_labels):
    """
    Check the accuracy and loss of a trained model on a set of test images and labels.

    Parameters:
        model (Keras model): Trained Keras model.
        test_images (numpy array): Array containing test image data.
        test_labels (numpy array): Array containing corresponding test labels.

    Returns:
        test_loss (float): Loss of the model on the test set.
        test_acc (float): Accuracy of the model on the test set.
    """
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)

    return test_loss, test_acc