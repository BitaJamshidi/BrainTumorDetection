#Import necessary libraries
import os
import shutil
import numpy as np
import pandas as pd
import cv2
import keras
import imutils
from sklearn.model_selection import train_test_split
import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Flatten,BatchNormalization,Dropout,Activation,MaxPool2D,MaxPooling2D
#from tensorflow.keras.layers import Reshape
#from tensorflow.keras.models import Model
from keras.layers import Conv2D
from keras import Sequential
from keras import regularizers
from keras.regularizers import l2
from keras.optimizers import RMSprop,Adam
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras.metrics import Recall,Precision
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pydotplus import graphviz
from IPython.display import SVG, Image
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, accuracy_score
#from tensorflow.keras.utils import plot_model
from keras import utils

# Define the path to the dataset
IMG_PATH = 'C:/Users/mandish/Desktop/Master Thesis-Bita Jamshidi/Coding/CNN(VGG19) Optimized/'

# Create a list of all the image filenames
all_images = []
for folder in ['yes/', 'no/']:
    folder_path = os.path.join(IMG_PATH, folder)
    for filename in os.listdir(folder_path):
        
        all_images.append(os.path.join(folder_path, filename))

# Create a list of corresponding labels (0 for 'no', 1 for 'yes')
labels = [1 if 'Y' in filename else 0 for filename in all_images]

# Split the dataset into train, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(all_images, labels, test_size=0.03, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=123)

print(f'Train set size: {len(X_train)}')
print(f'Validation set size: {len(X_val)}')
print(f'Test set size: {len(X_test)}')

# Define the path to the dataset
image_size = 224
IMG_PATH = 'C:/Users/mandish/Desktop/Master Thesis-Bita Jamshidi/Coding/CNN(VGG19) Optimized/'

# Define the labels and their corresponding colors
labels = {0: 'No', 1: 'Yes'}
colors = {0: 'blue', 1: 'red'}

# Plot some images from the train set for each label
fig, axs = plt.subplots(2, 3, figsize=(10, 8))
for i, label in enumerate([0, 1]):
    images = [x for x, y in zip(X_train, y_train) if y == label][:3]
    for j, image_path in enumerate(images):
        img = plt.imread(image_path)
        axs[i, j].imshow(img, aspect='auto') # change aspect to 'auto'
        axs[i, j].set_title(labels[label], color=colors[label])
plt.show()

def preprocess_images(images):
    preprocessed_images = []
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        img = cv2.resize(
            img,
            dsize=(224,224),
            interpolation=cv2.INTER_CUBIC
        )
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

 # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        # crop
        ADD_PIXELS = 0
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        new_img = cv2.resize(
            new_img,
            dsize=(224,224))
        preprocessed_images.append(new_img)
    return np.array(preprocessed_images)

# Apply the preprocessing to all the data subsets 
X_train_pre = preprocess_images(X_train)
X_val_pre = preprocess_images(X_val)
X_test_pre = preprocess_images(X_test)

# Transform the subsets to numpy arrays 
X_train_pre_vgg = np.array([preprocess_input(image) for image in X_train_pre])
X_val_pre_vgg = np.array([preprocess_input(image) for image in X_val_pre])
X_test_pre_vgg = np.array([preprocess_input(image) for image in X_test_pre])

# plot some images from X_train_pre
plt.figure(figsize=(5, 5))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train_pre_vgg[i], cmap='gray')
plt.show()

# Load the pre-trained VGG19 model (Model Initialization)
base_model = VGG19(
weights='imagenet', 
include_top=False,
input_shape=(image_size,image_size) + (3,)
)

# Create a new model by adding a few layers on top of the pre-trained model
input_tensor = base_model.output # get the output tensor of the base model
input_tensor = tf.reshape(input_tensor, (32, 7, 7, 512)) # reshape the tensor to match the batch size and the feature map size
model = Sequential()
model.add(base_model)
model.add(Conv2D(64, kernel_size = (3,3), padding='same', activation ='relu', input_shape = (image_size,image_size) + (3,))) 
model.add(MaxPool2D(pool_size=(1,1)))
model.add(Dropout(0.2))
model.add(BatchNormalization())  
model.add(Conv2D(filters = 128, kernel_size = (3,3), activation ='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))  
model.add(MaxPool2D(pool_size=(1,1)))  
model.add(Dropout(0.2))   
model.add(Conv2D(filters = 128, kernel_size = (3,3), activation ='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(MaxPool2D(pool_size=(1,1)))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 128, kernel_size = (3,3), activation ='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(MaxPool2D(pool_size=(1,1)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512,kernel_initializer='glorot_uniform', activation = "relu"))  
model.add(Dropout(0.2))
model.add(BatchNormalization())  
model.add(Flatten())
model.add(Dense(512,kernel_initializer='glorot_uniform', activation = "relu")) 
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Freeze the weights of the pre-trained model:
#  ( This means that the model will be updated during the training process,
#  which can cause overfitting You should set the trainable attribute of the model to False, 
#  so that only the new layers that you added will be trained.)
model.layers[0].trainable = False

# Compile the model with appropriate loss function, optimizer and metrics
model.compile(
optimizer = keras.optimizers.RMSprop(learning_rate=0.00008),
loss='binary_crossentropy',
metrics=['accuracy']
)

# Print the summary of the model
model.summary()

# Save the weights model:
model.save('My_model.keras')

# Define the training data generator with necessary data augmentation techniques
train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
brightness_range=[0.1, 1.5],
horizontal_flip=True,
vertical_flip=False,
fill_mode='nearest'
)

# Create the training data generator using the training dataset and the data generator
train_generator = train_datagen.flow(
X_train_pre_vgg,
y_train,
shuffle=True,
batch_size=32
)

# Define a callback to reduce the learning rate when the validation accuracy plateaus:
#  (using a ReduceLROnPlateau callback to reduce the learning rate when the validation accuracy plateaus.
#  This is a good technique to avoid getting stuck in local minima and improve the convergence of the model.)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-5)
class EvaluationCallback(Callback):
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
    
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val) > 0.5
        recall = recall_score(self.y_val, y_pred)
        precision = precision_score(self.y_val, y_pred)
        print(f' Val Recall: {recall:.4f} - Val Precision: {precision:.4f}')
        
evaluation_callback = EvaluationCallback(X_val, y_val)

# Define a callback to stop the training when validation accuracy reaches 90%:
# (This is a reasonable way to prevent overfitting and save time, but it might
#  also prevent you from reaching higher accuracy levels)
class StopOnAccuracy(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.9:
            print("\nReached 90% accuracy, stopping training...")
            self.model.stop_training = True

# Define the validation data generator with appropriate data preprocessing
# (Importing images using image data preprocessing provided from keras)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.20)

# Convert y_val to one-hot encoded vectors
y_val_categorical = utils.to_categorical(y_val)

# Create the validation data generator
val_generator = val_datagen.flow(
    x=X_val_pre_vgg,
    y=y_val_categorical,
    batch_size=32,
    shuffle=False
)

# Train the model using the training data generator, validation data generator and the defined callbacks
history = model.fit(
train_generator,
epochs=30,
validation_data=val_generator,
callbacks=[reduce_lr,StopOnAccuracy()]
)

# Plotting the Model
#plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
#Image('model.png',width=400, height=200)

# Plot the training and validation accuracy curves
plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.tight_layout()
plt.show()

# Plot the training and validation loss curves
plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.tight_layout()
plt.show()

HistoryDict = history.history
val_losses = HistoryDict["val_loss"]
val_acc = HistoryDict["val_accuracy"]
acc = HistoryDict["accuracy"]
losses = HistoryDict["loss"]
epochs = range(1,len(val_losses)+1)

plt.plot(epochs,val_losses,"k-",label="LOSS")
plt.plot(epochs,val_acc,"r",label="ACCURACY")
plt.title("LOSS & ACCURACY")
plt.xlabel("EPOCH")
plt.ylabel("Loss & Acc")
plt.legend()
plt.show()

plt.plot(epochs,losses,"k-",label="LOSS")
plt.plot(epochs,val_losses,"r",label="LOSS VAL")
plt.title("LOSS & LOSS VAL")
plt.xlabel("EPOCH")
plt.ylabel("LOSS & LOSS VAL")
plt.legend()
plt.show()

Dict_Summary = pd.DataFrame(history.history)
Dict_Summary.plot()

# Plot precision
#plt.figure(figsize=(5, 5))
#plt.subplot(1, 2, 2)
#plt.plot(history.history['precision'])
#plt.plot(history.history['val_precision'])
#plt.title('precision')
#plt.ylabel('precision')
#plt.xlabel('Epoch')
#plt.legend(['train', 'validation'], loc='upper right')
#plt.tight_layout()
#plt.show()

# Plot recall
#plt.figure(figsize=(5, 5))
#plt.subplot(1, 2, 1)
#plt.plot(history.history['recall'])
#plt.plot(history.history['val_recall'])
#plt.title('recall')
#plt.ylabel('recall')
#plt.xlabel('Epoch')
#plt.legend(['train', 'validation'], loc='upper right')
#plt.tight_layout()
#plt.show()

# Evaluate the model on the test set
loss, accuracy, precision, recall = model.evaluate(X_test_pre,y_test)

#Print the evaluation metrics
print('Test Accuracy: %.3f' % accuracy)
print('Test Precision: %.3f' % precision)
print('Test Recall: %.3f' % recall)
print('Test Loss: %.3f' % loss)

# Make predictions on X_test_pre
y_pred = model.predict(X_test_pre)
y_pred = np.round(y_pred).astype(int)

# Give the report
Class_Report =(classification_report(X_test_pre, y_pred, model.predict))
print(Class_Report)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred, X_test_pre, model.predict, normalize="true")
plt.figure(figsize=(10, 10))
sns.heatmap(cm, vmax=1, center=0 ,vmin=-1, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
 
# Calculate accuracy on test set
test_accuracy = accuracy_score(y_test, y_pred, X_test_pre)
print('Test accuracy:', test_accuracy)