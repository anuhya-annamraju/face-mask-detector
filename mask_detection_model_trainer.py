import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Load and save data and labels
# Current working directory
cwd = os.getcwd()

# Go one folder up
parent_dir = os.path.dirname(cwd)

# Append nested folders
path_face_mask_dataset = os.path.join(parent_dir, "kagglehub", "datasets", "omkargurav", "face_mask_dataset", "versions","1","data")

data, labels = [], []
categories = ["with_mask", "without_mask"]

for category in categories:
    path = os.path.join(path_face_mask_dataset, category)
    label = 0 if category == "with_mask" else 1
    
    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_name))
        resized_img = cv2.resize(img, (128, 128))
        data.append(resized_img)
        labels.append(label)
        

###
# Normalize the data 
data = np.array(data, dtype="float32")
labels = np.array(labels)
print(type(data))
data = data / 255.0

### 
# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

### visualize 
import matplotlib.pyplot as plt
import numpy as np

# Pick one image from your dataset
imgsample = x_train[0]  # Already normalized and shape (128,128,3)

# If you want to show the original image
plt.imshow(imgsample)
plt.title("Original Image")
plt.show()


### (Optional) Data Augmentation 
# Helps prevent overfitting and improves generalization:

# datagen = ImageDataGenerator(
#     rotation_range=15,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# datagen.fit(x_train)


### Build the CNN model
# A simple CNN for face mask detection:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

### Train the mode Without augmentation:

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=10
)

### Evaluate and save the model
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

model.save("face_mask_detector.h5")