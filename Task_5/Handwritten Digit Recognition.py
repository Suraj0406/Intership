# ==========================================
# Handwritten Digit Recognition using CNN
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report


# ==========================================
# Load MNIST Dataset
# ==========================================

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


# ==========================================
# Data Preprocessing
# ==========================================

# Normalize pixel values (0–255 → 0–1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN (add channel dimension)
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# Convert labels to categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# ==========================================
# Build CNN Model
# ==========================================

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))


# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ==========================================
# Train Model
# ==========================================

history = model.fit(
    X_train,
    y_train_cat,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)


# ==========================================
# Evaluate Model
# ==========================================

loss, accuracy = model.evaluate(X_test,y_test_cat)

print("\nTest Accuracy:", accuracy)


# ==========================================
# Predictions
# ==========================================

predictions = model.predict(X_test)

predicted_labels = np.argmax(predictions,axis=1)


# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report:")
print(classification_report(y_test,predicted_labels))


# ==========================================
# Visualize Predictions
# ==========================================

plt.figure(figsize=(10,5))

for i in range(10):
    
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[i].reshape(28,28),cmap="gray")
    
    plt.title("Pred: "+str(predicted_labels[i]))
    plt.axis("off")

plt.show()


# ==========================================
# Save Model
# ==========================================

model.save("digit_recognition_model.h5")

print("\nModel Saved Successfully!")