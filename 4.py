# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score

# def read_csv(file_path):
#     data = pd.read_csv(file_path)
#     return data

# def preprocess_data(data):
#     le = LabelEncoder()
#     for column in data.columns:
#         if data[column].dtype == object:
#             data[column] = le.fit_transform(data[column])
#     return data

# def build_ann(data):
#     data = preprocess_data(data)
#     X = data.iloc[:, :-1]
#     y = data.iloc[:, -1]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     scaler = StandardScaler()
#     scaler.fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)
    
#     mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
#     mlp.fit(X_train, y_train)
#     predictions = mlp.predict(X_test)
    
#     return accuracy_score(y_test, predictions)

# # Example usage:
# data = read_csv('4ds.csv')
# accuracy = build_ann(data)
# print("Accuracy:", accuracy)



import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize the images to the range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build the neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', #Loss function used for classification tasks
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1])
plt.legend(loc='lower right')
plt.show()
