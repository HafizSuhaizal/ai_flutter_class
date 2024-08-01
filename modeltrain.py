import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Assuming 3 classes: Earbuds, Watch, Spectacle
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set up data generators
#Change the path according to your own path
train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    '/Users/hafizsuhaizal/Downloads/dataset',  # Use the root directory containing subdirectories
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    '/Users/hafizsuhaizal/Downloads/dataset',  # Use the root directory containing subdirectories
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the model
# Save model in your own path
model.save('/Users/hafizsuhaizal/Downloads/converted_keras-8/keras_model.keras')
