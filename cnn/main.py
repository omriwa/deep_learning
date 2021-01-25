import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

test_dataget = ImageDataGenerator(rescale=1./255)

test_set = train_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

# Creating cnn
Sequential = tf.keras.models.Sequential
layers = tf.keras.layers
cnn = Sequential()
cnn.add(layers.Conv2D(filters=32, kernel_size=3, input_shape=[64, 64, 3], activation='relu'))

# Pooling
cnn.add(layers.MaxPooling2D(pool_size=2, strides=2))
 
# Another conv layer
cnn.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

# Flattening layer
cnn.add(layers.Flatten())

# Full connection layer
cnn.add(layers.Dense(units=128, activation='relu'))

# Output layer
cnn.add(layers.Dense(units=1, activation='sigmoid'))

# Training the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set,validation_data=test_set,epochs=25)

# Single prediction
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('./dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
print(training_set.class_indices)
print('dog' if result[0][0] == 1 else 'cat')
