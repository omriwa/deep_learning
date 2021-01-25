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
from tf.keras.models import Sequential
from tf.keras.layers as layers

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

