import numpy as np
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

#custom parameters
nb_class = 2
hidden_dim = 512

vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
print(x.shape)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
print(out.shape)
custom_vgg_model = Model(vgg_model.input, out)

for layer in custom_vgg_model.layers[:-3]:
	layer.trainable = False

print(custom_vgg_model.summary())
custom_vgg_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

train_data_dir = 'cropped_data/train'
validation_data_dir = 'cropped_data/test'
nb_train_samples = 419
nb_validation_samples = 100
epochs = 5
batch_size = 16

# training augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

# test augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,  
        target_size=(224, 224),  # all images will be resized to 224x224
        batch_size=batch_size,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

# custom_vgg_model.fit_generator(
#         train_generator,
#         steps_per_epoch=nb_train_samples // batch_size,
#         epochs=epochs,
#         validation_data=test_generator,
#         validation_steps=nb_validation_samples // batch_size)

for _ in range(epochs):
	for _ in range(nb_train_samples // batch_size):
		x,y = train_generator.next()
		y_ = []
		for i in range(len(y)):
			y_.append(to_categorical(y[i], nb_class))
		y_ = np.array(y_)
		print(custom_vgg_model.train_on_batch(x,y_))
	custom_vgg_model.save('saved_model.h5')  
