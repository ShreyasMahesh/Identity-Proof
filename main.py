import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers

# Load the dataset
execution_path = os.getcwd()
data_directory = os.path.join(execution_path, 'idenprof')
data_test_directory = os.path.join(data_directory, 'test')
data_train_directory = os.path.join(data_directory, 'train')

# Decide on the models
model_name = 'idenprof_model-{epoch:03d}-{val_accuracy:.4f}.weights.h5'
model_dir = os.path.join(execution_path, 'idenprof_models')

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

modelpath = os.path.join(model_dir, model_name)

# Checkpoint to save the best models only.
checkpoint = ModelCheckpoint(filepath=modelpath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True)

# Adjusting the Learning Rate (LR)
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 1e-4
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print("The learning rate is: ", lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# Image data generators for loading and augmenting images
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(data_train_directory,
                                                    target_size=(100, 100),
                                                    batch_size=32,
                                                    class_mode='binary',
                                                    color_mode='grayscale')

validation_generator = test_datagen.flow_from_directory(data_test_directory,
                                                        target_size=(100, 100),
                                                        batch_size=32,
                                                        class_mode='binary',
                                                        color_mode='grayscale')

# Model creation
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Model compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(train_generator,
          epochs=5,
          steps_per_epoch=train_generator.samples // train_generator.batch_size,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples // validation_generator.batch_size,
          callbacks=[checkpoint, lr_scheduler])

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print("Validation accuracy: {:.4f}".format(accuracy))

# Save the final model
final_model_path = os.path.join(model_dir, 'final_model.h5')
model.save(final_model_path)
print(f"Model saved to {final_model_path}")
