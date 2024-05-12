import os
from keras._tf_keras.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.api.layers import Conv2D

from tensorflow.python.keras import models, layers 
# Load the dataset
execution_path = os.getcwd()
data_directory = os.path.join(execution_path, 'idenprof')

data_test_directory = os.path.join(data_directory, 'test')


# Decide on the models
model_name = 'idenprof_model-{epoch:03d}-{val_acc}.weights.h5'
model_dir = os.path.join(execution_path, 'idenprof_models')

idenprof = os.listdir(data_test_directory)

# print("idenprof",len(idenprof))

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

modelpath = os.path.join(model_dir, model_name)

# Checkpoint to save the best models only.
checkpoint = ModelCheckpoint(filepath=modelpath,
                monitor='val_acc',
                verbose=1,
                save_best_only=True,
                save_weights_only = True)


# Adjusting the Learning Rate(LR)
def lr_schedule(epoch):
    # LR is scheduled to be reduced after 80, 120, 160, 180 epochs.

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

lr_schedule = LearningRateScheduler(lr_schedule)

# print(lr_schedule)
# Building the model
# Cov2D()->Filters in a model
# Level of Cov2D filters: 32, 64, 128
def resnet_model(input, channel_depth):
    Conv2D(level=32,filters=channel_depth, kernel_size=200, padding="same")
    Conv2D(level=64,filters=channel_depth, kernel_size=200, padding="same")
    Conv2D(level=128,filters=channel_depth, kernel_size=200, padding="same")

    strided_pool = 0
    
    for i in idenprof:
        strided_pool += i

    return strided_pool


# Model creation
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100)),
    layers.Flatten(),
    layers.Conv2D(64, (3,3), activation='relu', input_shape=(100, 100)),
    layers.Flatten(),
    layers.Conv2D(128, (3,3), activation='relu', input_shape=(100, 100)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')])




    
# Modile Compile
model.compile(optimizer='adam', loss="percentage", metrics=["accuracy"])


# Training the model
model.fit(idenprof['chef'], idenprof['doctor'], epochs=10, batch_size=200)

