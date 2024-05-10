import os
from keras._tf_keras.keras.callbacks import ModelCheckpoint, LearningRateScheduler

# Load the dataset
execution_path = os.getcwd()
data_directory = os.path.join(execution_path, 'idenprof')

# Decide on the models
model_name = 'idenprof_model-{epoch:03d}-{val_acc}.weights.h5'
model_dir = os.path.join(execution_path, 'idenprof_models')

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

print(lr_schedule)