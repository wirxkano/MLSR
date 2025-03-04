import os

path = ''
print(os.listdir(path))

train_hr_dir = os.path.join(path, 'DIV2K_train_HR')
train_lr_bicubic_dir = os.path.join(path, 'DIV2K_train_LR_bicubic/X2')
train_lr_unknown_dir = os.path.join(path, 'DIV2K_train_LR_unknown/X2')

val_hr_dir = os.path.join(path, 'DIV2K_valid_HR')
val_lr_bicubic_dir = os.path.join(path, 'DIV2K_valid_LR_bicubic/X2')
val_lr_unknown_dir = os.path.join(path, 'DIV2K_valid_LR_unknown/X2')

param_restore_path = os.path.join('checkpointx2.ckpt', '')
param_save_path = None

BATCH_SIZE = 16

LR_BETA = 1e-6
LR_ALPHA = 1e-5
LOG_STEP = 50
GRADIENT_NUMBER = 5
TRAIN_ITERATIONS = 10
VALIDATION_STEP = 5

PATCH_SIZE = 512