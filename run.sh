torchrun /kaggle/input/mlsr-rcan/MLSR/main.py \
--train_path /kaggle/input/div2k-high-resolution-images/DIV2K_train_HR/DIV2K_train_HR \
--val_path /kaggle/input/div2k-high-resolution-images/DIV2K_valid_HR/DIV2K_valid_HR \
--pretrain_path ./runs/RCAN_BIX2.pt \
--save_folder /kaggle/working/ \
--log_path /kaggle/working/logfile.log
