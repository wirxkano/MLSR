import argparse

parser = argparse.ArgumentParser(description='MLSR')

parser.add_argument('--phase', default='train', help='Training phase or testing phase')

parser.add_argument('--train_path', help='Train images path')
parser.add_argument('--val_path', help='Validation images path')
parser.add_argument('--pretrain_path', help='Pretrained weights for model')
parser.add_argument('--save_folder', help='Saved weights')
parser.add_argument('--log_path', help='Log path')
parser.add_argument('--snapshot_path', default=None, help='Checkpoint')

parser.add_argument('--test_path', help='Test images path', default=None)
parser.add_argument('--weights_path', help='Checkpoint file')

args = parser.parse_args()
