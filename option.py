import argparse

parser = argparse.ArgumentParser(description='MLSR')

# Subparsers for train and test phases
subparsers = parser.add_subparsers(dest='phase', help='Phase: train or test')
subparsers.required = True

# Train parser
train_parser = subparsers.add_parser('train', help='Training phase')
train_parser.add_argument('--train_path', required=True, help='Train images path')
train_parser.add_argument('--val_path', required=True, help='Validation images path')
train_parser.add_argument('--pretrain_path', help='Pretrained weights for model')
train_parser.add_argument('--save_folder', required=True, help='Saved weights folder')
train_parser.add_argument('--log_path', required=True, help='Log path')
train_parser.add_argument('--snapshot_path', default=None, help='Checkpoint path')

# Test parser
test_parser = subparsers.add_parser('test', help='Testing/Inference phase')
test_parser.add_argument('--test_path', required=True, help='Test images path')
test_parser.add_argument('--pretrain_path', help='Pretrained weights for model')
test_parser.add_argument('--weights_path', required=True, help='Checkpoint file')

args = parser.parse_args()
