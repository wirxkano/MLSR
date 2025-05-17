from sr_trainer import SRTrainer
from utils.logger import Logger
from option import args

if __name__ == '__main__':
    if args.phase == "train":
        logger = Logger("IDN", file=args.log_path)
        
        trainer = SRTrainer(train_path=args.train_path,
                            val_path=args.val_path,
                            pretrain_path=args.pretrain_path,
                            max_epoch=200,
                            batch_size=16,
                            patch_size=96,
                            scale=2,
                            pin_memory=True,
                            have_validate=True,
                            save_best_for=("ssim", "geq"),
                            save_period=10,
                            save_folder=args.save_folder,
                            snapshot_path=args.snapshot_path,
                            logger=logger)
        
        trainer.train()

    elif args.phase == "test":
        trainer = SRTrainer(None, None, args.pretrain_path, 0, 2, 0, 1, True)
        trainer.test(test_path=args.test_path, weights_path=args.weights_path)
