import os
import torchvision
from PIL import Image
from sr_trainer import SRTrainer
from option import args


if __name__ == "__main__":
    trainer = SRTrainer(None, None, args.pretrain_path, 0, 2, 0, 1, True)
    img = Image.open(args.test_path)
    out = trainer.inference(img, args.weights_path)
    out = torchvision.transforms.ToPILImage()(out)
    
    input_path = args.test_path
    input_dir = os.path.dirname(input_path)
    input_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(input_dir, f"{input_name}_x2.png")
    
    out.save(output_path)
    print(f"Saved output to: {output_path}")
