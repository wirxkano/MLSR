import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import datetime
from const import *


class SRTrainer:
    def __init__(self, train_loader, val_loader, network_model_class):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.batch_size = BATCH_SIZE
        self.log_step = LOG_STEP
        self.validation_step = VALIDATION_STEP
        self.train_iteration = TRAIN_ITERATIONS
        self.param_restore_path = param_restore_path
        self.param_save_path = param_save_path
        self.lr_beta = LR_BETA
        self.lr_alpha = LR_ALPHA
        self.gradient_number = GRADIENT_NUMBER
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.network_model_class = network_model_class
        
        self.build_success = False

    def set_optimizer(self):
        self.update_optimizer = optim.SGD(self.update_network.parameters(), lr=self.lr_alpha)
        self.fomaml_optimizer = optim.Adam(self.final_network.parameters(), lr=self.lr_beta)
        self.copy_optimizer = optim.SGD(self.copied_network.parameters(), lr=self.lr_alpha)

    def build(self):
        self.final_network = self.network_model_class().to(self.device)
        self.update_network = self.network_model_class().to(self.device)
        self.copied_network = self.network_model_class().to(self.device)

        self.set_optimizer()
        self.build_success = True
        print('>> Build complete!')

    def train_one_step(self, epoch):
        self.final_network.train()
        
        loss_values = []
        psnr_values = []
        
        for train_img_lr, train_img_bicubic, train_img_hr in self.train_loader:
            self.update_network.load_state_dict(self.final_network.state_dict())
            
            for _ in range(self.gradient_number):
                self.update_optimizer.zero_grad()
                loss = self.update_network.loss(train_img_lr.to(self.device), 
                                                train_img_bicubic.to(self.device),
                                                train_img_hr.to(self.device))
                loss.backward()
                self.update_optimizer.step()
                
        with torch.no_grad():
            for eval_img_lr, eval_img_bicubic, eval_img_hr in self.val_loader:
                loss = self.update_network.loss(eval_img_lr.to(self.device),
                                                eval_img_bicubic.to(self.device),
                                                eval_img_hr.to(self.device))
                psnr = self.update_network.psnr(eval_img_lr.to(self.device),
                                                eval_img_bicubic.to(self.device),
                                                eval_img_hr.to(self.device))
                
                loss_values.append(loss.item())
                psnr_values.append(psnr)
                
        self.fomaml_optimizer.step()

        return np.mean(loss_values), np.mean(psnr_values)

    def validation(self):
        self.copied_network.eval()
        test_size = len(self.val_loader)
        updated_psnr = []
        base_psnr = []

        with torch.no_grad():
            for i in range(test_size):
                img_lr, img_bicubic, img_hr, maml_img_lr, maml_img_bicubic, maml_img_hr = self.dataset.next(test=True)

                # Copy tham số từ final_network
                self.copied_network.load_state_dict(self.final_network.state_dict())

                base_psnr.append(self.copied_network.psnr(maml_img_lr.to(self.device), 
                                                          maml_img_bicubic.to(self.device), 
                                                          maml_img_hr.to(self.device)))

                # Cập nhật tham số
                for _ in range(self.gradient_number):
                    self.copy_optimizer.zero_grad()
                    loss = self.copied_network.loss(img_lr.to(self.device),
                                                    img_bicubic.to(self.device),
                                                    img_hr.to(self.device))
                    loss.backward()
                    self.copy_optimizer.step()

                updated_psnr.append(self.copied_network.psnr(maml_img_lr.to(self.device), 
                                                             maml_img_bicubic.to(self.device), 
                                                             maml_img_hr.to(self.device)))

        return np.mean(updated_psnr), np.mean(base_psnr)

    def train(self):
        assert self.param_save_path is not None, 'param_save_path is None'
        if not os.path.exists(self.param_save_path):
            os.makedirs(self.param_save_path)

        self.build()
        
        if(self.param_restore_path != None):
            restore_path = os.path.join(self.param_restore_path, 'model.ckpt')
            self.final_network.load_state_dict(torch.load(self.param_restore_path, weights_only=True), strict=False)
            self.final_network.eval()
            print('>> restored parameter from {}'.format(restore_path), flush=True)
                
        print('\n[*] Start training MLSR\n')

        loss_log, psnr_log, best_psnr_test = 0, 0, 0
        for epoch in range(1, self.train_iteration + 1):
            train_loss, train_psnr = self.train_one_step(epoch)
            loss_log += train_loss
            psnr_log += train_psnr

            if epoch % self.log_step == 0:
                loss_log /= self.log_step
                psnr_log /= self.log_step
                now = datetime.datetime.now()
                print("[{}] Step: [{}/{}] Loss: {:.6f} PSNR: {:.6f}".format(
                    now.strftime('%Y-%m-%d %H:%M:%S'), epoch, self.train_iteration, loss_log, psnr_log))
                loss_log, psnr_log = 0, 0

            if epoch % self.validation_step == 0:
                updated_psnr, base_psnr = self.validation()
                print(">> Test PSNR: (base: {:.6f}), (updated: {:.6f})".format(base_psnr, updated_psnr))

                if updated_psnr > best_psnr_test:
                    best_psnr_test = updated_psnr
                    torch.save(self.final_network.state_dict(), os.path.join(self.param_save_path, 'best_model.pth'))

                torch.save(self.final_network.state_dict(), os.path.join(self.param_save_path, 'last_model.pth'))
