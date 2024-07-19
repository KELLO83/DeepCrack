from torch import nn
import torch.utils
import torch.utils.data
from tools.visdom import Visualizer
from tools.checkpointer import Checkpointer
from config import Config as cfg
import torch


def get_optimizer(model):
    if cfg.use_adam:
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum, )


class DeepCrackTrainer(nn.Module):
    def __init__(self, model):
        super(DeepCrackTrainer, self).__init__()
        #self.vis = Visualizer(env=cfg.vis_env)
        self.model = model

        self.saver = Checkpointer(cfg.name, cfg.saver_path, overwrite=False, verbose=True, timestamp=True,
                                  max_queue=cfg.max_save)

        self.optimizer = get_optimizer(self.model)

        self.iter_counter = 0

        # -------------------- Loss --------------------- #

        self.mask_loss = nn.BCEWithLogitsLoss(reduction='mean',
                                              pos_weight=torch.tensor([cfg.pos_pixel_weight]))

        self.scaler = torch.cuda.amp.GradScaler()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.1,patience=10,min_lr=1e-8)
        
    def train_op(self, input, target,val = False):
        
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            output , f5 ,f4 ,f3, f2, f1 = self.model(input)
            loss = self.mask_loss(output.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
            
        if not val:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        self.iter_counter += 1
        
        return  loss
    
    def train_loss_op(self , input , target , val = False):
        """ summation get total loss"""
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output , f5 , f4 ,f3 , f2 ,f1 = self.model(input)
            output_loss = self.mask_loss(output.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
            fuse5_loss = self.mask_loss(f5.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
            fuse4_loss = self.mask_loss(f4.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
            fuse3_loss = self.mask_loss(f3.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
            fuse2_loss = self.mask_loss(f2.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
            fuse1_loss = self.mask_loss(f1.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        
        total_loss = output_loss + fuse5_loss + fuse4_loss + fuse3_loss + fuse2_loss + fuse1_loss
        
        if not val:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step(total_loss)
            print(self.scheduler.get_last_lr())        
        
        self.iter_counter += 1

        # self.log_loss = {
        #     'total_loss': total_loss.item(),
        #     'output_loss': output_loss.item(),
        #     'fuse5_loss': fuse5_loss.item(),
        #     'fuse4_loss': fuse4_loss.item(),
        #     'fuse3_loss': fuse3_loss.item(),
        #     'fuse2_loss': fuse2_loss.item(),
        #     'fuse1_loss': fuse1_loss.item()
        # }

        return total_loss

    def val_op(self, input, target):
        pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1, = self.model(input)

        output_loss = self.mask_loss(pred_output.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse5_loss = self.mask_loss(pred_fuse5.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse4_loss = self.mask_loss(pred_fuse4.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse3_loss = self.mask_loss(pred_fuse3.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse2_loss = self.mask_loss(pred_fuse2.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse1_loss = self.mask_loss(pred_fuse1.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size

        total_loss = output_loss + fuse5_loss + fuse4_loss + fuse3_loss + fuse2_loss + fuse1_loss

        self.log_loss = {
            'total_loss': total_loss.item(),
            'output_loss': output_loss.item(),
            'fuse5_loss': fuse5_loss.item(),
            'fuse4_loss': fuse4_loss.item(),
            'fuse3_loss': fuse3_loss.item(),
            'fuse2_loss': fuse2_loss.item(),
            'fuse1_loss': fuse1_loss.item()
        }

        return pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1,

    def acc_op(self, pred, target):
        mask = target

        pred = pred.clone()
        pred[pred > cfg.acc_sigmoid_th] = 1
        pred[pred <= cfg.acc_sigmoid_th] = 0

        pred_mask = pred[:, 0, :, :].contiguous()
        mask = mask[:, 0, :, :].contiguous()

        mask_acc = pred_mask.eq(mask).sum().item() / mask.numel()


        mask_pos_acc = pred_mask[mask > 0].eq(mask[mask > 0]).sum().item() / mask[mask > 0].numel()

        mask_neg_acc = pred_mask[mask <= 0].eq(mask[mask <= 0]).sum().item() / mask[mask <= 0].numel()


        self.log_acc = {
            'mask_acc': mask_acc,
            'mask_pos_acc': mask_pos_acc,
            'mask_neg_acc': mask_neg_acc,
        }
