from torch import nn
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

    def train_op(self, input, target):
        
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred_output = self.model(input)
            loss = self.mask_loss(pred_output.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return  loss
        
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
