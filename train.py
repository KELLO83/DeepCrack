from data.augmentation import augCompose, RandomBlur, RandomColorJitter
from data.dataset import readIndex, dataReadPip, loadedDataset , FileRead
from tqdm import tqdm
from model.deepcrack import DeepCrack
from trainer import DeepCrackTrainer
from config import Config as cfg
import numpy as np
import torch
import os
import cv2
import sys
from copy import deepcopy
#from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
from torch.utils.data.dataloader import DataLoader
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id

def visualize_batch_cv2(images, masks):
    batch_size = images.size(0)
    print(f"Batch_size  : {batch_size}")
    for i in range(batch_size): # [4,3,512,512] , [4,1,512,512]
        image = images[i].permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
        image = (image * 255).astype(np.uint8)  # 이미지 값을 0-255 범위로 변환
        mask = masks[i].squeeze(0).cpu().numpy()  # [1, H, W] -> [H, W]
        mask = (mask * 255).astype(np.uint8)  # 마스크 값을 0-255 범위로 변환
        
        print("image shape " ,image.shape)
        print("mask shape " ,mask.shape)
        # OpenCV를 사용하여 이미지와 마스크 표시
        image_named = "image"
        mask_named = "mask"
        cv2.namedWindow(image_named)
        cv2.namedWindow(mask_named)
        cv2.moveWindow(image_named,500,1000)
        cv2.moveWindow(mask_named,1500,1000)        
        cv2.imshow(image_named, image)
        cv2.imshow(mask_named, mask)
        cv2.waitKey()  #
        cv2.destroyAllWindows()

def main(opt):
    # ----------------------- dataset ----------------------- #
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    
    data_augment_op = augCompose(transforms=[[RandomColorJitter, 0.5], [RandomBlur, 0.2]])

    train_pipline = dataReadPip(transforms=data_augment_op)

    test_pipline = dataReadPip(transforms=None)

    train_dataset = FileRead(cfg.train_data_path, preprocess=train_pipline , image_type='jpg')

    test_dataset = FileRead(cfg.test_data_path, preprocess=test_pipline, image_type='jpg')

    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size,
                                               shuffle=True, num_workers=8, drop_last=True)

    val_loader = DataLoader(test_dataset, batch_size=cfg.val_batch_size,
                                             shuffle=False, num_workers=8, drop_last=True)
    
    
    try:
        pretrained = opt.weights.endswith(".pt")
    except:
        pass
    
    if pretrained:
        ckpt = torch.load(opt.weights, map_location=device)
        model.load_state_dict(ckpt["model"].float().state_dict())
        logging.info(f"Model ckpt loaded from {opt.weights}")
        model.to(device)

    start_epoch = 0
    if pretrained:
        print("======================pretrained=======================")
        if ckpt["optimizer"] is not None:
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt["best_loss"]
            logging.info(f"Optimizer loaded from {opt.weights}")
            if start_epoch < opt.epochs:
                logging.info(
                    f"{opt.weights} has been trained for {start_epoch} epochs. Fine-tuning for {opt.epochs} epochs"
                )
        del ckpt
        
    
    save_dir = opt.save_dir
    best , last = f"{save_dir}/best.pt" , f"{save_dir}/last.pt"
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    print("====================Train Start=====================================")
    # -------------------- build trainer --------------------- #

    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()

    model = DeepCrack()
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))

    trainer = DeepCrackTrainer(model).to(device)
    val_trainer = DeepCrackTrainer(model).to(device)
    
    writer = SummaryWriter('./log_dir')
    # if cfg.pretrained_model:
    #     pretrained_dict = trainer.saver.load(cfg.pretrained_model, multi_gpu=True)
    #     model_dict = model.state_dict()

    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)
    #     trainer.vis.log('load checkpoint: %s' % cfg.pretrained_model, 'train info')
    
    
    sclar_count = 0
    val_count = 0
    best_loss = 100
    for epoch in range(start_epoch, cfg.epoch):
        logging.info(("\n" + "%12s" * 3) % ("Epoch", "GPU Mem", "Loss"))
        model.train()
        epoch_loss = 0
        # ---------------------  training -------------------- #
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for  i   , (img, lab) in progress_bar:
            data, target = img.to(device) , lab.to(device)
            loss = trainer.train_op(data, target)
            epoch_loss += loss.item()
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            progress_bar.set_description(("%12s" * 2 + "%12.4g") % (f"{epoch + 1}/{cfg.epoch}", mem, loss))

        writer.add_scalar('loss/train',loss.item(),sclar_count)
        sclar_count += 1
        loss = validate(model , val_loader , device , val_trainer)
        logging.info(f"VALIDATION: Loss: {loss:.4f}")
        writer.add_scalar('loss/val',loss.item(),val_count)
        val_count += 1
        
        ckpt = {
            'epoch' : epoch,
            'best_loss' : loss,
            "model" : deepcopy(model).half(),
            "optimizer" : trainer.optimizer.state_dict(),
        }
        
        torch.save(ckpt , last)
        
        if best_loss > loss :
            best_loss = min(best_loss , loss.item())
            torch.save(ckpt , best)
            
    for f in best , last:
        strip_optimizers(f)
    
    writer.close()


@torch.inference_mode()
def validate(model, data_loader, device,val_trainer):
    model.eval()
    for image, target in tqdm(data_loader, total=len(data_loader)):
        image, target = image.to(device), target.to(device)
        with torch.no_grad():
            loss = val_trainer.train_op(image,target,val=True)
    model.train()
    return loss



def strip_optimizers(f: str) -> None:
    """Strip optimizer from 'f' to finalize training"""
    x = torch.load(f, map_location="cpu")
    for k in "optimizer", "best_score":
        x[k] = None
    x["epoch"] = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, f)
    mb = os.path.getsize(f) / 1e6  # get file size
    logging.info(f"Optimizer stripped from {f}, saved as {f} {mb:.1f}MB")



def parser_opt():
    parser = argparse.ArgumentParser(description="Model train")
    parser.add_argument('--save_dir',default='weight' , help="weight save path")
    parser.add_argument("--weights", type=str, default='', help="Pretrained model, default: None")
    
    return parser.parse_args()

if __name__ == '__main__':
    opt = parser_opt()
    main(opt)
