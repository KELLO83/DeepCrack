import os
import logging
import argparse
from tqdm.auto import tqdm
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cv2
import numpy as np

from sklearn.metrics import average_precision_score
from data.dataset import FileRead , dataReadPip
from model.deepcrack import DeepCrack
import pdb

def get_ap(output,target):
    output = torch.sigmoid(output)
    
    output = output.contiguous().flatten().detach().cpu().numpy()
    target = target.contiguous().flatten().detach().cpu().numpy()
    
    ap_score = average_precision_score(target , output)

    print("Ap score : {:.5f}%".format(ap_score*100))


def find_best_threshold(outputs , target):
    thresholds = np.arange(0, 1.01, 0.01)
    best_f1 = 0.0
    best_thresholds = 0.0
    for i in thresholds:
        f1 = get_f1_score(outputs,i , target)
        if f1 > best_f1:
            best_f1 = f1
            best_thresholds = i
            # print("best f1 update : {:.4f} best th update : {:.4f}".format(best_f1,best_thresholds))
    
    #print("best thresholds : ",best_thresholds)

    return best_thresholds , best_f1


def get_f1_score(outputs , th , target):
    output = torch.sigmoid(outputs)
    output = (output > th).float()
    Epsilon = 1e-8
    
    TP = (output * target).sum().item()
    FP = (output * (1-target)).sum().item()
    FN = ((1-output) * target).sum().item()
    
    precision = TP / (TP + FP + Epsilon )
    recall = TP / (TP + FN + Epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + Epsilon)
    
    return f1_score
    
def get_OIS(out_list : list) :
    f1_list = []
    for i , (out , target) in enumerate(out_list):
        output = torch.sigmoid(out)
        best_th , f_ = find_best_threshold(output , target)
        print(f"{i}-->Threshold : {best_th:.2f} f1_score : {f_:.4f}")
        f1_list.append(f_)
        
    OIS_RESULT =  sum(f1_list) / len(f1_list)    
    print(f"OIS --------->{OIS_RESULT:.3f}")
    
    
def get_ODS(out_list,th=0.5,flag=False):
    
    precision_list = []
    recall_list = []
    f1_score_list = []
    
    for output,target in out_list:
        output = torch.sigmoid(output)
        
        output = (output > th).float()
        
        TP = (output * target).sum().item()
        FP = (output * (1-target)).sum().item()
        FN = ((1-output) * target).sum().item()
        
        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

    p = sum(precision_list) / len(precision_list)
    r = sum(recall_list) / len(recall_list)
    f = sum(f1_score_list) / len(f1_score_list)
    #print(f"임계점 : {th} ------> 정밀도평균 : {p:.5f} 재현율평균 : {r:.5f} f1 score평균 : {f:.5f}")
    return th , p , r , f


@torch.inference_mode
def evaulte(model , data_loader,device):
    model.eval()
    loss_list = []
    out_list = []
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    for image , target  in tqdm(data_loader , total=len(data_loader)):
        image , target = image.to(device) , target.to(device)
        with torch.no_grad():
            output = model(image)
            loss = criterion(output,target)
            out_list.append((output,target))
            loss_list.append(loss.item())
    
    it_range = np.arange(0.1 , 1.01 , 0.01)
    best_th = 0.0
    best_f = 0.0
    best_p = 0.0
    best_r = 0.0
    get_OIS(out_list)
    for th in it_range:
        th_res , p , r , f= get_ODS(out_list,th)   
        if f > best_f:
            best_f = f 
            best_th = th_res
            best_p = p
            best_r = r
    print(f"----------> 정밀도평균 : {best_p:.5f} 재현율평균 : {best_r:.5f} ODS(F1 score 평균) : {best_f:.5f}")
    return best_th , sum(loss_list) / len(loss_list)


@torch.inference_mode
def image_segmentation_runnung(model , input , device , th):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        image,mask = input[0].to(device) , input[1].to(device)
        output = model(image)
        loss = criterion(output , mask)
        get_ap(output,mask)
        
    return output , loss , th

def parse_opt():
    parser = argparse.ArgumentParser(description="Inferece test Dataset for CRKW100 CRACKLS315 STONE331")
    parser.add_argument("--weight_path" ,type=str , default='weight/concat_loss/best.pt' , help='Input model weight path')
    parser.add_argument("--image_path" , type=str , default='data/CRKWH100_IMAGE/test' , help='data/CRKWH100_IMAGE')
    parser.add_argument('--image_type' , type=str , default='png' , help='CRKWH100 -> PNG other JPG')
    parser.add_argument('--resize' , type=bool , default=False , help="Stone dataset need true")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    model = DeepCrack()
    weight_path = opt.weight_path
    ckpt = torch.load(weight_path,map_location=device)
    model.load_state_dict(ckpt['model'].float().state_dict())
    model = model.to(device)
    
    # logging.info(
    #     f"Network:\n"
    #     f"\t{model.in_channels} input channels\n"
    #     f"\t{model.out_channels} output channels (number of classes)"
    # )
    

    image_path = opt.image_path
    
    parts = image_path.split('/')
    target = parts[1]
    mask_target = target.split('_')[0] + '_MASK'

    res = f"{parts[0]}/{mask_target}/{parts[2]}"
    mask_path = res
    
    image_type = opt.image_type
    
    
    test_pipline = dataReadPip(transforms=None)
    test_dataset = FileRead(opt.image_path , preprocess=test_pipline , image_type='jpg')
    test_loader = DataLoader(test_dataset,batch_size=4,num_workers=8,shuffle=False,pin_memory=True)
    
    for index , i in enumerate(test_loader):
        batch = i
        print(batch[0].shape)
        print(batch[1].shape)
        if index == 9:
            t = i
            image__ = t[0]
            label__ = t[1]
            break
    print("----------=======----------")
    pdb.set_trace()
    best_th , loss = evaulte(model,test_loader,device)
    print("Test Dataset loss :{:.5f}".format(loss))
    
    target =  image__.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    mask = label__.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    target = (target * 255).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)

    
    example = [image__ , label__]
    out_image , out_loss , th =  image_segmentation_runnung(model , example , device ,best_th)
    out_image = torch.sigmoid(out_image)
    print(f"out image loss : {out_loss:.3f}")
    print(f"Image probabilityDistribution min : {torch.min(out_image):.5f} max : {torch.max(out_image):.5f}")
    out_image = (out_image > th).float() * 255
    out_image = out_image.to(torch.uint8)
    out_image = out_image.squeeze(0).permute(1,2,0).detach().cpu().numpy()

    
    win_image_name = 'image'
    mask_image_name = 'mask'
    out_image_name = 'out'
    
    cv2.namedWindow(win_image_name)
    cv2.namedWindow(mask_image_name)
    cv2.namedWindow(out_image_name)
    
    cv2.moveWindow(win_image_name,800,1200)
    cv2.moveWindow(mask_image_name,1300,1200)
    cv2.moveWindow(out_image_name , 1800 , 1200)
    
    cv2.imshow(win_image_name,target)
    cv2.imshow(mask_image_name,mask)
    cv2.imshow(out_image_name,out_image)
    
    cv2.waitKey()
    cv2.destroyAllWindows()