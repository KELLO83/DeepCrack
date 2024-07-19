import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import natsort
import os 



def readIndex(index_path, shuffle=False):
    img_list = []
    with open(index_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    if shuffle is True:
        random.shuffle(img_list)
    return img_list

def _resize_and_pad(image , size=(512, 512)):
    image = np.array(image)
    
    h, w  = image.shape[:2]
    scale = size[0] / max(h, w)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    
    delta_w = size[1] - new_w
    delta_h = size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    color = [0, 0, 0] 
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    #print("resize image :",padded_image.shape)
    
    return padded_image


class dataReadPip(object):
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, image_path , mask_path):

        img = cv2.imread(image_path)
        lab = cv2.imread(mask_path)

        if img is None or lab is None:
            raise FileNotFoundError(f"{image_path} or {mask_path}")

        if len(lab.shape) != 2:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)


        if self.transforms is not None:

            img, lab = self.transforms(img, lab)

        img = _preprocess_img(img)
        lab = _preprocess_lab(lab)
        
        lab = lab.unsqueeze(dim=0)
        return img, lab
    
def _preprocess_img(cvImage):
    '''
    :param cvImage: numpy HWC BGR 0~255
    :return: tensor img CHW BGR  float32 cpu 0~1
    '''
    
    cvImage = _resize_and_pad(cvImage)
    
    cvImage = cvImage.transpose(2, 0, 1).astype(np.float32) / 255

    
    return torch.from_numpy(cvImage)

def _preprocess_lab(cvImage):
    '''
    :param cvImage: numpy 0(background) or 255(crack pixel)
    :return: tensor 0 or 1 float32
    '''
    
    cvImage = _resize_and_pad(cvImage)
    
    cvImage = cvImage.astype(np.float32) / 255

    return torch.from_numpy(cvImage)


class loadedDataset(Dataset):
    """
    Create a torch Dataset from data
    """

    def __init__(self, dataset, preprocess=None):
        super(loadedDataset, self).__init__()
        self.dataset = dataset
        if preprocess is None: # preprcess get file path
            preprocess = lambda x: x
        self.preprocess = preprocess

    def __getitem__(self, index):
        return self.preprocess(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


class FileRead(Dataset):
    def __init__(self,
                 file_path : str,
                 image_type : str,
                 preprocess : dataReadPip,
                 val : bool=False):
        
        super(FileRead , self).__init__()
        self.image_path = file_path
        self.mask_path = self.__change_path(self.image_path)
        
        self.image_type = image_type
        self.preprocess = preprocess
        
        self.image_filenames = self.__load_file_names(file_path , self.image_type)
        self.mask_filenames = self.__load_file_names(self.mask_path , 'bmp')
        

    def __len__(self) -> int:
        return len(self.image_filenames)
    
    
    def __getitem__(self , idx ):
        filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_path , f"{filename}.{self.image_type}")
        mask_path = os.path.join(self.mask_path , f"{filename}.bmp")
        
        image , mask = self.preprocess(image_path , mask_path)
        #self.__visualize_cv2(image,mask)
        
        # image = self.__resize_and_pad(image)
        # mask = self.__resize_and_pad(mask)
        return image , mask
    
    def __change_path(self,file_path):
        return file_path.replace('IMAGE','MASK')
    
    def __load_file_names(self , file_path , file_extension):
        filenames = [ os.path.splitext(filename)[0] for filename in os.listdir(file_path) if filename.endswith(file_extension)]
        return natsort.natsorted(filenames)
    
    def __resize_and_pad(self , image , size=(512, 512)):
        image = np.array(image)
        
        h, w = image.shape[:2]
        scale = size[0] / max(h, w)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        
        delta_w = size[1] - new_w
        delta_h = size[0] - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        color = [0, 0, 0] 
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        #print("resize image :",padded_image.shape)
        
        return padded_image
    
    def __visualize_cv2(self , images, masks):
        image = images.permute(1, 2, 0).cpu().numpy()  
        image = (image * 255).astype(np.uint8)  
        mask = masks.squeeze(0).cpu().numpy()  
        mask = (mask * 255).astype(np.uint8)  
        print("image shape " ,image.shape)
        print("mask shape " ,mask.shape)

        image_named = "image"
        mask_named = "mask"
        cv2.namedWindow(image_named)
        cv2.namedWindow(mask_named)
        cv2.moveWindow(image_named,1000,1000)
        cv2.moveWindow(mask_named,1500,1000)        
        cv2.imshow(image_named, image)
        cv2.imshow(mask_named, mask)
        cv2.waitKey()  
        cv2.destroyAllWindows()
    