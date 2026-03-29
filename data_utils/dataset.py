from torch.utils.data import Dataset
from torchvision import transforms as TF
# from torchvision.transforms import functional 
import os.path as osp
import os
from PIL import Image
from natsort import natsorted
import torch
from functools import partial
from torch.utils.data import DataLoader
try:
    from .transforms import paired_transform
except: 
    from transforms import paired_transform
    
def get_dataloader( dataset_root,
                    img_size=256,
                    batch_size=4,
                    transforms=None,
                    pin_memory=False,
                    shuffle=True, 
                    num_workers=0, 
                    prefetch_factor=2,
                    training=True,
                    synthesis=True,
                    return_dataset=False):
    '''
    synthesis dataset: path must include sub_dirs ['smoky', 'smokeless']
    real dataset: path is 'dataset_root'
    '''
    smoky_dir = None
    smokeless_dir = None
    if synthesis:
        sub_dir = ['smoky','smokeless']
        smoky_dir = osp.join(dataset_root,sub_dir[0])
        if not os.path.exists(smoky_dir):
            raise FileNotFoundError(f"'{sub_dir}' does not exist in '{dataset_root}'")
        smokeless_dir = osp.join(dataset_root,sub_dir[1])
        if not os.path.exists(smokeless_dir):
            raise FileNotFoundError(f"'{sub_dir[1]}' does not exist in '{dataset_root}'")
    else:
        smoky_dir = dataset_root    

    dataset = SM_Dataset(smoky_dir=smoky_dir,
                         smokeless_dir=smokeless_dir,
                         training=training,
                         synthesis=synthesis,
                         img_size=img_size,
                         transforms=transforms)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            prefetch_factor=prefetch_factor,
                            shuffle=shuffle)
    if return_dataset:
        return dataset
    return dataloader

class SM_Dataset(Dataset):
    def __init__(self,smoky_dir=None,smokeless_dir=None,
                 training=True,synthesis=True,img_size=256,transforms=None):
        super().__init__()
        self.smoky_path = []  # smoky image
        self.smokeless_path = [] # somkeless image
        self.training = training
        self.synthesis=synthesis
        self.img_size = img_size
        self.transforms = transforms
        if transforms is None:
            self.trans = TF.Compose([TF.Resize((img_size,img_size),antialias=True),
                                                                 TF.ToTensor(),])
        elif transforms=='paired_transform':
            self.trans = partial(paired_transform,crop_size=img_size,hflip=True,rotation=True)
            print(f'paired_transform is used, crop_size={img_size}, hflip=True, rotation=True')
        else:
            self.trans = transforms
        self.smoky_path = self.get_smoky_img(smoky_dir) 
        if self.synthesis:
            self.smokeless_path = self.get_smoky_less(smokeless_dir)
    
    def __getitem__(self, idx):
        smoky = Image.open(self.smoky_path[idx]).convert('RGB')
        if (not self.training) and (not self.synthesis):
            smoky = self.trans(smoky)
            return smoky, self.smoky_path[idx]
        elif not self.training:
            smokeless = Image.open(self.smokeless_path[idx]).convert('RGB')
            smoky = self.trans(smoky)
            smokeless = self.trans(smokeless)
            return smoky, smokeless
        else:
            smokeless = Image.open(self.smokeless_path[idx]).convert('RGB')
            if self.transforms =='paired_transform':
                smoky, smokeless = self.trans(smoky, smokeless)
                return smoky, smokeless
            else:
                seed = torch.random.seed()
                smoky = self._aug(smoky,seed=seed)
                smokeless = self._aug(smokeless,seed=seed)
                return smoky, smokeless

    def _aug(self, img: Image, seed=None):
        if seed is not None:
            torch.random.manual_seed(seed)
        trans_img = self.trans(img)
        return trans_img 
    
    def __len__(self,):
        return len(self.smoky_path)            
    
    def get_smoky_img(self,smoky_path):
        if not osp.exists(smoky_path):
            raise "Path is not exist!"
        smoky_list = []
        for dir, _, files in os.walk(smoky_path):
            for file in files:
                smoky_list.append(osp.join(dir,file))
        return natsorted(smoky_list)
    
    def get_smoky_less(self,smokeless_dir):
        if not osp.exists(smokeless_dir):
            raise "Path is not exist!"
        smokeless_list = []
        for dir, _, files in os.walk(smokeless_dir):
            for file in files:
                smokeless_list.append(osp.join(dir,file))
        return natsorted(smokeless_list)
    

