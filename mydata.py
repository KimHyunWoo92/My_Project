import os
from torch.utils.data import Dataset
from PIL import Image

def read_data():
    dir='./Extract_Imgs'

    images = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            images.append(os.path.join(root, f))

    return images

class MyDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list=file_list
        self.transform=transform
        
    def __getitem__(self,idx):
        img_path=self.file_list[idx]
        img=Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img
    
    def __len__(self):
        return len(self.file_list)
