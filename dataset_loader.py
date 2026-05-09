import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DeepfakeDataset(Dataset):

    def __init__(self,folder):

        self.images=[]
        self.labels=[]
        real=os.path.join(folder,"real")
        fake=os.path.join(folder,"fake")

        for img in os.listdir(real):
            self.images.append(os.path.join(real,img))
            self.labels.append(0)

        for img in os.listdir(fake):
            self.images.append(os.path.join(fake,img))
            self.labels.append(1)

        self.transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):

        img=cv2.imread(self.images[index])
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img=self.transform(img)

        label=torch.tensor(self.labels[index]).float()

        return img,label