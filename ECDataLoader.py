import torch.utils.data as data
import os
from PIL import Image, ImageDraw, ImageFont

class ECDatas(data.Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.ids=[]
        allclass = os.listdir(root)
        for cl in allclass:
            clpath=os.path.join(root,cl)
            imglist=os.listdir(clpath)
            self.ids.extend([(cl,x) for x in imglist])


    def __getitem__(self, index):
        target,img_id = self.ids[index]
        #print(os.path.join(self.root,target,img_id))
        img = Image.open(os.path.join(self.root,target,img_id))
        #print(img.shape)

        if self.transform is not None:
            img = self.transform(img)

        return img, int(target)

    def __len__(self):
        return len(self.ids)