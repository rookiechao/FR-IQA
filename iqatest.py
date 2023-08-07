import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.cuda
from torch.utils.data import Dataset
from Correlation_calculation import get_PLCC, get_SROCC, mos_scatter, get_KROCC
# from swin_transformer import SwinTransformer
# from model import swin_small_patch4_window7_224
from model import SwinTransformer
from vit_model_feature import vit_base_patch16_224_in21k
from nat import nat_mini


def make_model1():
    model=models.vgg16(pretrained=True).features[:4]	
    model=model.eval()	
    model.cuda()
    return model
def make_model2():
    model=models.vgg16(pretrained=True).features[:9]	
    model=model.eval()	
    model.cuda()	
    return model
def make_model3():
    model=models.vgg16(pretrained=True).features[:16]	
    model=model.eval()	
    model.cuda()	
    return model
def make_model4():
    model=models.vgg16(pretrained=True).features[:23]	
    model=model.eval()	
    model.cuda()	
    return model
def make_model5():
    model=models.vgg16(pretrained=True).features[:30]	
    model=model.eval()	
    model.cuda()	
    return model


def prepare_image(image, resize = False, repeatNum = 1):
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0).repeat(repeatNum,1,1,1)

def gm(img):
    img = 0.299*img[:,0,:,:]+0.587*img[:,1,:,:]+0.114*img[:,2,:,:].unsqueeze(1)
    dx = torch.Tensor([[3, 0, -3], [10, 0, -10],  [3,  0, -3]]).float()/16
    dy = torch.Tensor([[3, 10, 3], [0, 0, 0],  [-3,  -10, -3]]).float()/16
    dx = dx.reshape(1,1,3,3)
    dy = dy.reshape(1,1,3,3)
    IxY1 = F.conv2d(img, dx, stride=1, padding =1)    
    IyY1 = F.conv2d(img, dy, stride=1, padding =1)    
    gradientMap = torch.sqrt(IxY1**2 + IyY1**2+1e-12)
    return gradientMap

class MyDataset(Dataset):
    def __init__(self, filepath,image_dir,vitpath):
        self.image_dir=image_dir
        self.vitpath=vitpath
        self.xy = np.loadtxt(filepath,str,delimiter=',',skiprows=1)
    def __getitem__(self, index):
        xy=self.xy
        dist_name=xy[index][0]
        ref_name=xy[index][1]
        mos= float(xy[index][2])
        
        dist_path=os.path.join(self.image_dir,dist_name)
        ref_path=os.path.join(self.image_dir,ref_name)
        
        vit_dist=os.path.join(self.vitpath,dist_name)
        vit_ref=os.path.join(self.vitpath,ref_name)
        
        dist = prepare_image(Image.open(dist_path).resize((224,224)).convert("RGB"), repeatNum = 1).cuda()
        ref = prepare_image(Image.open(ref_path).resize((224,224)).convert("RGB"), repeatNum = 1).cuda()
        return vit_dist,vit_ref,dist,ref,mos

    def __len__(self):
        return self.xy.shape[0]

def iqa(dist,ref,vit_dist,vit_ref):
    
    #stage1
    vgg0=model1(Variable(ref))
    vgg0=vgg0.data.cpu()
    vgg1=model1(Variable(dist))
    vgg1=vgg1.data.cpu()
    vggSimMatrix1 = (2*vgg0*vgg1 + 1)/(vgg0**2 + vgg1**2 + 1)

   
    
    #stage2
    vgg2=model2(Variable(ref))
    vgg2=vgg2.data.cpu()
    vgg3=model2(Variable(dist))
    vgg3=vgg3.data.cpu()
    vggSimMatrix2 = (2*vgg2*vgg3 + 1)/(vgg2**2 + vgg3**2 + 1)
    
    #stage3
    vgg4=model3(Variable(ref))
    vgg4=vgg4.data.cpu()
    vgg5=model3(Variable(dist))
    vgg5=vgg5.data.cpu()
    vggSimMatrix3 = (2*vgg4*vgg5 + 1)/(vgg4**2 + vgg5**2 + 1)
    
    
    #stage4
    vgg6=model4(Variable(ref))
    vgg6=vgg6.data.cpu()
    vgg7=model4(Variable(dist))
    vgg7=vgg7.data.cpu()
    vggSimMatrix4 = (2*vgg6*vgg7 + 1)/(vgg6**2 + vgg7**2 + 1)    
    
    #stage5
    vgg8=model5(Variable(ref))
    vgg8=vgg8.data.cpu()
    vgg9=model5(Variable(dist))
    vgg9=vgg9.data.cpu()
    vggSimMatrix5 = (2*vgg8*vgg9 + 1)/(vgg8**2 + vgg9**2 + 1)    
    
    ref=ref.cpu()
    dist=dist.cpu()
    gradientMap1=gm(dist)
    gradientMap2=gm(ref)
    gradientSimMatrix = (2*gradientMap1*gradientMap2 + 0.0023)/(gradientMap1**2 + gradientMap2**2 + 0.0023)


    vit1 = torch.load(vit_dist+'.pt').cpu()
    vit2 = torch.load(vit_ref+'.pt').cpu()
    vitSimMatrix = (2*vit1*vit2 + 0.000001) /(vit1**2 + vit2**2 + 0.000001)

    std1=torch.std(vggSimMatrix1.view(vggSimMatrix1.shape[0],-1),dim=1)
    std2=torch.std(vggSimMatrix2.view(vggSimMatrix2.shape[0],-1),dim=1)
    std3=torch.std(vggSimMatrix3.view(vggSimMatrix3.shape[0],-1),dim=1)
    std4=torch.std(vggSimMatrix4.view(vggSimMatrix4.shape[0],-1),dim=1)
    std5=torch.std(vggSimMatrix5.view(vggSimMatrix5.shape[0],-1),dim=1)
    std6=torch.std(gradientSimMatrix.view(gradientSimMatrix.shape[0],-1),dim=1)
    
    s1 = vggSimMatrix1.mean()*std1
    s2 = vggSimMatrix2.mean()*std2
    s3 = vggSimMatrix3.mean()*std3
    s4 = vggSimMatrix4.mean()*std4
    s5 = vggSimMatrix5.mean()*std5
    s6 = gradientSimMatrix.mean()*std6
    s7 = vitSimMatrix.mean()



    v1=torch.abs(vgg0-vgg1)
    v2=torch.abs(vgg2-vgg3)
    v3=torch.abs(vgg4-vgg5)
    v4=torch.abs(vgg6-vgg7)
    v5=torch.abs(vgg8-vgg9)
    v6=torch.abs(gradientMap1-gradientMap2)
    v7=torch.abs(vit1-vit2)

    n1=torch.std(v1)*0.2+s1
    n2=torch.std(v2)*0.2+s2
    n3=torch.std(v3)*0.2+s3
    n4=torch.std(v4)*0.2+s4
    n5=torch.std(v5)*0.2+s5
    n6=torch.std(v6)*0.2+s6
    n7=torch.std(v7)

    return 0.8*(n6+n3)+0.2*(n5+1-s7)
 
if __name__ == '__main__':

    dataset = MyDataset('./CSIQdmos_sorted1.csv','./csiq/all','./csiq/swin_iqa')
    #dataset = MyDataset('./live/LIVE.csv','./live','./live/swin_iqa')
    #dataset = MyDataset('./tid2013/TID_2013.csv','/data0/wuchao/IQAcode/codetest/tid2013','./tid2013/swin_iqa')


    device = torch.device('cuda:0')
    model = models.vgg16(pretrained=True).to(device)

    model1=make_model1()
    model2=make_model2()
    model3=make_model3()
    model4=make_model4()
    model5=make_model5()

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    

    len_dataset = dataset.__len__()
    pred = []
    label = []
    for i in range(len_dataset): 

        vit_dist,vit_ref,dist,ref,mos = dataset[i] 
        score = iqa(dist,ref,vit_dist,vit_ref)
        
        print("{}/{}____predict:{}, mos:{}".format(i,len_dataset,score,mos))
        pred.append(score)
        label.append(mos) 
    pred = torch.Tensor(pred).numpy()
    label = torch.Tensor(label).numpy()
    val_plcc = np.around(get_PLCC(pred, label),4)
    val_srocc = np.around(get_SROCC(pred, label),4)
    val_krocc = np.around(get_KROCC(pred, label),4)
    print("plcc: {}, srocc: {}, krocc: {}".format(val_plcc, val_srocc, val_krocc))

