# code is based on https://github.com/katerakelly/pytorch-maml
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.utils.data.sampler import Sampler

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def mini_imagenet_folders():
    
    
    metatrain_mods = ['8PSK','AM-DSB','BPSK','GFSK','PAM4','QAM64']
    
    metatest_mods = ['QPSK','AM-SSB','QAM64','CPFSK','WBFM']
    
    random.seed(1)
    random.shuffle(metatrain_mods)
    random.shuffle(metatest_mods)
    
    

    return metatrain_mods, metatest_mods

class IQTask(object):

    def __init__(self, character_folders, num_classes, train_num, test_num, path):

        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        self.path = path
        
        class_folders = random.sample(self.character_folders,self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        snr = 18

        with open(path,'rb') as f:
            rfdata = pickle.load(f,encoding='latin1')
            
        # Make empty array for concat
        self.train_samples= np.array([]).reshape(0,2,128)
        self.test_samples = np.array([]).reshape(0,2,128)
        
        self.train_labels =[]
        self.test_labels = []
        
        for mod in class_folders:
            temp = rfdata[mod,snr]
            # print(temp.shape)
            ind = np.arange(temp.shape[0])
            np.random.shuffle(ind)
            temp = temp[ind]
            self.train_samples = np.vstack([self.train_samples, temp[:train_num]])
            # print(train_samples.shape)
            self.test_samples = np.vstack([self.test_samples, temp[train_num:train_num+test_num]])
            for i in range(train_num):
                self.train_labels.append(labels[mod])
            for j in range(test_num):
                self.test_labels.append(labels[mod])
    
        
        # return os.path.join(*sample.split('/')[:-1])


class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.iq_data = self.task.train_samples if self.split == 'train' else self.task.test_samples
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.iq_data)

    def _getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")
        
class MiniImagenet(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(MiniImagenet, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        iq = self.iq_data[idx]
        if self.transform is not None:
            iq = self.transform(iq)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return iq, label


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_mini_imagenet_data_loader(task, num_per_class=1, split='train',shuffle = False):
    #normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    dataset = MiniImagenet(task,split=split,transform=transforms.Compose([transforms.ToTensor()]))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader

