import cv2
import os
import ast
import io
import torch
import ujson as json
import nori2 as nori
import numpy as np
import random
# import imgaug.augmenters as iaa
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.path = 's3://huangzhewei/data/slomo/'
        self.dataset_name = dataset_name
        self.nf = nori.Fetcher()
        self.h = 256
        self.w = 448
        xx = np.arange(0, self.w).reshape(1,-1).repeat(self.h,0)
        yy = np.arange(0, self.h).reshape(-1,1).repeat(self.w,1)
        self.grid = np.stack((xx,yy),2).copy()
        self.mp = {}
        f = json.load(open('/data/challenge/test.json'))
        for i in f:
            self.mp[i[0]] = i[1]
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def readimg(self, x):
        return cv2.imdecode(np.frombuffer(io.BytesIO(self.nf.get(self.mp[x])).getbuffer(), np.uint8), cv2.IMREAD_COLOR)
    
    def load_data(self):
        self.train_data = []
        self.val_data = []
        self.flow_data = []
        path = '/data/anime/datasets/train_10k/'
        for d in os.listdir(path):
            img0 = (path + d + '/frame1.jpg')
            img1 = (path + d + '/frame2.jpg')
            img2 = (path + d + '/frame3.jpg')
            self.train_data.append((1, (img0, img1, img2)))
        def read(name):            
            with nori.smart_open(self.path + 'vimeo_flow_largev13.nori/' + name) as f:
                for data in f:
                    try:
                        id = eval(data)['DataID']
                        self.flow_data.append(id)
                    except:
                        pass
        with open("data/adobe240fps_folder_{}.txt".format('train')) as f:
            data = f.readlines()
            for l in data:
                l = l.strip('\n')
                path = '/data/adobe240/frame/{}/{}'.format('train', l)
                interval = 7                    
                for i in range(0, len(os.listdir(path)) - 14, interval):
                    data_tuple = []
                    for j in range(7):
                        data_tuple.append('{}/{}.png'.format(path, i+j))
                    self.train_data.append((2, data_tuple))
                    data_tuple = []
                    for j in range(7):
                        data_tuple.append('{}/{}.png'.format(path, i+j*2))
                    self.train_data.append((2, data_tuple))
                            
        with nori.smart_open('s3://chenmingrui/datasets/vimeo_septuplet/vimeo_septuplet.json') as f:
            for data in json.load(f):
                self.train_data.append((0, data))
        with nori.smart_open(self.path + '123test.json') as f:
            for data in json.load(f):
                self.val_data.append((0, data))
        if self.dataset_name == 'train':
            self.meta_data = self.train_data
        else:
            self.meta_data = self.val_data
        self.nr_sample = len(self.meta_data)        

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index, training=False):
        data = self.meta_data[index][1]
        datasetid = self.meta_data[index][0]
        if datasetid == 1:
            img0 = cv2.imread(data[0])
            gt = cv2.imread(data[1])
            img1 = cv2.imread(data[2])
            if np.random.uniform(0, 1) < 0.5:
                img0 = cv2.resize(img0, (480, 270), interpolation=cv2.INTER_LINEAR)
                img1 = cv2.resize(img1, (480, 270), interpolation=cv2.INTER_LINEAR)
                gt = cv2.resize(gt, (480, 270), interpolation=cv2.INTER_LINEAR)
            else:
                img0 = cv2.resize(img0, (960, 540), interpolation=cv2.INTER_LINEAR)
                img1 = cv2.resize(img1, (960, 540), interpolation=cv2.INTER_LINEAR)
                gt = cv2.resize(gt, (960, 540), interpolation=cv2.INTER_LINEAR)
            step = 0.5
        elif datasetid == 2:
            ind = [0, 1, 2, 3, 4, 5, 6]
            random.shuffle(ind)
            ind = ind[:3]
            ind.sort()
            img0 = self.readimg(data[ind[0]])
            gt = self.readimg(data[ind[1]])
            img1 = self.readimg(data[ind[2]])
            step = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)
        elif not training:
            img0 = np.frombuffer(self.nf.get(data[0]), dtype='uint8').reshape(256, 448, 3)
            gt = np.frombuffer(self.nf.get(data[1]), dtype='uint8').reshape(256, 448, 3)
            img1 = np.frombuffer(self.nf.get(data[2]), dtype='uint8').reshape(256, 448, 3)
            step = 0.5
        else:
            ind = [0, 1, 2, 3, 4, 5, 6]
            random.shuffle(ind)
            ind = ind[:3]
            ind.sort()
            img0 = cv2.imdecode(np.frombuffer(io.BytesIO(self.nf.get(data[ind[0]])).getbuffer(), np.uint8), cv2.IMREAD_COLOR)
            gt = cv2.imdecode(np.frombuffer(io.BytesIO(self.nf.get(data[ind[1]])).getbuffer(), np.uint8), cv2.IMREAD_COLOR)
            img1 = cv2.imdecode(np.frombuffer(io.BytesIO(self.nf.get(data[ind[2]])).getbuffer(), np.uint8), cv2.IMREAD_COLOR)
            step = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)
        return img0, gt, img1, step
            
    def __getitem__(self, index):
        if self.dataset_name == 'train':
            img0, gt, img1, timestep = self.getimg(index, True)
            if np.random.uniform(0, 1) < 0.5:
                p = np.random.choice([1.5, 2.0])
                h, w = int(256 * p), int(448 * p)
                img0 = cv2.resize(img0, (w, h), interpolation=cv2.INTER_CUBIC)
                img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_CUBIC)
                gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_CUBIC)
            img0, gt, img1 = self.aug(img0, gt, img1, 224, 224)
            if random.uniform(0, 1) < 0.2:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            if random.uniform(0, 1) < 0.2:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep
        else:
            img0, gt, img1, timestep = self.getimg(index, training=False)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0), timestep
    
if __name__ == '__main__':
    ds = DataLoader(VimeoDataset('train'))
