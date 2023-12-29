import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from torch.autograd import Variable
import skimage.measure as skim
import scipy.misc
from skimage.color import rgb2yuv, yuv2rgb
from yuv_frame_io import YUV_Read,YUV_Write
import lpips
loss_fn_alex = lpips.LPIPS(net='alex').cuda()

torch.distributed.init_process_group(backend="nccl", world_size=4)
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

from model import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from util import *
from torch.utils.data.distributed import DistributedSampler

exp = os.path.abspath('.').split('/')[-1]

log_path = '../../train_log1116/{}'.format(exp)
if local_rank == 0:
    writer = SummaryWriter(log_path + '/train')
    writer_val = SummaryWriter(log_path + '/validate')

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
    return 1e-4 * mul

def train(model):
    step = 0
    nr_eval = 0
    dataset = VimeoDataset('train')
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    dataset_val = VimeoDataset('validation')
    val_data = DataLoader(dataset_val, batch_size=8, pin_memory=True, num_workers=16)
    # model.load_model('../../train_log1116/RIFEv50', local_rank)
    evaluate(model, val_data, 0)
    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            distill = True
            learning_rate = get_learning_rate(step)
            pred, info = model.update(imgs, gt, learning_rate, training=True, distill=distill, timestep=timestep)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', info['loss_l1'], step)
                writer.add_scalar('loss/tea', info['loss_tea'], step)
                writer.add_scalar('loss/cons', info['loss_cons'], step)
            if step % 1000 == 1 and local_rank == 0:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                for i in range(4):
                    imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1), step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                writer.flush()
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, info['loss_l1']))
            step += 1
        nr_eval += 1
        if nr_eval % 10 == 0:
            evaluate(model, val_data, step)
        model.save_model(log_path, local_rank)    
        dist.barrier()

def evaluate(model, val_data, nr_eval):
    loss_l1_list = []
    loss_cons_list = []
    loss_tea_list = []
    psnr_list = []
    lpips_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    for i, data in enumerate(val_data):
        data_gpu, timestep = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.        
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
            merged_img = info['merged_tea']
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_cons_list.append(info['loss_cons'].cpu().numpy())
        for j in range(gt.shape[0]):
            if local_rank == 0:
                lpips = loss_fn_alex(gt[j] * 2 - 1, pred[j] * 2 - 1).detach().cpu().data
                lpips_list.append(lpips)
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr)
            if local_rank == 0:
                lpips_list.append(lpips)
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        if i == 0 and local_rank == 0:
            for j in range(4):
                imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')
    
    eval_time_interval = time.time() - time_stamp

    if local_rank != 0:
        return
    writer_val.add_scalar('benchmark/psnr', np.array(psnr_list).mean(), nr_eval)
    writer_val.add_scalar('benchmark/psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)
    writer_val.add_scalar('benchmark/lpips', np.array(lpips_list).mean(), nr_eval)
    '''
    name_list = [
        ('/data/slomo/HD_dataset/HD720p_GT/parkrun_1280x720_50.yuv', 720, 1280),
        ('/data/slomo/HD_dataset/HD720p_GT/shields_1280x720_60.yuv', 720, 1280),
        ('/data/slomo/HD_dataset/HD720p_GT/stockholm_1280x720_60.yuv', 720, 1280),
        ('/data/slomo/HD_dataset/HD1080p_GT/BlueSky.yuv', 1080, 1920),
        ('/data/slomo/HD_dataset/HD1080p_GT/Kimono1_1920x1080_24.yuv', 1080, 1920),
        ('/data/slomo/HD_dataset/HD1080p_GT/ParkScene_1920x1080_24.yuv', 1080, 1920),
        ('/data/slomo/HD_dataset/HD1080p_GT/sunflower_1080p25.yuv', 1080, 1920),
        ('/data/slomo/HD_dataset/HD544p_GT/Sintel_Alley2_1280x544.yuv', 544, 1280),
        ('/data/slomo/HD_dataset/HD544p_GT/Sintel_Market5_1280x544.yuv', 544, 1280),
        ('/data/slomo/HD_dataset/HD544p_GT/Sintel_Temple1_1280x544.yuv', 544, 1280),
        ('/data/slomo/HD_dataset/HD544p_GT/Sintel_Temple2_1280x544.yuv', 544, 1280),
    ]
    tot = 0.
    for data in name_list:
        psnr_list = []
        name = data[0]
        h = data[1]
        w = data[2]
        if 'yuv' in name:
            Reader = YUV_Read(name, h, w, toRGB=True)
        for index in range(0, 100, 2):
            if 'yuv' in name:
                IMAGE1, success1 = Reader.read(index)
                gt, _ = Reader.read(index + 1)
                IMAGE2, success2 = Reader.read(index + 2)
                if not success2:
                    break
            I0 = torch.from_numpy(np.transpose(IMAGE1, (2,0,1)).astype("float32") / 255.).cuda().unsqueeze(0)
            I1 = torch.from_numpy(np.transpose(IMAGE2, (2,0,1)).astype("float32") / 255.).cuda().unsqueeze(0)

            if h == 720:
                pad = 24
            elif h == 1080:
                pad = 4
            else:
                pad = 16
            pader = torch.nn.ReplicationPad2d([0, 0, pad, pad])
            I0 = pader(I0)
            I1 = pader(I1)
            with torch.no_grad():
                imgs = torch.cat((I0, I1), 1)
                flow, mask, merged, _, _ = model.flownet(imgs, scale=[8, 4, 2])
                pred = model.predict(imgs, flow, merged, training=False)
                pred = pred[:, :, pad: -pad]
            out = (np.round(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)).astype('uint8')
            if 'yuv' in name:
                diff_rgb = 128.0 + rgb2yuv(gt / 255.)[:, :, 0] * 255 - rgb2yuv(out / 255.)[:, :, 0] * 255
                mse = np.mean((diff_rgb - 128.0) ** 2)
                PIXEL_MAX = 255.0
                psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
            psnr_list.append(psnr)
        tot += np.mean(psnr_list)
    writer_val.add_scalar('/data/slomo/HD_psnr', tot / len(name_list), nr_eval)
    name = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
    IE_list = []
    for i in name:
        i0 = cv2.imread('/data/other-data/{}/frame10.png'.format(i)).transpose(2, 0, 1) / 255.
        i1 = cv2.imread('/data/other-data/{}/frame11.png'.format(i)).transpose(2, 0, 1) / 255.
        gt = cv2.imread('/data/other-gt-interp/{}/frame10i11.png'.format(i))
        h, w = i0.shape[1], i0.shape[2]
        imgs = torch.zeros([1, 6, 480, 640])
        ph = (480 - h) // 2
        pw = (640 - w) // 2
        imgs[:, :3, :h, :w] = torch.from_numpy(i0).unsqueeze(0).float()
        imgs[:, 3:, :h, :w] = torch.from_numpy(i1).unsqueeze(0).float()
        I0 = imgs[:, :3].cuda()
        I2 = imgs[:, 3:].cuda()
        with torch.no_grad():
            flow, mask, merged, _, _ = model.flownet(imgs)
            pred = merged[2] # model.predict(imgs, flow[2], merged[2], training=False)
        out = pred[0].cpu().numpy().transpose(1, 2, 0)
        out = np.round(out[:h, :w] * 255)
        IE_list.append(np.abs((out - gt * 1.0)).mean())
    '''
    print('eval time: {}'.format(eval_time_interval)) 
    writer_val.add_scalar('loss/l1', np.array(loss_l1_list).mean(), nr_eval)
    writer_val.add_scalar('loss/tea', np.array(loss_tea_list).mean(), nr_eval)
    writer_val.add_scalar('loss/cons', np.array(loss_cons_list).mean(), nr_eval)
    # writer_val.add_scalar('IE', np.array(IE_list).mean(), nr_eval)
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=600, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    args = parser.parse_args()
    # args.step_per_epoch = 51313 // args.batch_size    
    model = Model(args.local_rank)
    train(model)
        
