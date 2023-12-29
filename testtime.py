
import cv2
import torch.nn as nn
# from PWCNet import *
# from LiteNet import *
from Flownet import *
import time
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# flownet = FlownetCas().to(device)
model = Model()
# img = torch.rand([1, 3, 720, 1280]).to(device) / 255.
model.eval()
'''
for i in range(30):
    with torch.no_grad():
        img = torch.rand([1, 3, 480, 640]).to(device) / 255.
        imgs = torch.cat((img, img), 1)
        flow, _ = model.flownet(imgs)
        flow = F.interpolate(flow, scale_factor=2, mode="bilinear")
        p = model.predict(img, img, flow, training=False)
'''
tot_time = 0
# img = torch.rand([1, 3, 720, 1280]).to(device) / 255.
# flow = torch.zeros([1, 2, 720, 1280]).to(device)
torch.backends.cudnn.benchmark = False
cnt = 0
path = '/data/slomo/demo/puck.mp4'
videoCapture = cv2.VideoCapture(path)
success, frame = videoCapture.read()
frame = frame[:720, :1280]
tot = 0.
while success:
    lastframe = frame
    success, frame = videoCapture.read()
    frame = frame[:720, :1280]
    with torch.no_grad():
        I0 = torch.from_numpy(lastframe.transpose(2, 0, 1)).float().unsqueeze(0)
        I1 = torch.from_numpy(frame.transpose(2, 0, 1)).float().unsqueeze(0)
        imgs = torch.cat((I0, I1), 1).to(device) / 255.
        for i in range(50):
            flow, mask, merged, _, _ = model.flownet(imgs)
#            pred = model.predict(imgs, flow[2], merged[2], training=False)
        torch.cuda.synchronize()
        time_stamp = time.time()
        for i in range(50):
            flow, mask, merged, _, _ = model.flownet(imgs)
#            pred = model.predict(imgs, flow[2], merged[2], training=False)
        torch.cuda.synchronize()
        print((time.time() - time_stamp) / 50)
        break
