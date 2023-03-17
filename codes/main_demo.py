import argparse
import numpy as np

parser = argparse.ArgumentParser(description='W-NetPan: Double-U Network for Inter-sensor Self-supervised Pan-sharpening')
parser.add_argument('-ep', '--epochs', dest='NUM_EPOCHS', default=20000, type=int)
parser.add_argument('-lr', '--learning-rate', dest='LERNING_RATE', default=0.001, type=float)
parser.add_argument('-gpu', '--gpu', dest='GPU', default="0", type=str)
parser.add_argument('-losses', '--loss_weights', dest='LOSSES', nargs='+', default=[0.5, 1, 1, 1, 1], type=float) 
parser.add_argument('-ncc_win', '--ncc_window_size', dest='NCC_WIN', default=9, type=int) 

args = parser.parse_args()
NUM_EPOCHS = args.NUM_EPOCHS
LERNING_RATE = args.LERNING_RATE
GPU = args.GPU
LOSSES = list(args.LOSSES)
NCC_WIN = args.NCC_WIN



from data import load_data

xtra = load_data('ms.tif', 0.5)
ytra = load_data('pan.tif', 0.5)
ztra = load_data('ms_gt.tif', 0.5)



import torch
from model import WNetPan

RATIO = ytra.shape[2]//xtra.shape[2]
INPUT_PAN_SIZE = (ytra.shape[2], ytra.shape[3])
MS_BANDS = xtra.shape[1]
PAN_BANDS = ytra.shape[1]

mynet = WNetPan(RATIO, INPUT_PAN_SIZE, MS_BANDS, PAN_BANDS)
    
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:'+GPU if torch.cuda.is_available() else 'cpu')

model = mynet.to(device)
model.train()

optimizer = torch.optim.Adam([
    {'params': model.flow.parameters()},
    {'params': model.unet_model.parameters()},
    {'params': model.hr.parameters()},
    ], lr=LERNING_RATE)



from losses import Grad, NCC, MSE

losses  = [Grad('l2').loss] + [NCC(win=NCC_WIN,GPU=GPU).loss] + [MSE().loss] + [NCC(win=NCC_WIN,GPU=GPU).loss] + [MSE().loss] 
weights = LOSSES

        
from torch.utils.data import TensorDataset, DataLoader

tensor_xtra = torch.Tensor(xtra.astype(np.float32)) # transform to torch tensors
tensor_ytra = torch.Tensor(ytra.astype(np.float32))
tensor_ztra = torch.Tensor(ztra.astype(np.float32))
tradata = TensorDataset(tensor_xtra,tensor_ytra,tensor_ztra) # create datset
traloader = DataLoader(dataset=tradata,batch_size=1, shuffle=False) # create dataloader


best_loss = None
for epoch in range(NUM_EPOCHS):

    gt_mse_monitor = 0
    loss_list = np.array([0.0 for _ in range(len(losses))])
    str_improved_loss = ""
    
    for iteration, batch in enumerate(traloader):

        in_ms, in_pan, in_gt = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        
        y_pred  = model(in_ms, in_pan) # [flow, pan_gt, ini_pan, ms_gt, end_ms, end_pan, result]                               
        hr = y_pred[-1]
        
        y_true = [in_pan, y_pred[1], y_pred[3], in_pan, in_pan]
        y_pred = [y_pred[0], y_pred[2], y_pred[4], y_pred[5], y_pred[5]] 
        
        loss = 0
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list[n] += curr_loss.item()
            loss += curr_loss

        gt_mse_monitor += MSE().loss(in_gt,hr)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        

    loss_list = loss_list/(iteration+1)
    loss = np.sum(loss_list)
    fnum = lambda x:'{:.9f}'.format(x)
    str_loss_info = 'loss: %.9f  (%s)' % (loss, ', '.join(map(fnum,loss_list)))


    if best_loss==None or loss<best_loss: # saving the best model                                    
        
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, 'model.pth')
        
        best_loss = loss
        best_epoch = epoch+1
        str_improved_loss = "*"


    str_epoch_info = 'Epoch: %06d' % (epoch + 1)
    str_best_epoch_info = 'BEST(#{:06d},{:.9f})'.format(best_epoch, best_loss)
    str_gt_mse_monitor = 'GT-MSE({:.9f})'.format(gt_mse_monitor/(iteration+1))
    print('  '.join((str_epoch_info, str_loss_info, str_gt_mse_monitor, str_best_epoch_info, str_improved_loss)), flush=True)
