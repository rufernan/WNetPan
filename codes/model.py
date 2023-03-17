import torch
import torch.nn as nn
from torch.distributions.normal import Normal



class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        KS = 3; PD = 1
        K1=16; K2=32

        # encoder
        self.Conv1 = nn.Conv2d(in_channels=img_ch,out_channels=K1,kernel_size=KS,stride=2,padding=PD)
        self.Conv2 = nn.Conv2d(in_channels=K1,out_channels=K2,kernel_size=KS,stride=2,padding=PD)
        self.Conv3 = nn.Conv2d(in_channels=K2,out_channels=K2,kernel_size=KS,stride=2,padding=PD)
        self.Conv4 = nn.Conv2d(in_channels=K2,out_channels=K2,kernel_size=KS,stride=2,padding=PD)
        
        # decoder
        self.Conv5 = nn.Conv2d(in_channels=K2,out_channels=K2,kernel_size=KS,stride=1,padding=PD)
        self.Conv6 = nn.Conv2d(in_channels=K2+K2,out_channels=K2,kernel_size=KS,stride=1,padding=PD)
        self.Conv7 = nn.Conv2d(in_channels=K2+K2,out_channels=K2,kernel_size=KS,stride=1,padding=PD)
        self.Conv8 = nn.Conv2d(in_channels=K2+K2,out_channels=K2,kernel_size=KS,stride=1,padding=PD)
        
        # extra Conv
        self.Conv9 = nn.Conv2d(in_channels=K2+K1,out_channels=K1,kernel_size=KS,stride=1,padding=PD)
        self.Conv10 = nn.Conv2d(in_channels=K1,out_channels=output_ch,kernel_size=KS,stride=1,padding=PD)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.activation = nn.LeakyReLU(0.2)


    def forward(self,x):
        
        # encoding path
        x1 = self.activation(self.Conv1(x))
        x2 = self.activation(self.Conv2(x1))
        x3 = self.activation(self.Conv3(x2))
        x4 = self.activation(self.Conv4(x3))

        # decoding + concat path
        d5 = self.activation(self.Conv5(x4))
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.upsample(d5)

        d6 = self.activation(self.Conv6(d5))
        d6 = torch.cat((x3,d6),dim=1)
        d6 = self.upsample(d6)

        d7 = self.activation(self.Conv7(d6))
        d7 = torch.cat((x2,d7),dim=1)
        d7 = self.upsample(d7)

        d8 = self.activation(self.Conv8(d7))
        d8 = torch.cat((x1,d8),dim=1)
        d8 = self.upsample(d8)

        e1 = self.activation(self.Conv9(d8))
        e2 = self.activation(self.Conv10(e1))

        return e2





class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)


    def forward(self, src, flow):

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nn.functional.grid_sample(src, new_locs, align_corners=True, mode=self.mode)





class WNetPan(nn.Module):
    def __init__(self, RATIO=4, pan_img_shape=(512,512), ms_ch=4, pan_ch=1, backbone_out_ch=32):
   
        super(WNetPan, self).__init__()

        self.ratio = RATIO
        self.ms_ch = ms_ch
        
        self.up_ms = nn.Upsample(scale_factor=RATIO, mode='bicubic')
        
        self.unet_model0 = U_Net(
            img_ch=pan_ch+pan_ch,
            output_ch=backbone_out_ch
        )

        self.flow = nn.Conv2d(in_channels=backbone_out_ch, out_channels=len(pan_img_shape), kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.transformer = SpatialTransformer(pan_img_shape)
        
        self.unet_model = U_Net(
            img_ch=ms_ch+pan_ch,
            output_ch=backbone_out_ch
        )
        
        self.hr = nn.Conv2d(in_channels=backbone_out_ch, out_channels=ms_ch, kernel_size=1, padding=0)
        self.hr_ms = nn.AvgPool3d((1,RATIO,RATIO))
        self.hr_pan = nn.AvgPool3d((ms_ch,1,1))
        
        self.lrelu = nn.LeakyReLU(0.2)
        
        

    def forward(self, ms, pan):
         
        if self.ratio>1:
            ms = self.up_ms(ms)

        ms_pan = torch.mean(ms, dim=1, keepdim=True)
        
        x = torch.cat([ms_pan, pan], dim=1)
        x = self.unet_model0(x)
        flow_field = self.flow(x)        
        ini_pan = self.transformer(ms_pan, flow_field)
        ms = self.transformer(ms, flow_field)
        
        ms_gt = self.hr_ms(ms)
        
        pan_rep = pan.repeat(1,ms.shape[1],1,1)
        residuals = torch.sub(pan_rep, ms)

        x = torch.cat([ms, pan], dim=1)
        x = self.unet_model(x)
        x = self.lrelu(self.hr(x)) + ms + pan_rep
        
        end_ms = self.hr_ms(x)
        end_pan = torch.mean(x,dim=1, keepdim=True)

        return flow_field, pan, ini_pan, ms_gt, end_ms, end_pan, x
