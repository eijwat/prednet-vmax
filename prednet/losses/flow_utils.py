'''
Utilities for flow prediction, including:
    a torch module for OpenCV Farneback flow (FBFlow)
    image warping
    flow clipping
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2

class FBFlow(nn.Module):
    def __init__(self):
        super(FBFlow, self).__init__()
        # TODO: Allow flow parameters to be set with **kwargs

    def forward(self, im1, im2):
        batch_size = im1.shape[0]
        device = im1.device

        # Iterate through batches and calculate flows
        flows = []
        # NOTE: the loop indexes [0], not [b], so every flow in a batch is
        # computed from the first sample. Kept as in the legacy code; it only
        # matters for batchsize > 1.
        for b in range(batch_size):
            cvim1 = np.array(im1.cpu().detach()[0].permute(1,2,0))
            cvim2 = np.array(im2.cpu().detach()[0].permute(1,2,0))

            #cvim1 = cv2.cvtColor(cvim1, cv2.COLOR_BGR2GRAY) * 255
            #cvim2 = cv2.cvtColor(cvim2, cv2.COLOR_BGR2GRAY) * 255

            # TODO: HANDLE greyscale images
            cvim1 = cvim1[:,:,0]
            cvim2 = cvim2[:,:,0]

            flow = cv2.calcOpticalFlowFarneback(cvim1, cvim2, None,
                                                pyr_scale=.5,
                                                levels=8,
                                                winsize=40,
                                                iterations=3,
                                                poly_n=12,
                                                poly_sigma=1.5,
                                                flags=0)

            flow = torch.tensor(flow).permute(2,0,1).to(device).unsqueeze(0)
            flows.append(flow)


        flow = torch.cat(flows, dim=0)
        return flow

def clip_flow(flow, thresh, reduction_clip=False, scale_clip=None):
    # Either apply tanh soft thresholding (reduce = False)
    # OR apply hard tresholding, and set large flows to 0 (reduce = True)
    # OR if scale_clip==True, multiply flow by 1-\epsilon
    if scale_clip:
        epsilon = scale_clip
        return flow * (1 - epsilon)
        
    if not reduction_clip:
        thresh = float(thresh)
        eps = 1e-8
        scale = torch.tanh(flow.norm(dim=1) / thresh) * thresh / (flow.norm(dim=1) + eps)
    else:
        mask = flow.norm(dim=1) < thresh
        return mask.unsqueeze(1) * flow

    return scale.unsqueeze(1) * flow
    
def normalize_flow(flow, im):
    '''
    Normalize pixel-offset flow to absolute [-1, 1] flow
    input :
        flow : tensor (b, 2, h, w)
        im : tensor (b, c, h, w)
            This is just to get size of image
            (could be replaced by a size argument...)
    output :
        flow : tensor (b, h, w, 2) (for `F.grid_sample`)
    '''
    h = im.shape[2]
    w = im.shape[3]
    device = flow.device

    # Get base pixel coordinates (just "gaussian integers")
    base = torch.meshgrid(torch.arange(h), torch.arange(w))[::-1]
    base = torch.stack(base).float().to(device)
    size = torch.tensor([w, h]).float().to(device)

    # Convert to absolute coordinates
    flow = flow + base

    # Convert to [-1, 1] for grid_sample
    flow = -1 + 2.*flow/(-1 + size)[:,None,None]
    flow = flow.permute(0,2,3,1)
    
    return flow
    

def warp(im, flow, padding_mode='zeros'):
    '''
    requires absolute flow, normalized to [-1, 1]
        (see `normalize_flow` function)
    '''
    warped = F.grid_sample(im, flow, padding_mode=padding_mode, align_corners=True)

    return warped

