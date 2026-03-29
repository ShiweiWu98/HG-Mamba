import math
from math import exp
from skimage import color
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(pred, gt):
    pred = pred.clamp(0, 1).cpu().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)

def ciede2000(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Input tensors must have the same dimensions")
    
    img1 = tensor1.clone().clamp_(0,1.0).permute(1, 2, 0).cpu().numpy()
    img2 = tensor2.clone().permute(1, 2, 0).cpu().numpy()

    if img1.max() > 1.0 or img1.min() < 0.0 or img2.max() > 1.0 or img2.min() < 0.0:
        print(img1)
        print(img2)
        raise ValueError("Input tensors must have values in the range [0, 1]")

    lab1 = color.rgb2lab(img1)
    lab2 = color.rgb2lab(img2)
    
    delta_e = color.deltaE_ciede2000(lab1, lab2) 
    
    mean_delta_e = np.mean(delta_e)
    return mean_delta_e


if __name__ == "__main__":
    torch.manual_seed(666)
    x = torch.randn((3,3,256,256))
    y = torch.randn((3,3,256,256))
    print(psnr(x,y))
    print(psnr(x[0],y[0]))