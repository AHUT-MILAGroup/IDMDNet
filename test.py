# coding:utf-8
import os
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import Fusion_dataset
from FusionNet_SEF_RES_DAM_FS_INN import FusionNet
from tqdm import tqdm
import numpy as np
from PIL import Image
from thop import profile  # 导入thop库的profile函数

def RGB2YCrCb(rgb_image):


    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    # 虽然表示的是y通道 
    # 但是确实是将rgb图转成灰度图的公式 和dataprocessing.py的最终结果是一样的
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):

    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out

# tensor to PIL Image
def tensor2img(img, is_norm=True):
  img = img.cpu().float().numpy()
  if img.shape[0] == 1:
    img = np.tile(img, (3, 1, 1))
  if is_norm:
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
  img = np.transpose(img, (1, 2, 0))  * 255.0
  return img.astype(np.uint8)

def save_img_single(img, name, is_norm=True):
  img = tensor2img(img, is_norm=True)
  img = Image.fromarray(img)
  img.save(name)

  

def main(ir_dir='./test_imgs/irf', vi_dir='./test_imgs/vis', save_dir='./', fusion_model_path='./fusionmodel_final.pth'):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    fusionmodel = FusionNet()
    fusionmodel = fusionmodel.to(device)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')
    test_dataset = Fusion_dataset('val', ir_path=ir_dir, vi_path=vi_dir)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for it, (img_vis, img_ir, name) in enumerate(test_bar):
            img_vis = img_vis.to(device)
            img_ir = img_ir.to(device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vis)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)
            fused_img= fusionmodel(vi_Y, img_ir)
            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            for k in range(len(name)):
                img_name = name[k]
                save_path = os.path.join(save_dir, img_name)
                save_img_single(fused_img[k, ::], save_path)
                test_bar.set_description('Fusion {0} Sucessfully!'.format(name[k]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run IDMDNet with pytorch')
    ## model
    parser.add_argument('--model_path', '-M', type=str, default='./model_SEF_RES_DAM_FS_INN/Fusion_INN1/fusion_model.pth')
    ## dataset
    parser.add_argument('--ir_dir', '-ir_dir', type=str, default='')
    parser.add_argument('--vi_dir', '-vi_dir', type=str, default='')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='')
    parser.add_argument('--batch_size', '-B', type=int, default=2)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % ('IDMDNet', args.gpu))
    main(ir_dir=args.ir_dir, vi_dir=args.vi_dir, save_dir=args.save_dir, fusion_model_path=args.model_path)
