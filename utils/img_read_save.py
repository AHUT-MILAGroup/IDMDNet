import numpy as np
import cv2
import os
from skimage.io import imsave

def image_read_cv2(path, mode):
    img_BGR = cv2.imread(path).astype('float32')
    # 原来的代码 但是由于我的图片是红外图像 没有对应的mode 因此mode是none 直接返回img 
    # assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    elif mode == 'None':
        img = img_BGR
    return img

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    image = image.astype(np.uint8)
    imsave(os.path.join(savepath, "{}.png".format(imagename)),image)