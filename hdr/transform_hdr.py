
import cv2 as cv2
import numpy as np
import os

def to_hdr(img_lst, output_path):
    """
    Function to take in a list of images that will be transformed into HDR image
    Parameters:
    img_list: list of all the paths of the images
    output_path : path to write HDR image
    Return:
    HDR image is writted into the output path
    """
    path = output_path
    cv_img_list = [cv2.imread(img) for img in img_lst]
    
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(cv_img_list)
    res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')

    img_num = img_lst[0].split('_')[1].split('//')[-1]
    cv2.imwrite(os.path.join(path , str(img_num)+"_hdr.png"), res_mertens_8bit)
