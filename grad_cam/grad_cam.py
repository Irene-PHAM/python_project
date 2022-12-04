
# All imports
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import softmax, interpolate
from torch import Tensor
import torch.nn.functional as F
from torchvision.io.image import read_image
from torchvision.transforms.functional import  resize, to_pil_image
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
from typing import List, Tuple, Dict
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_model(path_to_model, device):
    if device.type == 'cpu':
        model = torch.load(path_to_model, map_location=torch.device('cpu'))
    else:
        model = torch.load(path_to_model)
    return model.eval()
    

class GradCam:
    """
    GradCam class to perform grad-cam algorithm on input images to visualise extracted features used
    used for classification task.
    """
    def __init__(self, model, img_path_list) -> None:
        """ Initialise with loaded VGG-19 model and list of images
        Parameters:
            model: loaded VGG-19 model
            img_path_list: list of Moire images under various exposure
        """
        self.model = model
        self.img_path_list = img_path_list
        # Use GradCAMpp model from torchcam
        # 'features' layer is the default layers of VGG-19 model that extracting features
        self.cam_extractor = GradCAMpp(self.model, 'features')
    
    @staticmethod
    def img_to_tensor(img_path:str) -> Tensor:
        """ Transform image to tensor
        Parameters:
            img_path : path of the image
        Return:
            Image tensor resize into 224*224 size tensor
        """
        img = read_image(img_path)
        input_tensor = resize(img, (224, 224)).float().to(device)
        return input_tensor
    
    def get_read_img(self) -> List:
        """ Read in image from input image list
        Return:
            list of image read in 
        """
        return [read_image(path) for path in self.img_path_list]
    
    def fuse_cams(self) -> Tensor:
        """ Stack interpolated output of grad-cam algorithm on list of input images together
        Return:
        Multiple exposure stacked tensor 
        """
        # Retrieve the list of CAM tensor
        cams = self.multi_image_cam_tensor_list()
        _shape = tuple(map(max, zip(*[tuple(cam.shape[1:]) for cam in cams]))) 
        if len(cams) ==1:
            return None
        # Interpolate all CAMs
        interpolation_mode = "bilinear" if cams[0].ndim == 3 else "trilinear" if cams[0].ndim == 4 else "nearest"
        scaled_cams = [
            F.interpolate(cam.unsqueeze(1), _shape, mode=interpolation_mode, align_corners=False) for cam in cams
        ]
        # Fuse them
        return torch.stack(scaled_cams).max(dim=0).values.squeeze(1)

    def get_layered_cam_array(self) -> Tuple:
        """ Get a CAM array from the combination of highlighted part of CAM from single exposure image
        Return:
            A final layered CAM with pixels of the highest value from the input images
        """
        # Get list of CAM array
        list_of_cams = self.get_multi_image_cam_array_list()
        # Get the flatten list
        list_of_cam_flattened = [list(np.concatenate(cam).flat) for cam in list_of_cams]
        # Get the highest pixel value from all layers in each respective position and combine them
        # together for the final layer
        layered_tuple=list(map(max, zip(*[tuple(cam) for cam in list_of_cam_flattened]))) 
        return np.reshape(layered_tuple, (-1, 7))
    def count_highlighted_pixel(self) -> Dict:
        """ Count the highlighted pixel (with value >0.5) from each input image and the fused layer
        Return:
            Dictionary of name of the image as key and count of highlighted pixel as value
        """
        multi_cam_array_list = self.get_multi_image_cam_array_list()
        multi_count = {k:(self.calculate_bright_pixel(v)) for k,v in zip(self.img_path_list, multi_cam_array_list )}
        layered_cam_array = self.get_layered_cam_array()
        layered_cam_array_count = self.calculate_bright_pixel(layered_cam_array)
        multi_count['fused_layer'] =  layered_cam_array_count 
        return multi_count

    @staticmethod
    def calculate_bright_pixel(array_list: List) -> int:
        """ Count number of highlighted pixel in the array, highlighted pixel is considered as 
        value that is >= 0.5
        Return:
            Count of the pixel >=0.5
        """
        count = 0
        for array in array_list:
            count +=len(list(filter (lambda x : x >= 0.5, array)))
        return count    
    
    def get_multi_image_cam_array_list(self) -> List:
        """ Get a list of CAM in array from input images
        Return: List of CAM arrays
        """
        tensor_list = self.get_multi_image_cam_tensor_list()
        return [tensor.squeeze(0).detach().cpu().numpy() for tensor in tensor_list]
        
    def get_multi_image_cam_tensor_list(self) -> List:
        """ Get a list of CAM in tensor from input images
        Return: List of CAM tensors
        """
        tensor_list = []
        for img in self.img_path_list:
            input_tensor = self.img_to_tensor(img)
            out = self.model(input_tensor.unsqueeze(0))
            cams = self.cam_extractor(0,out)
            tensor_list.append(cams[0])
        return tensor_list
            
    def overlay_on_image(self):
        """ Visualise CAM(features that used by model to classify images) on images. 
            This helps us see what region the model target to decide class of the input image
        """ 
        for name, img, cam in zip( self.img_path_list, self.get_read_img(), self.get_multi_image_cam_tensor_list()):
            result = overlay_mask(to_pil_image(img), to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)
            plt.imshow(result); plt.axis('off'); plt.title(name); plt.show()

class GradCamSingle:
    """
    GradCam class to perform grad-cam algorithm on input images to visualise extracted features used
    used for classification task.
    """
    def __init__(self, model, img) -> None:
        """ Initialise with loaded VGG-19 model and list of images
        Parameters:
            model: loaded VGG-19 model
            img: Moire images
        """
        self.model = model
        self.img = img
        # Use GradCAMpp model from torchcam
        # 'features' layer is the default layers of VGG-19 model that extracting features
        self.cam_extractor = GradCAMpp(self.model, 'features')
    
    @staticmethod
    def img_to_tensor(img_path:str) -> Tensor:
        """ Transform image to tensor
        Parameters:
            img_path : path of the image
        Return:
            Image tensor resize into 224*224 size tensor
        """
        img = read_image(img_path)
        input_tensor = resize(img, (224, 224)).float().to(device)
        return input_tensor
    
    def get_read_img(self) -> List:
        """ Read in image """
        return read_image(self.img)
        
    def get_cam(self) -> List:
        """ Get grad-cam from input image
        """
        input_tensor = self.img_to_tensor(self.img)
        out = self.model(input_tensor.unsqueeze(0))
        cams = self.cam_extractor(0,out)
        return cams
            
    def overlay_on_image(self):
        """ Visualise CAM(features that used by model to classify images) on input image.
            This helps us see what region the model target to decide class of the input image
        """ 
        for name, cam in zip(self.cam_extractor.target_names, self.get_cam()):
            result = overlay_mask(to_pil_image(self.get_read_img()), to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)
            return result

