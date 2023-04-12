import cv2
import numpy as np
import torch
import src
from PIL import Image
ENCODER = 'resnet152'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'

class Segmentator:
    def __init__(self, path_model: str):
        self.model = src.UnetPlusPlus(
            encoder_name='resnet152',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation= 'sigmoid')
        checkpoint = torch.load(path_model)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        self.input_resize = (768, 768)
        self.output_resize = (980, 615)
        print('ok')
    def predict(self, image):
        image = cv2.resize(
            image,
            (1280, 872),
            interpolation=cv2.INTER_AREA
        )
        image = image[135:750, 145:1125]
        preprocessed_image = self.preproc_input(image)
        preprocessed_image = preprocessed_image.to(self.device)
        with torch.no_grad():
            outputs = self.model.forward(preprocessed_image)
        return cv2.addWeighted(
            image,
            1,
            self.preproc_output(outputs.cpu().detach().numpy()),
            0.6,
            0
        )

    def preproc_input(self, image):
        b, g, r = cv2.split(image)
        image = cv2.merge([b, g, r])
        image = cv2.resize(
            image,
            self.input_resize,
            interpolation=cv2.INTER_AREA
        ).astype('float32')
        image = (np.rint(image)).astype(np.uint8)
        image = np.moveaxis(image, -1, 0)
        return torch.tensor(np.array([image]), dtype = torch.float)
    
    def preproc_output(self, output):
        predict = (output[0][0]*255).astype(np.uint8)
        predict = cv2.resize(
            predict,
            (980, 615),
            interpolation=cv2.INTER_AREA
        )
        img = np.array(Image.fromarray(predict))
        rgb_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        _, _, red = cv2.split(rgb_image)
        zeros = np.zeros(red.shape, np.uint8)
        redBGR = cv2.merge([zeros, zeros, red])
        return redBGR
