# -*- coding: utf-8 -*- 
# @Time 2020/3/4 0:41
# @Author wcy
import base64
from time import time
import numpy as np
import cv2
from PIL import Image

from model import Model


class Camera(object):
    def __init__(self, flag=0):
        self.frames = [open(f + '.jpg', 'rb').read() for f in ['1', '2', '3']]
        self.model = Model(flag=flag)


    def get_frame(self):
        frame = self.model.get_fream()
        if not frame is None:
            ret2, buf2 = cv2.imencode(".jpg", frame)
            img_bin2 = Image.fromarray(np.uint8(buf2)).tobytes()
            # model_base64_2 = base64.b64encode(img_bin2)
            return img_bin2
        return self.frames[int(time()) % 3]