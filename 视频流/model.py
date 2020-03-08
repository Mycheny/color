# -*- coding: utf-8 -*- 
# @Time 2020/3/3 20:54
# @Author wcy
import colorsys
import operator
import time
import platform
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976, delta_e_cie1994, delta_e_cie2000
import threading
from rgb2lab import RGB2Lab


class Model(object):
    frame = None
    colors = []
    last_rgb = [0, 0, 0]
    sleep = 2000
    console_h = 40
    cell_w = 18
    cell_h = 12
    # cell_w = 36
    # cell_h = 6
    n = 36
    side = 10

    def cv2ImgAddText(self, img, text, left, top, textColor=(255, 0, 0), textSize=20):
        if platform.system() == 'Windows':
            if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # 创建一个可以在给定图像上绘图的对象
            draw = ImageDraw.Draw(img)
            # 字体的格式
            fontStyle = ImageFont.truetype(
                "font/simsun.ttc", textSize, encoding="utf-8")
            # 绘制文本
            draw.text((left, top), text, textColor, font=fontStyle)
            # 转换回OpenCV格式
            return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        elif platform.system() == 'Linux':
            cv2.putText(img, text, (left, top + self.side + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, textColor, 1)
            return img
        else:
            return img

    def color_distance(self, rgb_1, rgb_2):
        """

        :param rgb_1: rgb
        :param rgb_2: rgb
        :return:
        """
        lab1 = RGB2Lab(rgb_1[::-1])
        lab2 = RGB2Lab(rgb_2[::-1])
        color1 = LabColor(lab_l=lab1[0], lab_a=lab1[1], lab_b=lab1[2])
        color2 = LabColor(lab_l=lab2[0], lab_a=lab2[1], lab_b=lab2[2])
        delta_e = delta_e_cie2000(color1, color2)
        return delta_e

    def run1(self):
        while True:
            color1 = [np.random.randint(0, 255) for ii in range(3)]
            distances = [self.color_distance(color1, color2) for color2 in self.colors]
            min_distance_index = np.argmin(distances)
            min_distance_value = np.min(distances)
            x1, y1 = min_distance_index // self.cell_h, min_distance_index % self.cell_h
            x2, y2 = x1 + 1, y1 + 1
            x1, y1 = x1 * self.n + self.side, y1 * self.n + self.side
            x2, y2 = x2 * self.n - self.side, y2 * self.n - self.side
            # x1, y1 = x1 * n, y1 * n
            # x2, y2 = x2 * n - int(n/2), y2 * n
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color1[::-1], -1)
            brightness = (0.3 * color1[0] + 0.6 * color1[1] + 0.1 * color1[2]) / 255  # 亮度
            textColor = (255, 255, 255) if brightness < 0.5 else (0, 0, 0)
            self.frame = self.cv2ImgAddText(self.frame, f"{int(min_distance_value)}", x1, y1, textColor=textColor,
                                            textSize=14)
            time.sleep(0.1)

    def run2(self):
        last_min_distance_index = 0
        last_x1, last_y1 = 0, 0
        last_x2, last_y2 = 0, 0
        while True:
            color1 = [np.random.randint(0, 255) for ii in range(3)]
            distances = [self.color_distance(color1, color2) for color2 in self.colors]
            for index, distance in enumerate(distances):
                x1, y1 = index // self.cell_h, index % self.cell_h
                x2, y2 = x1 + 1, y1 + 1
                x1, y1 = x1 * self.n + self.side, y1 * self.n + self.side
                x2, y2 = x2 * self.n - self.side, y2 * self.n - self.side
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), color1[::-1], -1)
                brightness = (0.3 * color1[0] + 0.6 * color1[1] + 0.1 * color1[2]) / 255  # 亮度
                textColor = (255, 255, 255) if brightness < 0.5 else (0, 0, 0)
                self.frame = self.cv2ImgAddText(self.frame, f"{int(distance)}", x1, y1, textColor=textColor,
                                                textSize=14)
                time.sleep(0.001)

            cv2.rectangle(self.frame, (last_x1, last_y1), (last_x2, last_y2),
                          self.colors[last_min_distance_index][::-1], 1)
            min_distance_index = np.argmin(distances)
            thickness = 2
            x1, y1 = min_distance_index // self.cell_h, min_distance_index % self.cell_h
            x2, y2 = x1 + 1, y1 + 1
            x1, y1 = x1 * self.n + thickness, y1 * self.n + thickness
            x2, y2 = x2 * self.n - thickness, y2 * self.n - thickness
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            last_x1, last_y1 = x1, y1
            last_x2, last_y2 = x2, y2
            last_min_distance_index = min_distance_index
            time.sleep(5)

    def get_fream(self):
        return self.frame

    def __init__(self, flag=0):
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    self.colors.append((i * 51, j * 51, k * 51))

        h, w = self.cell_h * self.n, self.cell_w * self.n
        self.frame = np.zeros((h + 40, w, 3), dtype=np.uint8)
        for index, color in enumerate(self.colors):
            x1, y1 = index // self.cell_h, index % self.cell_h
            x2, y2 = x1 + 1, y1 + 1
            cv2.rectangle(self.frame, (x1 * self.n, y1 * self.n), (x2 * self.n, y2 * self.n), color[::-1], -1)
        if flag == 0:
            t = threading.Thread(target=self.run1)
        else:
            t = threading.Thread(target=self.run2)
        t.setDaemon(True)
        t.start()
