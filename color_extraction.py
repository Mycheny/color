# -*- coding: utf-8 -*- 
# @Time 2020/3/7 17:11
# @Author wcy

import collections
import colorsys
import copy
import csv
import math
import os
import pickle
import sys
import copy
import pandas as pd
import cv2
import imageio
import numpy as np
import re
import matplotlib.pyplot as plt
import xlrd
from PIL import Image, ImageDraw, ImageFont
from colormath.color_diff import delta_e_cie2000, delta_e_cmc
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from rgb2lab import RGB2Lab
from colormath.color_objects import LabColor

np.random.seed(1)


class ColorType(object):
    PURE = 1  # 纯色
    JOINT = 2  # 拼接
    TEXTURE = 3  # 花纹


class ColorIdentify(object):
    def __init__(self):
        self.min_height = 256  # 处理时压缩图片高度至此
        self.exist_colors = {}
        self.unusual_colors = []
        self.costume_color_dict = {}  # 服饰颜色字典， 颜色名：RGB
        self.basis_color_dict = {}  # 基础颜色字典
        self.init_costume_color_dict()  # 初始化服饰颜色字典
        self.init_basis_color_dict()  # 初始化基础颜色字典

    def init_costume_color_dict(self, file_path=r'E:\PycharmProjects\服饰颜色识别\颜色\color.xlsx'):
        # 文件路径的中文转码，如果路径非中文可以跳过
        file_path = file_path.encode('utf-8').decode('utf-8')
        # 获取数据
        data = xlrd.open_workbook(file_path)
        table = data.sheet_by_name('Sheet1')
        nrows = table.nrows
        # 获取一行的数值，例如第5行
        for i in range(nrows):
            rowvalue = table.row_values(i)
            color_name1 = rowvalue[1]
            color_value1 = rowvalue[2].strip()
            if re.match('[0-9]* [0-9]* [0-9]*', color_value1):
                self.costume_color_dict[color_name1] = [int(i) for i in color_value1.split(" ")]

    def init_basis_color_dict(self):
        """
        初始化基础颜色范围 （黑色，白色，红色，橙色...）
        :return:
        """
        self.basis_color_dict = collections.defaultdict(list)

        # 黑色
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 46])
        color_list = []
        color_list.append(lower_black)
        color_list.append(upper_black)
        self.basis_color_dict['black'] = color_list

        # #灰色
        lower_gray = np.array([0, 0, 46])
        upper_gray = np.array([180, 43, 220])
        color_list = []
        color_list.append(lower_gray)
        color_list.append(upper_gray)
        self.basis_color_dict['gray'] = color_list

        # 白色
        lower_white = np.array([0, 0, 221])
        upper_white = np.array([180, 30, 255])
        color_list = []
        color_list.append(lower_white)
        color_list.append(upper_white)
        self.basis_color_dict['white'] = color_list

        # 红色
        lower_red = np.array([156, 43, 46])
        upper_red = np.array([180, 255, 255])
        color_list1 = []
        color_list1.append(lower_red)
        color_list1.append(upper_red)

        # 红色2
        lower_red = np.array([0, 43, 46])
        upper_red = np.array([10, 255, 255])
        color_list2 = []
        color_list2.append(lower_red)
        color_list2.append(upper_red)
        self.basis_color_dict['red'] = (color_list1, color_list2)

        # 橙色
        lower_orange = np.array([11, 43, 46])
        upper_orange = np.array([25, 255, 255])
        color_list = []
        color_list.append(lower_orange)
        color_list.append(upper_orange)
        self.basis_color_dict['orange'] = color_list

        # 黄色
        lower_yellow = np.array([26, 43, 46])
        upper_yellow = np.array([34, 255, 255])
        color_list = []
        color_list.append(lower_yellow)
        color_list.append(upper_yellow)
        self.basis_color_dict['yellow'] = color_list

        # 绿色
        lower_green = np.array([35, 43, 46])
        upper_green = np.array([77, 255, 255])
        color_list = []
        color_list.append(lower_green)
        color_list.append(upper_green)
        self.basis_color_dict['green'] = color_list

        # 青色
        lower_cyan = np.array([78, 43, 46])
        upper_cyan = np.array([99, 255, 255])
        color_list = []
        color_list.append(lower_cyan)
        color_list.append(upper_cyan)
        self.basis_color_dict['cyan'] = color_list

        # 蓝色
        lower_blue = np.array([100, 43, 46])
        upper_blue = np.array([124, 255, 255])
        color_list = []
        color_list.append(lower_blue)
        color_list.append(upper_blue)
        self.basis_color_dict['blue'] = color_list

        # 紫色
        lower_purple = np.array([125, 43, 46])
        upper_purple = np.array([155, 255, 255])
        color_list = []
        color_list.append(lower_purple)
        color_list.append(upper_purple)
        self.basis_color_dict['purple'] = color_list


    def get_dominant_image(self, frame, mask, n_colors):
        def recreate_image(codebook, labels, mask, w, h):
            """从代码簿和标签中重新创建（压缩）图像"""
            mask_flatten = mask.flatten()
            labels_ = np.zeros_like(mask_flatten, dtype=np.int) - 1
            labels_[mask_flatten.astype(np.bool)] = labels
            labels_ = np.reshape(labels_, mask.shape)
            d = codebook.shape[1]
            image = np.zeros((w, h, d))
            image[labels_ == -1] = [0, 255, 0]
            for i, rgb in enumerate(codebook):
                image[labels_ == i] = rgb
            return image.astype(np.uint8)

        w, h, c = tuple(frame.shape)
        image_array = frame[mask.astype(np.bool)]
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)
        labels = kmeans.predict(image_array)
        image = recreate_image(kmeans.cluster_centers_, labels, mask, w, h)
        return image

    def get_color_type(self, dominant_image, dominant_color_names):
        """
        获取颜色类型(纯色，拼接， 杂色）
        :param frame:
        :return: ColorType
        """
        color_type = ColorType.PURE
        dominant_color_names_bak = copy.deepcopy(dominant_color_names)
        count = 0
        for i, dominant_color_name1 in enumerate(dominant_color_names):
            for j, dominant_color_name2 in enumerate(dominant_color_names[i + 1:]):
                distance = self.color_distance_cie2000(dominant_color_name1[1], dominant_color_name2[1], Kl=1)
                if distance < 5:
                    similar_color_name1, similar_color_rgb1, similar_color_score1, ratio1 = dominant_color_name1
                    similar_color_name2, similar_color_rgb2, similar_color_score2, ratio2 = dominant_color_name2
                    # r = dominant_color_names_bak.pop(i if ratio1 < ratio2 else j + i + 1 - count)
                    count+=1
        texture = cv2.Canny(dominant_image, 100, 200)
        texture_len = np.sum(texture == 255)
        # print(f"texture_len {texture_len}", f"dominant_color {len(dominant_color_names_bak)}  ")
        dominant_color_rgb = np.array([i[3] for i in dominant_color_names_bak])
        if texture_len > 2000 and len(dominant_color_names_bak) > 3:
            color_type = ColorType.TEXTURE  # 杂色
        elif np.sum(dominant_color_rgb > 0.1) >= 2 and len(dominant_color_names_bak) > 1:
            color_type = ColorType.JOINT  # 拼接色
        return color_type

    def get_basis_color_num(self, frame, frame_mask):
        """
        获取基础颜色种数
        :param frame:
        :param frame_mask: 掩码 忽略获取颜色的部分
        :return: 基础颜色种数颜色数
        """
        valid_mask = np.sum(frame_mask)  # 有效mask和
        kerne2 = np.ones((5, 5), np.float32) / 25
        frame = cv2.filter2D(frame, -1, kerne2)  # 平滑圖片
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        count = 0
        for d in self.basis_color_dict:
            if isinstance(self.basis_color_dict[d], list):
                mask = cv2.inRange(hsv, self.basis_color_dict[d][0], self.basis_color_dict[d][1])
            else:
                # if isinstance(color_dict[d], tuple)
                mask1 = cv2.inRange(hsv, self.basis_color_dict[d][0][0], self.basis_color_dict[d][0][1])
                mask2 = cv2.inRange(hsv, self.basis_color_dict[d][1][0], self.basis_color_dict[d][1][1])
                mask = mask1 + mask2
            mask3 = mask * frame_mask
            pro = np.sum(mask3 == 255) / valid_mask
            if pro > 1 / len(self.basis_color_dict.keys()):
                # print(d)
                count += 1
        return count

    def color_distance_cie2000(self, rgb_1, rgb_2, Kl=2, Kc=1, Kh=1):
        """计算色差"""
        lab1 = RGB2Lab(rgb_1[::-1])
        lab2 = RGB2Lab(rgb_2[::-1])
        color1 = LabColor(lab_l=lab1[0], lab_a=lab1[1], lab_b=lab1[2])
        color2 = LabColor(lab_l=lab2[0], lab_a=lab2[1], lab_b=lab2[2])
        delta_e = delta_e_cie2000(color1, color2, Kl=2, Kc=1, Kh=1)
        return delta_e

    def get_color_names(self, dominant_colors):
        """
        获取与对应rgb值最相似的颜色名, 相似颜色rgb，相似颜色得分(越小越好), 该色占服饰面积百分比
        :param dominant_colors: 服饰主体颜色  dict {rgb:ratio}
        :return: [[相似颜色名, 相似颜色rgb，相似颜色得分(越小越好), 该色占服饰面积百分比]]
        """
        def takeSecond(elem):
            return elem[3]
        dominant_colors_names = []
        for rgb, ratio in dominant_colors.items():
            result = {color_name: self.color_distance_cie2000(rgb, [r, g, b]) for color_name, (r, g, b) in
                      self.costume_color_dict.items()}
            similar_color_name = min(result, key=result.get)
            costume_color_rgb = self.costume_color_dict[similar_color_name]
            similar_color_score = result[similar_color_name]
            dominant_colors_names.append([similar_color_name, rgb, similar_color_score, ratio])
        dominant_colors_names.sort(key=takeSecond, reverse=True)
        return dominant_colors_names

    def get_costume_mask(self, image_resize):
        """
        获取服饰位置mask
        :param image_resize:
        :return:
        """
        def takeSecond(elem):
            return elem[0]
        mask = np.zeros(image_resize.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        h, w, c = image_resize.shape
        kernel = np.ones((5, 5), np.float32) / 25
        kerne2 = np.ones((10, 10), np.float32) / 25
        # masks, ratios = [], []
        datas = []
        for boder in [0.05, 0.07, 0.09, 0.11][::-1]:
            border_x = boder
            border_y = boder * 1.0
            rect = (int(w * border_x), int(h * border_y), int(w * (1 - border_x * 2)), int(h * (1 - border_y * 2)))
            cv2.grabCut(image_resize, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
            src_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # 获取可能是背景的区域
            open_mask = cv2.morphologyEx(src_mask, cv2.MORPH_OPEN, kernel)  # 开运算后的mask
            ratio = open_mask.sum() / open_mask.size  # 前景占比
            datas.append([ratio, open_mask])
            # cv2.imshow("image_resize", image_resize)
            # cv2.imshow("", np.hstack((src_mask*255, open_mask*255)))
            # cv2.waitKey(0)
            # ratios.append(ratio)
            # masks.append(open_mask)
        # ratios, masks = [list(t) for t in zip(*sorted(zip(ratios, masks), reverse=True))]
        datas.sort(key=takeSecond, reverse=True)
        res_mask = datas[0][1]
        for i, (ratio, mask) in enumerate(datas):
            if ratio < 0.55:
                res_mask = res_mask
                break
        return res_mask

    def get_dominant_colors(self, frame, mask):
        """
        获取主要颜色
        :param frame:
        :param mask: 用于忽略背景
        :return:
        """
        frame_dominant = frame[mask.astype(np.bool)]
        color_ratio = {tuple(rgb[::-1]): np.sum((frame_dominant == rgb)[:, 0]) / frame_dominant.shape[0] for rgb in
                       np.unique(frame_dominant, axis=0)}
        return color_ratio

    def predict(self, frame, label_color_name=None):
        """
        三通道 RGB图片
        :param frame:
        :return:
        """
        frame = copy.deepcopy(frame)
        width = int(frame.shape[1] * self.min_height / frame.shape[0])
        image_resize = cv2.resize(frame, (width, self.min_height))
        mask = self.get_costume_mask(image_resize)
        basis_color_num = self.get_basis_color_num(image_resize, mask)
        dominant_image = self.get_dominant_image(image_resize, mask, basis_color_num)
        dominant_color_rgb = self.get_dominant_colors(dominant_image, mask)
        dominant_color_names = self.get_color_names(dominant_color_rgb)
        color_type = self.get_color_type(dominant_image, dominant_color_names)
        if color_type==1:
            dominant_color_names = dominant_color_names[:1]
        return {"color_type":color_type, "color_names":dominant_color_names, "dominant_color_rgb":dominant_color_rgb}


def drow(frame, color_name, color_rgb, dominant_color_rgb, color_type):
    w, h = 80, 80
    image2 = cv2ImgAddText(frame, f"", 10, 80, textSize=14)
    i=-1
    color = (int(dominant_color_rgb[2]), int(dominant_color_rgb[1]), int(dominant_color_rgb[0]))
    cv2.rectangle(image2, ((i + 1) * w + (i + 1) * 10, 0), ((i + 2) * w + (i + 1) * 10, h), color, -1)
    i = 0
    cv2.rectangle(image2, ((i + 1) * w + (i + 1) * 10, 0), ((i + 2) * w + (i + 1) * 10, h), color_rgb[::-1], -1)
    if color_type>1:
        path = os.path.join("D:\\color", str(color_type))
    else:
        path = os.path.join("D:\\color", color_name)
    if not os.path.isdir(path):
        os.mkdir(path)
    file_path = f"{path}/{color_name}_{np.random.randint(1000, 9999)}.jpg".encode('utf-8').decode('utf-8')
    cv2.imencode('.jpg', image2)[1].tofile(file_path)

def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=20):
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


if __name__ == '__main__':
    ci = ColorIdentify()
    root = "no_label"
    # color_list = os.listdir(root)
    color_list = ["没有标注的颜色4500张图片"]
    count = 0
    ok = 0
    for a, color in enumerate(color_list[:500]):
        # color = "拼色"
        if "拼色" in color or "花色" in color:
            continue
        file_names = os.listdir(os.path.join(root, color))
        for b, file_name in enumerate(file_names):
            try:
                # file_name = "-103543582.jpg"
                # print(f"\r {a}/{len(color_list)} {b}/{len(file_names)}", end="")
                file_path = os.path.join(root, color, file_name)
                # file_path  = "服饰\黄色\黄色 (2).jpg"
                # color = "黄色"

                file_path = file_path.encode('utf-8').decode('utf-8')
                frame = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
                if frame is None:
                    tmp = imageio.mimread(file_path)
                    if tmp is not None:
                        imt = np.array(tmp)
                        imt = imt[0]
                        frame = imt[:, :, 0:3]
                if frame.shape[2] == 4 and frame.dtype == np.uint16:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR).astype(np.uint8)
                # color_name = re.sub("[\(\)（） 0-9]*", "", color)
                color_name = re.sub("[\(\)（） 0-9a-zA-Zjpg.]*", "", file_name)
                if frame.shape[2] == 4:
                    frame = frame[..., :3]
                # print(file_path)
                info = ci.predict(frame, color_name)
                color_type = info["color_type"]
                color_name = info["color_names"][0][0]
                color_rgb = ci.costume_color_dict[color_name]
                dominant_color_rgb = [k for k, v in sorted(info["dominant_color_rgb"].items(), reverse=True)][0]
                drow(frame, color_name, color_rgb, dominant_color_rgb, color_type)
            except Exception as e:
                print(e)
                continue
