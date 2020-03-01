import collections
import colorsys
import math
import os

import cv2
import imageio
import numpy as np
import re
import matplotlib.pyplot as plt
import xlrd
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from rgb2lab import RGB2Lab


class ColorType(object):
    PURE = 1  # 纯色
    JOINT = 2  # 拼接
    TEXTURE = 3  # 花纹


class ColorIdentify(object):
    def __init__(self):
        self.costume_color_dict = {}  # 服饰颜色字典， 颜色名：RGB
        self.basis_color_dict = {}  # 基础颜色字典
        self.init_costume_color_dict()  # 初始化服饰颜色字典
        self.init_basis_color_dict()  # 初始化基础颜色字典

    def init_costume_color_dict(self, file_path=r'E:\PycharmProjects\服饰颜色识别\颜色\猪圈关键字（55色）(4)(1).xlsx'):
        # 文件路径的中文转码，如果路径非中文可以跳过
        file_path = file_path.encode('utf-8').decode('utf-8')
        # 获取数据
        data = xlrd.open_workbook(file_path)
        table = data.sheet_by_name('猪圈颜色')
        nrows = table.nrows
        # 获取一行的数值，例如第5行
        for i in range(nrows):
            rowvalue = table.row_values(i)
            color_name1 = rowvalue[2]
            color_value1 = rowvalue[3]
            color_name2 = rowvalue[6]
            color_value2 = rowvalue[7]
            if re.match('[0-9]* [0-9]* [0-9]*', color_value1):
                self.costume_color_dict[color_name1] = [int(i) for i in color_value1.split(" ")]
            if re.match('[0-9]* [0-9]* [0-9]*', color_value2):
                self.costume_color_dict[color_name2] = [int(i) for i in color_value2.split(" ")]

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

    def get_color_type(self, frame):
        """
        获取颜色类型
        :param frame:
        :return: ColorType
        """
        texture = cv2.Canny(frame, 100, 200)  # 边缘检测
        texture_len = np.sum(texture == 255)  # 边缘长度
        print(texture_len)
        if texture_len < 2500:
            return ColorType.PURE  # 返回纯色类型
        elif texture_len < 4000:
            return ColorType.JOINT  # 返回拼接类型
        return ColorType.TEXTURE  # 返回花纹类型

    def get_n_color(self, frame):
        """
        获取基础颜色种数
        :param frame:
        :return: 基础颜色种数颜色 （int）
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_dict = self.basis_color_dict
        count = 0
        for d in color_dict:
            if isinstance(color_dict[d], list):
                mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
            elif isinstance(color_dict[d], tuple):
                mask1 = cv2.inRange(hsv, color_dict[d][0][0], color_dict[d][0][1])
                mask2 = cv2.inRange(hsv, color_dict[d][1][0], color_dict[d][1][1])
                mask = mask1 + mask2
            pro = np.sum(mask == 255) / mask.size
            # cv2.imshow(d, mask)
            # print(d, pro)
            if pro > 0.1 / len(color_dict.keys()):
                count += 1
        # cv2.waitKey(0)
        return count

    def get_dominant_color2(self, frame, color_type, color_num):
        def flatten(a):
            for each in a:
                if not isinstance(each, list):
                    yield each
                else:
                    yield from flatten(each)

        def takeSecond(elem):
            return elem[1]

        image = Image.fromarray(frame.astype('uint8')).convert('RGB')
        height = 128
        width = int(image.height * height / image.width)
        train_image = image.resize((width, height), Image.ANTIALIAS)
        dataset = [[(r, g, b)] * count for count, (r, g, b) in
                   train_image.getcolors(train_image.size[0] * train_image.size[1]) if
                   (min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235) - 16.0) / (235 - 16) < 0.9]
        dataset = list(flatten(dataset))
        # y_pred1 = DBSCAN().fit_predict(dataset)
        estimator = KMeans(n_clusters=color_num, random_state=9)
        try:
            estimator.fit(dataset)
        except Exception as e:
            print(e)
        center = estimator.cluster_centers_.tolist()
        score = [np.sum(estimator.labels_ == i) for i in range(color_num)]
        dominant_colors = [[c, s] for c, s in zip(center, score)]
        dominant_colors.sort(key=takeSecond, reverse=True)
        return dominant_colors

    def get_dominant_color1(self, frame, color_type, color_num):
        """
        获取颜色主成分
        :param frame:
        :return:
        """

        def takeSecond(elem):
            return elem[1]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame.astype('uint8')).convert('RGB')
        dominant_colors = []
        for count, (r, g, b) in image.getcolors(image.size[0] * image.size[1]):
            # 转为HSV标准
            hue, saturation, value = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
            y = (y - 16.0) / (235 - 16)
            # 忽略高亮色
            if y > 0.9:
                continue
            score = (saturation + 0.1) * count
            dominant_colors.append([[r, g, b], score])
        dominant_colors.sort(key=takeSecond, reverse=True)
        if len(dominant_colors)==0:
            dominant_colors = [[[frame[..., 0].mean(), frame[..., 1].mean(), frame[..., 2].mean()], 1]]
        return dominant_colors

    def get_color_names(self, dominant_colors):
        def color_distance_lab(rgb_1, rgb_2):
            """
            LAB颜色空间 颜色距离
            :param rgb_1:
            :param rgb_2:
            :return:
            """
            R_1, G_1, B_1 = rgb_1
            R_2, G_2, B_2 = rgb_2
            rmean = (R_1 + R_2) / 2
            R = R_1 - R_2
            G = G_1 - G_2
            B = B_1 - B_2
            return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))

        def colour_distance_rgb(rgb_1, rgb_2):
            v1 = (np.array(rgb_1) - 128) / 255
            v2 = (np.array(rgb_2) - 128) / 255
            num = float(v1.dot(v2.T))
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            cos = num / (denom + 0.000001)  # 余弦值
            sim = 0.5 + 0.5 * cos  # 根据皮尔逊相关系数归一化
            return sim

        def colour_distance_rgb2(rgb_1, rgb_2):
            v1 = np.array(rgb_1)
            v2 = np.array(rgb_2)
            a = (v1 - v2)/256
            sim = np.sqrt(np.mean(np.square(a)))
            return 1-sim

        def colour_distance_rgb3(rgb_1, rgb_2):
            v1 = np.array(rgb_1) / 255
            v2 = np.array(rgb_2) / 255
            num = float(v1.dot(v2.T))
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            cos = num / (denom + 0.000001)  # 余弦值
            sim = cos  # 根据皮尔逊相关系数归一化
            return sim

        def color_distance_lab2(rgb_1, rgb_2):
            from colormath.color_objects import LabColor
            from colormath.color_diff import delta_e_cie1976
            lab1 = RGB2Lab(rgb_1[::-1])
            lab2 = RGB2Lab(rgb_2[::-1])
            # Reference color.
            color1 = LabColor(lab_l=lab1[0], lab_a=lab1[1], lab_b=lab1[2])
            # Color to be compared to the reference.
            color2 = LabColor(lab_l=lab2[0], lab_a=lab2[1], lab_b=lab2[2])
            # This is your delta E value as a float.
            delta_e = delta_e_cie1976(color1, color2)
            return delta_e

        dominant_colors_names = []
        for rgb, score in dominant_colors:
            similar_color = ""
            temp = -1
            for color_name, (r, g, b) in self.costume_color_dict.items():
                if color_name == "土色" or color_name == "暗灰":
                    pass
                # sim = colour_distance_rgb2(rgb, [r, g, b])
                # sim1 = colour_distance_rgb(rgb, [r, g, b])
                # sim2 = color_distance_lab2(rgb, [r, g, b])
                # sim3 = color_distance_lab(rgb, [r, g, b])
                sim = colour_distance_rgb3(rgb, [r, g, b])
                if sim > temp:
                    temp = sim
                    similar_color = color_name
            dominant_colors_names.append(similar_color)
        return dominant_colors_names

    def predict(self, frame, label_color_name=None):
        """
        三通道 RGB图片
        :param frame:
        :return:
        """
        height = 256
        width = int(frame.shape[1] * height / frame.shape[0])
        img = cv2.resize(frame, (width, height))
        temp_color_type = self.get_color_type(img)
        color_num = self.get_n_color(img)
        img2 = self.get_dominant_image(img, color_num).astype(np.uint8)
        # cv2.imshow("img", img)
        # cv2.imshow("img2", img2)
        # cv2.waitKey(0)

        dominant_colors1 = self.get_dominant_color1(img2, temp_color_type, color_num)  # bgr
        # dominant_colors2 = self.get_dominant_color2(img2, temp_color_type, color_num)
        color_names = self.get_color_names(dominant_colors1[:3])
        if not label_color_name is None:
            self.drow(frame, label_color_name, color_names)

    def drow(self, frame, label_name, pro_names):
        w, h = 70, 70
        try:
            cv2.rectangle(frame, (0, 0), (w, h), self.costume_color_dict[label_name][::-1], -1)
        except Exception as e:
            print(e)
        image2 = self.cv2ImgAddText(frame, f"标注为{label_name}", 10, 80, textSize=14)

        for i in range(len(pro_names)):
            cv2.rectangle(image2, ((i + 1) * w + (i + 1) * 10, 0), ((i + 2) * w + (i + 1) * 10, h),
                          self.costume_color_dict[pro_names[i]][::-1], -1)
        image2 = self.cv2ImgAddText(image2, f"检测为{' '.join(pro_names)}", 90, 80, textSize=14)

        file_path = f"image3/{label_name}_{'_'.join(pro_names)}_{np.random.random()}.jpg".encode('utf-8').decode(
            'utf-8')
        # cv2.imwrite(file_path, image2)
        cv2.imencode('.jpg', image2)[1].tofile(file_path)

    def cv2ImgAddText(self, img, text, left, top, textColor=(255, 0, 0), textSize=20):
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

    def get_dominant_image(self, img, n_colors):
        def recreate_image(codebook, labels, w, h):
            """从代码簿和标签中重新创建（压缩）图像"""
            d = codebook.shape[1]
            image = np.zeros((w, h, d))
            label_idx = 0
            for i in range(w):
                for j in range(h):
                    image[i][j] = codebook[labels[label_idx]]
                    label_idx += 1
            return image

        w, h, c = tuple(img.shape)
        image_array = np.reshape(img, (w * h, c))
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)
        labels = kmeans.predict(image_array)
        image = recreate_image(kmeans.cluster_centers_, labels, w, h)
        return image

if __name__ == '__main__':
    ci = ColorIdentify()
    root = "服饰"
    color_list = os.listdir(root)
    for color in color_list:
        if "拼色" in color or "花色" in color:
            continue
        file_names = os.listdir(os.path.join(root, color))
        for file_name in file_names:
            file_path = os.path.join(root, color, file_name)
            # file_path  = "服饰\深灰\深灰 (1).jpg"
            # color = "深灰"
            file_path = file_path.encode('utf-8').decode('utf-8')
            china = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
            if china is None:
                tmp = imageio.mimread(file_path)
                if tmp is not None:
                    imt = np.array(tmp)
                    imt = imt[0]
                    china = imt[:, :, 0:3]
            color_name = re.sub("[\(\)（） 0-9]*", "", color)
            if china.shape[2]==4:
                china = china[..., :3]
            # print(color_name, file_path)
            ci.predict(china, color_name)
            # print()
