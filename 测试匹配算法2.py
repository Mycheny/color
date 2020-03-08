# -*- coding: utf-8 -*- 
# @Time 2020/3/3 20:54
# @Author wcy
import colorsys
import copy
import math
import operator
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976, delta_e_cie1994, delta_e_cie2000

from rgb2lab import RGB2Lab

frame = None
group_colors = {}
last_rgb = [0, 0, 0]
sleep = 2000


# RGB格式颜色转换为16进制颜色格式
def RGB_to_Hex(rgb):
    RGB = rgb.split(',')  # 将RGB格式划分开来
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    print(color)
    return color


# RGB格式颜色转换为16进制颜色格式
def RGB_list_to_Hex(RGB):
    # RGB = rgb.split(',')  # 将RGB格式划分开来
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    # print(color)
    return color


# 16进制颜色格式颜色转换为RGB格式
def Hex_to_RGB(hex):
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    rgb = str(r) + ',' + str(g) + ',' + str(b)
    # print(rgb)
    return rgb, [r, g, b]


# 生成渐变色
def gradient_color(color_list, color_sum=200):
    color_center_count = len(color_list)
    # if color_center_count == 2:
    #     color_center_count = 1
    color_sub_count = int(color_sum / (color_center_count - 1))
    color_index_start = 0
    color_map = []
    for color_index_end in range(1, color_center_count):
        color_rgb_start = Hex_to_RGB(color_list[color_index_start])[1]
        color_rgb_end = Hex_to_RGB(color_list[color_index_end])[1]
        r_step = (color_rgb_end[0] - color_rgb_start[0]) / color_sub_count
        g_step = (color_rgb_end[1] - color_rgb_start[1]) / color_sub_count
        b_step = (color_rgb_end[2] - color_rgb_start[2]) / color_sub_count
        # 生成中间渐变色
        now_color = color_rgb_start
        color_map.append(RGB_list_to_Hex(now_color))
        for color_index in range(1, color_sub_count):
            now_color = [now_color[0] + r_step, now_color[1] + g_step, now_color[2] + b_step]
            color_map.append(RGB_list_to_Hex(now_color))
        color_index_start = color_index_end
    return color_map


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


def color_distance(rgb_1, rgb_2):
    """

    :param rgb_1: rgb
    :param rgb_2: rgb
    :return:
    """
    lab1 = RGB2Lab(rgb_1[::-1])
    lab2 = RGB2Lab(rgb_2[::-1])
    color1 = LabColor(lab_l=lab1[0], lab_a=lab1[1], lab_b=lab1[2])
    color2 = LabColor(lab_l=lab2[0], lab_a=lab2[1], lab_b=lab2[2])
    delta_e = delta_e_cie2000(color1, color2, Kl=2, Kc=2, Kh=2)
    return delta_e


def color_distance2(rgb_1, rgb_2):
    R_1, G_1, B_1 = rgb_1
    R_2, G_2, B_2 = rgb_2
    rmean = (R_1 + R_2) / 2
    R = R_1 - R_2
    G = G_1 - G_2
    B = B_1 - B_2
    return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))


def drow(min_distance_index, min_distance_value, means, frame2, i):
    if min_distance_index in means.keys():
        means[min_distance_index] = i / (i + 1) * means[min_distance_index] + 1 / (i + 1) * min_distance_value
    else:
        means[min_distance_index] = min_distance_value
    for k, v in means.items():
        x1, y1 = k // cell_h, k % cell_h
        x2, y2 = x1 + 1, y1 + 1
        cv2.rectangle(frame2, (x1 * n + 2, y1 * n + 2), (x2 * n - 2, y2 * n - 2), colors[k][::-1], -1)

        x1, y1 = k // cell_h, k % cell_h
        x1, y1 = x1 * n + side, y1 * n + side
        brightness = (0.3 * colors[k][0] + 0.6 * colors[k][1] + 0.1 * colors[k][2]) / 255  # 亮度
        textColor = (255, 255, 255) if brightness < 0.5 else (0, 0, 0)
        frame2 = cv2ImgAddText(frame2, f"{int(v + 0.5)}", x1, y1, textColor=textColor, textSize=14)
    cv2.imshow("frame2", frame2)


def HSVDistance(rgb_1, rgb_2):
    hsv_1 = colorsys.rgb_to_hsv(rgb_1[0] / 255.0, rgb_1[1] / 255.0, rgb_1[2] / 255.0)
    hsv_2 = colorsys.rgb_to_hsv(rgb_2[0] / 255.0, rgb_2[1] / 255.0, rgb_2[2] / 255.0)
    H_1, S_1, V_1 = hsv_1
    H_2, S_2, V_2 = hsv_2
    R = 100
    angle = 30
    h = R * math.cos(angle / 180 * math.pi)
    r = R * math.sin(angle / 180 * math.pi)
    x1 = r * V_1 * S_1 * math.cos(H_1 / 180 * math.pi)
    y1 = r * V_1 * S_1 * math.sin(H_1 / 180 * math.pi)
    z1 = h * (1 - V_1)
    x2 = r * V_2 * S_1 * math.cos(H_2 / 180 * math.pi)
    y2 = r * V_2 * S_1 * math.sin(H_2 / 180 * math.pi)
    z2 = h * (1 - V_2)
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def run1():
    global frame, group_colors
    frame2 = np.copy(frame)
    means = {}
    mean = 0
    for i in range(10000):
        color1 = [np.random.randint(0, 255) for ii in range(3)]
        distances = [color_distance(color1, color2) for color2 in colors]
        # distances = [HSVDistance(color1, color2) for color2 in colors]
        min_distance_index = np.argmin(distances)
        min_distance_value = np.min(distances)
        x1, y1 = min_distance_index // cell_h, min_distance_index % cell_h
        x2, y2 = x1 + 1, y1 + 1
        x1, y1 = x1 * n + side, y1 * n + side
        x2, y2 = x2 * n - side, y2 * n - side
        # x1, y1 = x1 * n, y1 * n
        # x2, y2 = x2 * n - int(n/2), y2 * n
        cv2.rectangle(frame, (x1, y1), (x2, y2), color1[::-1], -1)
        brightness = (0.3 * color1[0] + 0.6 * color1[1] + 0.1 * color1[2]) / 255  # 亮度
        textColor = (255, 255, 255) if brightness < 0.5 else (0, 0, 0)
        frame = cv2ImgAddText(frame, f"{int(min_distance_value)}", x1, y1, textColor=textColor, textSize=14)
        mean = i/(i+1)*mean+1/(i+1)*min_distance_value
        # print(mean)

        if min_distance_index in group_colors.keys():
            if len(group_colors[min_distance_index])>17:
                group_colors[min_distance_index].pop(0)
            group_colors[min_distance_index].append([color1, min_distance_value])
        else:
            group_colors[min_distance_index] = [[color1, min_distance_value]]
        # drow(min_distance_index, min_distance_value, means, frame2, i)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
    cv2.waitKey(0)



def run2():
    global frame, sleep
    last_min_distance_index = 0
    last_x1, last_y1 = 0, 0
    last_x2, last_y2 = 0, 0
    for i in range(10000):
        color1 = [np.random.randint(0, 255) for ii in range(3)]
        distances = [color_distance(color1, color2) for color2 in colors]
        for index, distance in enumerate(distances):
            x1, y1 = index // cell_h, index % cell_h
            x2, y2 = x1 + 1, y1 + 1
            x1, y1 = x1 * n + side, y1 * n + side
            x2, y2 = x2 * n - side, y2 * n - side
            cv2.rectangle(frame, (x1, y1), (x2, y2), color1[::-1], -1)
            brightness = (0.3 * color1[0] + 0.6 * color1[1] + 0.1 * color1[2]) / 255  # 亮度
            textColor = (255, 255, 255) if brightness < 0.5 else (0, 0, 0)
            frame = cv2ImgAddText(frame, f"{int(distance)}", x1, y1, textColor=textColor, textSize=14)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
        cv2.rectangle(frame, (last_x1, last_y1), (last_x2, last_y2), colors[last_min_distance_index][::-1], 1)
        min_distance_index = np.argmin(distances)
        thickness = 2
        x1, y1 = min_distance_index // cell_h, min_distance_index % cell_h
        x2, y2 = x1 + 1, y1 + 1
        x1, y1 = x1 * n + thickness, y1 * n + thickness
        x2, y2 = x2 * n - thickness, y2 * n - thickness
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        last_x1, last_y1 = x1, y1
        last_x2, last_y2 = x2, y2
        last_min_distance_index = min_distance_index
        cv2.imshow(window_name, frame)
        cv2.waitKey(sleep)


def MouseEvent(event, x, y, flags, param):
    global frame, last_rgb, sleep, console_h, group_colors
    if event == cv2.EVENT_FLAG_LBUTTON:
        rgb = frame[y, x, :][::-1].tolist()
        if not operator.eq(last_rgb, rgb):
            last_hsv = [round(i, 2) for i in colorsys.rgb_to_hsv(last_rgb[0] / 255.0, last_rgb[1] / 255.0, last_rgb[2] / 255.0)]
            hsv = [round(j, 2) for j in colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)]

            last_Lab = [round(i, 1) for i in RGB2Lab(last_rgb[::-1])]
            Lab = [round(j, 1) for j in RGB2Lab(rgb[::-1])]

            cv2.rectangle(frame, (0, h), (w, h+console_h), (0, 0, 0), -1)
            frame = cv2ImgAddText(frame, f"██", 5, h + 3, textColor=tuple(last_rgb), textSize=16)
            frame = cv2ImgAddText(frame, f"██", 5, h+int(console_h/2)+3, textColor=tuple(rgb), textSize=16)

            frame = cv2ImgAddText(frame, f"RGB {last_rgb}, HSV {last_hsv}, Lab {last_Lab}", console_h, h+3, textColor=(255, 255, 255), textSize=16)
            frame = cv2ImgAddText(frame, f"RGB {rgb}, HSV {hsv}, Lab {Lab}", console_h, h+int(console_h/2)+3, textColor=(255, 255, 255), textSize=16)
            print(f"RGB {rgb}")
            last_rgb = rgb
    if event == cv2.EVENT_LBUTTONDBLCLK:
        start_x = 600
        index = x//n*cell_h+y//n
        print(index)
        if index in group_colors.keys():
            group_color = group_colors[index]
            for i, group in enumerate(group_color):
                color = group[0]
                pro = group[1]
                cv2.rectangle(frame, (start_x+i*console_h, h), (start_x+console_h+i*console_h, h+console_h), color[::-1], -1)
                frame = cv2ImgAddText(frame, f"{int(pro+0.5)}", start_x+i*console_h+5, h+5,
                                      textColor=(255, 255, 255), textSize=16)


    if flags == cv2.EVENT_FLAG_RBUTTON:
        if sleep == 0:
            sleep = 2000
            print(f"继续")
            cv2.destroyWindow(window_name)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, MouseEvent)  # 窗口与回调函数绑定
        else:
            print(f"暂停")
            sleep = 0


def g_color():
    global colors
    for i in range(6):
        for j in range(6):
            for k in range(6):
                colors.append((i * 51, j * 51, k * 51))


def g_color2():
    global colors
    # input_colors = ["#000000", "#40FAFF", "#00EBEB", "#00EB00", "#FFC800", "#FC9600", "#FA0000", "#C800FA", "#FF64FF", "#FFFFFF"]
    # input_colors = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#99004c", "#7e0023"]
    # input_colors = ["#000000", "#777777", "#FFFFFF", "#FF0000", "#FF7700", "#FFFF00", "#00FF00", "#00FFFF", "#0000FF", "#FF00FF"]
    input_colors = ["#000000", "#777777", "#FFFFFF",
                    "#FF0000", "#000000",
                    "#FF7700", "#FFFFFF",
                    "#FFFF00", "#000000",
                    "#00FF00", "#FFFFFF",
                    "#00FFFF", "#000000",
                    "#0000FF", "#FFFFFF",
                    "#FF00FF", "#000000",]
    colors0x = gradient_color(input_colors, color_sum=216+8)
    colors = [[int(f'0x{color[1:3]}',16), int(f'0x{color[3:5]}',16), int(f'0x{color[5:7]}',16)] for color in colors0x]
    print()


if __name__ == '__main__':
    colors = []
    window_name = "frame"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, MouseEvent)  # 窗口与回调函数绑定

    # g_color()
    g_color2()

    console_h = 40
    # cell_w = 18
    # cell_h = 12
    cell_w = 36
    cell_h = 6
    n = 36
    side = 10
    h, w = cell_h * n, cell_w * n
    frame = np.zeros((h+40, w, 3), dtype=np.uint8)
    for index, color in enumerate(colors):
        x1, y1 = index // cell_h, index % cell_h
        x2, y2 = x1 + 1, y1 + 1
        cv2.rectangle(frame, (x1 * n, y1 * n), (x2 * n, y2 * n), color[::-1], -1)
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)
    run1()
    # run2()
