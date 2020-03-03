# -*- coding: utf-8 -*- 
# @Time 2020/3/3 20:54
# @Author wcy
import colorsys
import operator

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976, delta_e_cie1994, delta_e_cie2000

from rgb2lab import RGB2Lab

frame = None
last_rgb = [0, 0, 0]
sleep = 2000


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
    delta_e = delta_e_cie2000(color1, color2)
    return delta_e


def run1():
    global frame
    for i in range(10000):
        color1 = [np.random.randint(0, 255) for ii in range(3)]
        distances = [color_distance(color1, color2) for color2 in colors]
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
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)



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
    global frame, last_rgb, sleep, console_h
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


if __name__ == '__main__':
    window_name = "frame"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, MouseEvent)  # 窗口与回调函数绑定
    colors = []
    for i in range(6):
        for j in range(6):
            for k in range(6):
                colors.append((i * 51, j * 51, k * 51))
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
