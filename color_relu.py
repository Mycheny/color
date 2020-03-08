# -*- coding: utf-8 -*- 
# @Time 2020/3/5 14:11
# @Author wcy


# RGB格式颜色转换为16进制颜色格式
import re

import cv2
import numpy as np
import xlrd


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


def get_costume_color_dict(file_path=r'E:\PycharmProjects\服饰颜色识别\颜色\猪圈关键字（55色）(4)(1).xlsx'):
    costume_color_dict = {}
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
            costume_color_dict[color_name1] = [int(i) for i in color_value1.split(" ")]
        if re.match('[0-9]* [0-9]* [0-9]*', color_value2):
            costume_color_dict[color_name2] = [int(i) for i in color_value2.split(" ")]
    return costume_color_dict


if __name__ == '__main__':
    costume_color_dict = get_costume_color_dict()
    colors = [RGB_list_to_Hex(color) for color in costume_color_dict.values()]
    [colors.insert(i*2, "#000000") if i % 2 == 0 else colors.insert(i*2, "#FFFFFF") for i in range(len(colors) + 1)]
    colors0x = gradient_color(colors, color_sum=512)
    colors2 = [[int(f'0x{color[1:3]}', 16), int(f'0x{color[3:5]}', 16), int(f'0x{color[5:7]}', 16)] for color in colors0x]

    cell_w = len(colors2)
    cell_h = 1
    n = 5
    h, w = cell_h * n, cell_w * n
    frame = np.zeros((h + 40, w, 3), dtype=np.uint8)
    for index, color in enumerate(colors2):
        x1, y1 = index // cell_h, index % cell_h
        x2, y2 = x1 + 1, y1 + 1
        cv2.rectangle(frame, (x1 * n, y1 * n), (x2 * n, y2 * n), color[::-1], -1)
    cv2.imshow("window_name", frame)
    cv2.waitKey(0)
    print()