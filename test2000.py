# -*- coding: utf-8 -*- 
# @Time 2020/3/5 16:15
# @Author wcy
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor

from rgb2lab import RGB2Lab


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
    delta_e = delta_e_cie2000(color1, color2, Kl=3, Kc=1, Kh=1) # Kl 亮度, Kc 饱和度, Kh 色调 的权重
    return delta_e, lab1, lab2


if __name__ == '__main__':
    a = [221, 160, 221]
    b = [93, 91, 145]
    c = [173, 216, 230]
    d = [27, 38, 38]

    p1, lab1, lab11 = color_distance(a, b)
    p2, lab2, lab22 = color_distance(c, d)
    p3, lab3, lab33 = color_distance(d, b)
    print()
