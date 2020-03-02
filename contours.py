import os
import re

import cv2
import imageio
import numpy as np


def takeSecond(elem):
    return elem[1]


if __name__ == '__main__':
    root = "服饰"
    color_list = os.listdir(root)
    count = 0
    ok = 0
    for color in color_list:
        if not "拼色" in color or not "花色" in color:
            file_names = os.listdir(os.path.join(root, color))
            for file_name in file_names:
                file_path = os.path.join(root, color, file_name)
                # file_path  = "服饰\柠檬黄\柠檬黄 (1).jpg"
                # color = "柠檬黄"
                file_path = file_path.encode('utf-8').decode('utf-8')
                china = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
                if china is None:
                    tmp = imageio.mimread(file_path)
                    if tmp is not None:
                        imt = np.array(tmp)
                        imt = imt[0]
                        china = imt[:, :, 0:3]
                color_name = re.sub("[\(\)（） 0-9]*", "", color)
                if china.shape[2] == 4:
                    china = china[..., :3]
                image = china
                h, w, ch = image.shape
                result = np.zeros((h, w, ch), dtype=np.uint8)
                thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, binary = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                _, contours, __ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                areas = [cv2.contourArea(contours[i]) / binary.size for i in range(len(contours))]
                datas = [[contour, area] for contour, area in zip(contours, areas) if area < 0.95 and area > 0.0001]
                datas.sort(key=takeSecond, reverse=True)
                datas = [[datas[i][0], datas[i][1], np.mean(datas[i][0], axis=0)[0].astype(np.int)] for i in
                         range(len(datas))]
                datas = [data for data in datas if
                         data[2][0] > w * 0.1 and data[2][0] < w * 0.9 and data[2][1] > h * 0.1 and data[2][
                             1] < h * 0.9]

                [cv2.drawContours(result, [data[0] for data in datas], i, (
                int(np.random.random() * 255), int(np.random.random() * 255), int(np.random.random() * 255)), -1)
                 for i in range(len(datas))]
                [cv2.circle(result, tuple(np.mean(datas[i][0], axis=0)[0].astype(np.int)), 3, (0, 0, 255)) for i in
                 range(len(datas))]
                print(len(datas))
                a = 1
                if len(datas) > 10:
                    a = 0
                cv2.imshow("thresh", thresh)
                cv2.imshow("res", result)
                cv2.waitKey(a)
