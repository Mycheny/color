import os

import imageio
import numpy as np
import cv2
from matplotlib import pyplot as plt


def a(img):
    height = 128
    width = int(img.shape[1] * height / img.shape[0])
    img = cv2.resize(img, (width, height))
    image = np.copy(img)
    h, w, c = img.shape
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    h, w, c = img.shape
    pros = []
    kernel = np.ones((5, 5), np.float32) / 25
    kerne2 = np.ones((10, 10), np.float32) / 25
    for i  in [0.05, 0.08, 0.1, 0.15][::-1]:
        boder = i
        rect = (int(w*boder), int(h*boder), int(w*(1-boder*2)), int(h*(1-boder*2)))
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        dst = cv2.filter2D(mask2, -1, kerne2)
        opening = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        # closing = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        pro = mask2.sum() / mask2.size
        pros.append(pro)
        cv2.imshow("11", mask2*255)
        cv2.imshow("dst", dst*255)
        cv2.imshow("opening", opening*255)
        # cv2.imshow("closing", closing*255)
        mask2 = opening
        cv2.waitKey(500)
    print(pros)
    img = img * mask2[:, :, np.newaxis]
    img += 255 * (1 - cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR))
    # mean = np.mean(img)
    # img = img - mean
    # img = img * 0.9 + mean * 0.9
    # img = img.astype(np.uint8)
    # cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 0), 1)
    cv2.imshow("", np.hstack((image, img)))
    cv2.waitKey(2000)


if __name__ == '__main__':
    root = "no_label"
    color_list = os.listdir(root)
    color_list = ["没有标注的颜色4500张图片"]
    count = 0
    ok = 0
    for color in color_list:
        if "拼色" in color or "花色" in color:
            continue
        file_names = os.listdir(os.path.join(root, color))
        for file_name in file_names[5:]:
            file_name = "-103543582.jpg"
            print(file_name, end="  ")
            # file_name = "Screenshot_20200223_201324_com.taobao.taobao.jpg"
            file_path = os.path.join(root, color, file_name)
            file_path = file_path.encode('utf-8').decode('utf-8')
            china = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
            if china is None:
                tmp = imageio.mimread(file_path)
                if tmp is not None:
                    imt = np.array(tmp)
                    imt = imt[0]
                    china = imt[:, :, 0:3]
            if china.shape[2] == 4:
                china = china[..., :3]
            a(china)
