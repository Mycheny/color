import os

import imageio
import numpy as np
import cv2
from matplotlib import pyplot as plt


def a(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    h, w, c = img.shape
    rect = (20, 20, 760, 760)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    img += 255 * (1 - cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR))
    # plt.imshow(img)
    # plt.show()
    img = np.array(img)
    mean = np.mean(img)
    img = img - mean
    img = img * 0.9 + mean * 0.9
    img /= 255
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    root = "test"
    color_list = os.listdir(root)
    count = 0
    ok = 0
    for color in color_list:
        if "拼色" in color or "花色" in color:
            continue
        file_names = os.listdir(os.path.join(root, color))
        for file_name in file_names:
            file_path = os.path.join(root, color, file_name)
            for file_path in os.listdir("E:\PycharmProjects\服饰颜色识别\image7\土色"):
                file_path  = os.path.join("E:\PycharmProjects\服饰颜色识别\image7\土色", file_path)
                # color = "黄色"
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
