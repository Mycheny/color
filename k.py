import os

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
import collections


def getColorList():
    dict = collections.defaultdict(list)

    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list

    # #灰色
    # lower_gray = np.array([0, 0, 46])
    # upper_gray = np.array([180, 43, 220])
    # color_list = []
    # color_list.append(lower_gray)
    # color_list.append(upper_gray)
    # dict['gray']=color_list

    # 白色
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list

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
    dict['red'] = (color_list1, color_list2)

    # 橙色
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    # 黄色
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    # 绿色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    # 青色
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list

    # 蓝色
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list

    # 紫色
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list

    return dict


def get_n_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_dict = getColorList()
    count = 0
    for d in color_dict:
        if isinstance(color_dict[d], list):
            mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        elif isinstance(color_dict[d], tuple):
            mask1 = cv2.inRange(hsv, color_dict[d][0][0], color_dict[d][0][1])
            mask2 = cv2.inRange(hsv, color_dict[d][1][0], color_dict[d][1][1])
            mask = mask1 + mask2
        pro = np.sum(mask == 255)/mask.size
        print(d, pro)
        if pro>0.1/len(color_dict.keys()):
            count+=1
    print(count)
    return count


if __name__ == '__main__':
    root = "服饰"
    color_list = os.listdir(root)
    a = 0
    for color in color_list:
        # if "拼色" in color or "花色" in color:
        #     continue
        file_names = os.listdir(os.path.join(root, color))
        for file_name in file_names:
            file_path = os.path.join(root, color, file_name)
            file_path = file_path.encode('utf-8').decode('utf-8')
            # 加载sklearn中样图
            # china = cv2.imread(file_path)
            # file_path = "hua2.jpg"
            china = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
            ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
            if china is None:
                tmp = imageio.mimread(file_path)
                if tmp is not None:
                    imt = np.array(tmp)
                    imt = imt[0]
                    china = imt[:, :, 0:3]

            china=cv2.cvtColor(china, cv2.COLOR_RGB2BGR)


            n_colors = get_n_color(china)
            # cv2.imwrite(f"image2/{n_colors}_{a}.jpg", china)
            a+=1
            # cv2.imshow("image", china)
            # cv2.waitKey(800)
            # n_colors = 5
            china = np.array(china, dtype=np.float64) / 255

            # 加载图像并转换成二维数字阵列
            w, h, d = original_shape = tuple(china.shape)
            assert d == 3
            image_array = np.reshape(china, (w * h, d))
            print("一个小样本数据拟合模型")
            t0 = time()
            image_array_sample = shuffle(image_array, random_state=0)[:1000]
            kmeans = KMeans(n_clusters=n_colors,
            random_state=0).fit(image_array_sample)
            print("完成时间 %0.3fs." % (time() - t0))
            # Get labels for all points
            print("预测全图像上的颜色指数（k-均值）")
            t0 = time()
            labels = kmeans.predict(image_array)
            print("完成时间 %0.3fs." % (time() - t0))

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

            # 与原始图像一起显示所有结果
            plt.figure(1)
            plt.clf()
            ax = plt.axes([0, 0, 1, 1])
            plt.axis('off')
            plt.title('Original image (96,615 colors)')
            plt.imshow(china)
            plt.figure(2)
            plt.clf()
            ax = plt.axes([0, 0, 1, 1])
            plt.axis('off')
            plt.title('Quantized (64 colors, K-Means)')
            plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w,
            h))
            plt.show()