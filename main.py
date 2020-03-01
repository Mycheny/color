import colorsys
import csv
import os
import re
import numpy as np
import xlrd
import cv2
import PIL.Image as Image
from PIL import ImageFont, ImageDraw
from sklearn.cluster import DBSCAN, KMeans


def get_color_dict():
    color_dict = {}
    file_path = r'E:\PycharmProjects\服饰颜色识别\颜色\猪圈关键字（55色）(4)(1).xlsx'
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
            color_dict[color_name1] = [int(i) for i in color_value1.split(" ")]
        if re.match('[0-9]* [0-9]* [0-9]*', color_value2):
            color_dict[color_name2] = [int(i) for i in color_value2.split(" ")]


    # # 文件路径的中文转码，如果路径非中文可以跳过
    # file_path = file_path.encode('utf-8').decode('utf-8')
    # # 获取数据
    # data = xlrd.open_workbook(file_path)
    # table = data.sheet_by_name('猪圈颜色')
    # # 获取总行数
    # nrows = table.nrows
    # # 获取总列数
    # ncols = table.ncols
    # # 获取一行的数值，例如第5行
    # for i in range(nrows):
    #     rowvalue = table.row_values(i)
    #     color_name1 = rowvalue[1]
    #     color_value1 = rowvalue[2]
    #     color_name2 = rowvalue[4]
    #     color_value2 = rowvalue[5]
    #     if re.match('[0-9]* [0-9]* [0-9]*', color_value1):
    #         color_dict[color_name1] = [int(i) for i in color_value1.split(" ")]
    #     if re.match('[0-9]* [0-9]* [0-9]*', color_value2):
    #         color_dict[color_name2] = [int(i) for i in color_value2.split(" ")]
    return color_dict


def get_dominant_color(image):
    def flatten(a):
        for each in a:
            if not isinstance(each, list):
                yield each
            else:
                yield from flatten(each)

    max_score = 0.0001
    dominant_color = None
    height = 128
    width = int(image.height*height/image.width)
    train_image = image.resize((width, height),Image.ANTIALIAS)
    dataset = [[(r, g, b)]*count for count, (r, g, b) in train_image.getcolors(train_image.size[0] * train_image.size[1]) if
         (min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235) - 16.0) / (235 - 16) < 0.9]
    dataset = list(flatten(dataset))
    y_pred1 = DBSCAN().fit_predict(dataset)
    y_pred2 = KMeans(n_clusters=3, random_state=9).fit_predict(dataset)
    for count, (r, g, b) in image.getcolors(image.size[0] * image.size[1]):
        # 转为HSV标准
        hue, saturation, value = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
        y = (y - 16.0) / (235 - 16)
        # 忽略高亮色
        if y > 0.9:
            continue
        score = (saturation + 0.1) * count
        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)
    return dominant_color


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
    np.random.seed(0)
    # s = np.random.rand(180, 353, 3)*255
    # image = s.astype(np.uint8)
    # image = np.zeros((180, 353, 3), dtype=np.uint8)
    a = 3
    h = np.expand_dims(np.tile(np.arange(0, 180, dtype=np.uint8), (353 * a, 1)).T, axis=2)
    sv = np.ones((180, 353 * a, 2), dtype=np.uint8) * 255
    frame = np.concatenate((h, sv), axis=2)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    color_dict = get_color_dict()
    root = "服饰"
    color_list = os.listdir(root)
    count, ok = 0, 0
    f = open('res.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["标注的", "预测的"])
    for color in color_list:
        if "拼色" in color or "花色" in color:
            continue
        file_names = os.listdir(os.path.join(root, color))
        for file_name in file_names:
            file_path = os.path.join(root, color, file_name)
            file_path = "服饰/朱红/朱红 (2).png"
            color = "朱红"

            image = Image.open(file_path)
            image = image.convert('RGB')
            dominant_color = get_dominant_color(image)
            similar_color = ""
            temp = 0
            for color_name, (r, g, b) in color_dict.items():
                v1 = np.array(dominant_color)
                v2 = np.array([r, g, b])
                num = float(v1.dot(v2.T))
                denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                cos = num / denom  # 余弦值
                sim = 0.5 + 0.5 * cos  # 根据皮尔逊相关系数归一化
                if sim > temp:
                    temp = sim
                    similar_color = color_name
            label_color = color_dict[color]
            lh, ls, lv = colorsys.rgb_to_hsv(label_color[0] / 255.0, label_color[1] / 255.0, label_color[2] / 255.0)
            dh, ds, dv = colorsys.rgb_to_hsv(dominant_color[0] / 255.0, dominant_color[1] / 255.0,
                                             dominant_color[2] / 255.0)
            # cv2.circle(frame, (count, int(lh*180)), 1, (0, 0, 0))
            # cv2.circle(frame, (count, int(dh*180)), 1, (255, 255, 255))
            cv2.line(frame, (count * a, int(lh * 180)), ((count * a) + 2, int(lh * 180)), (0, 0, 0))
            cv2.line(frame, (count * a, int(dh * 180)), ((count * a) + 2, int(dh * 180)), (255, 255, 255))
            cv2.imshow("", frame)
            cv2.waitKey(100)
            print(color, similar_color, dominant_color, color_dict[color], temp)
            image2 = np.array(image)
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image2, (0, 0), (70, 70), color_dict[color][::-1], -1)
            image2 = cv2ImgAddText(image2, f"标注为{color}", 10, 80, textSize=14)

            cv2.rectangle(image2, (80, 0), (150, 70), dominant_color[::-1], -1)
            image2 = cv2ImgAddText(image2, f"检测为{similar_color}", 90, 80, textSize=14)

            file_path = f"image/{color}_{count}.jpg".encode('utf-8').decode('utf-8')
            # cv2.imwrite(file_path, image2)
            cv2.imencode('.jpg', image2)[1].tofile(file_path)
            csv_writer.writerow([color, similar_color])
            if color == similar_color:
                ok += 1
            count += 1
    f.close()
    cv2.imshow("", frame)
    cv2.waitKey(0)
    print(ok, "/", count)
