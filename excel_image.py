# -*- coding: utf-8 -*- 
# @Time 2020/3/4 17:50
# @Author wcy

import pandas as pd
import numpy as np

columns = [['a', 'b', 'c', 'd', 'e']]  # 创建形状为（10，5） 的DataFrame 并设置二级标题
demo_df = pd.DataFrame(np.arange(15).reshape(3, 5), columns=columns)
print(demo_df)


def style_color(df, colors):
    """

    :param df: pd.DataFrame
    :param colors: 字典  内容是 {标题:颜色}
    :return:
    """
    return df.style.apply(style_apply, colors=colors)


def style_apply(series, colors, back_ground=''):
    series_name = series.name[0]
    if series_name == "a":
        a = [f'background-color: {color}' for color in colors]

    else:
        a = ["" for i in range(3)]
    return a

style_df = style_color(demo_df, ['#1C1C1C', '#00EEEE', '#1A1A1A'])

with pd.ExcelWriter('df_style.xlsx', engine='openpyxl') as writer:  # 注意： 二级标题的to_excel index 不能为False
    style_df.to_excel(writer, sheet_name='sheet_name')