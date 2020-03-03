# -*- coding: utf-8 -*- 
# @Time 2020/3/3 13:00
# @Author wcy

import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36',
    'Host': 'htmlcolorcodes.com',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}


def name_is_exists1(tag):
    return tag.name == "article"


def name_is_exists2(tag):
    return tag.name == "table"


if __name__ == '__main__':

    url = "http://htmlcolorcodes.com/zh/yanse-ming/"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        res = response.text.encode('ISO-8859-1').decode('utf-8')
        soup = BeautifulSoup(res, 'lxml')
        titles = soup.find(name="article")
        datas = {}
        sections = titles.find_all(name="section")
        for section in sections:
            name = section.attrs.get("id")
            h4 = section.find_all(name="h4")
            colors = [[h4[i * 3].text, h4[i * 3 + 1].text, h4[i * 3 + 2].text] for i in range(int(len(h4) / 3))]
            colors_dict = {name: rgb for name, h, rgb in colors}
            if len(colors) > 0: datas[name] = colors_dict
        print()
