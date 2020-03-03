# -*- coding: utf-8 -*- 
# @Time 2020/3/3 16:19
# @Author wcy

import requests
response = requests.post(
     'https://api.remove.bg/v1.0/removebg',
     files={'image_file': open('1111.jpg', 'rb')}, #这里填写图片路径
 # 这里填写图片路径
    data={'size': 'auto'},
     headers={'X-Api-Key': '1UxvQqHUmHAYPQ1kENm6vidj'}, #这里替换成你的api key
 )
if response.status_code == requests.codes.ok:
     with open('no-bg.png', 'wb') as out:
         out.write(response.content)
else:
    print("Error:", response.status_code, response.text)