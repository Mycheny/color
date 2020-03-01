import os
import re

if __name__ == '__main__':
    root = "服饰"
    path_list = os.listdir(root)
    for name in path_list:
        src_path = os.path.join(root, name)
        name = re.sub("[\(\)（） 0-9]*", "", name)
        dst_path = os.path.join(root, name)
        os.rename(src_path, dst_path)