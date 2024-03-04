import argparse
import os
import json


'''
这里的代码来源于：https://blog.csdn.net/kiong_/article/details/135492019
'''
class Params():
    def __init__(self,json_path):
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)



