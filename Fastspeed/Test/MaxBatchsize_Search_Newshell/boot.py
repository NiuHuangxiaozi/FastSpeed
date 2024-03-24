import os
import subprocess
import numpy as np
from typing import List
# ########################################################################################################################
# #   do not delete the following comments
# #__Dependency__#
# MODEL_NAME="#__MODEL_NAME__#"
# MODEL_ARGS="#__MODEL_ARGS__#"
# OPTIMIZER_NAME="#__OPTIMIZER_NAME__#"
# CRITERION_NAME="#__CRITERION_NAME__#"
# DEVICE="#__DEVICE__#"
#
# CEILING_BATCHSIZE="#__CEILING_BATCHSIZE__#"
# TEMPSAMPLE_PATH="#__TEMPSAMPLE_PATH__#"
# ########################################################################################################################



def list2str(s:List[str])->str:
    result=""
    for item in s:
        result+=(item+'\n')
    return result

def repalce_macro(data:str,d:dict)->str:
    for initial_value in d:
        data=data.replace(initial_value,d[initial_value])
    return data
def main():
    file_path= "template.py"
    target_file_path="instance.py"
    data=None


    dependency_list=["from model import *"]
    macros={
        "#__MODEL_NAME__#":"AlexNet",
        "#__MODEL_ARGS__#":"*[10]",

        "#__OPTIMIZER_NAME__#":"Adam",
        "#__CRITERION_NAME__#":"CrossEntropyLoss",
        "#__DEVICE__#":"0",
        "#__CEILING_BATCHSIZE__#":"1000",
        "#__TEMPSAMPLE_PATH__#":"./Data/cifar_testsample.pt"
    }
    with (open(file_path, 'r', encoding='utf-8') as f):
        data = f.read()
        data=data.replace("#__Dependency__#",list2str(dependency_list))
        data=repalce_macro(data,macros)

    with open(target_file_path, 'w', encoding='utf-8') as f:
         f.write(data)
    cmd="python instance.py"
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    print("The max batchsize is ",result.stdout.decode('utf-8'))

if __name__=="__main__":
    main()