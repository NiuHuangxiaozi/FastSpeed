import os



#这个里面是一些帮助debug的函数


__all__ = ["global_rank_print", ]




# print function
def global_rank_print(
   target_rank:int,
   content:str):
    if int(os.environ["RANK"])==target_rank:
        print(str)


#exception function
class MyException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message

    def __str__(self):
        return f"{self.code}: {self.message}"






