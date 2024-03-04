import os



#这个里面是一些帮助debug的函数


__all__ = ["global_rank_print", ]


#
def global_rank_print(
   target_rank:int,
   content:str):
    if int(os.environ["RANK"])==target_rank:
        print(str)
