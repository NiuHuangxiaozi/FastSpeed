import os



#这个里面是一些帮助debug的函数


__all__ = ["global_rank_print", ]




# print function
def global_rank_print(
   target_rank:int,
   content:str
):
    if int(os.environ["RANK"])==target_rank:
        print(content)


#exception function
class MyException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message

    def __str__(self):
        return f"{self.code}: {self.message}"


################################################
#MyException(0, "partition_list or batchsize_list can't be None in the manual mode!")
#MyException(1,"Your choice of partition_method("+str(partition_method)+") is wrong [neither manual or autobalanced!")

#



################################################






