
import torch
import time



def matrix_multiply(device, size=2048, duration=3600):
    # 创建在指定设备上的大矩阵
    A = torch.rand(size, size, device=device)
    B = torch.rand(size, size, device=device)
    start_time = time.time()

    # 持续执行矩阵乘法，模拟长时间计算
    while time.time() - start_time < duration:
        C = torch.matmul(A, B)
        # 通过添加额外操作确保计算不会被优化掉
        A.add_(C)

    print(f"Completed long-running matrix multiplication on {device}")



if __name__ == '__main__':
    from threading import Thread
    device=[0,1]
    maxtrix_size=[512,512]
    threads=[]
    for device_index in range(len(device)):
        # 使用多线程在多个GPU上并行加载任务
        threads.append(Thread(target=matrix_multiply, args=(device[device_index],maxtrix_size[device_index])))

    for _thread in threads:
        _thread.start()

    print("Running background tasks on GPU 0 and GPU 1")
