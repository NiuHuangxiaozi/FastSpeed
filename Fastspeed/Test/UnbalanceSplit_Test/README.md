### 构件：不均匀划分数据的策略验证

#### 简单介绍：

这个测试是为了检测模型的sampler.py模块是否正常的能够进行异构数据划分。我检测的方法是创建了一个特别简单的数据集，只有50个简单的样例，然后分别在2个和4个GPU上测试了数据划分的效果，测试结果显示成功进行了划分，没有发现数据分错、发重的问题。





#### 新的体会：

并且在测试的过程中我们还发现了一个新的知识：

在这段代码中：

```python
sampler = Distributed_Elastic_Sampler(dataset=dataset,shuffle=True, partition_strategy=sampler_dict)
train_loader = DataLoader(dataset=dataset,
                              batch_size=batchsize_list[global_rank],
                              shuffle=False,  # 这个值必须设置为false，否则会导致多个节点可能都抽到同一个样例的结果
                              sampler=sampler,
                              pin_memory=True,
                              num_workers=4
                              )
```







##### 第一个shuffle：

```python
if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
```

通过上面的代码，设置为True这样我们每次会对抽样的索引进行随机的打乱，这里看到对这个索引数组的打乱只取决于seed和opoch，这个每一个节点都是一样的，seed都设成0，epoch每一轮大家都一样，通过train_loader.sampler.set_epoch(epoch)对epoch进行更新。

##### 第二个shuffle：

只能设置为false，因为sampler参数和shuffle=True是互斥的，所以使用sampler就不能设置为True。

##### 结果：

基于上述的发现，我将在框架中：Fastspeed.py文件中设置sampler的时候加上shuffle=True,

```python
# 3
                ########################################################################################################
                # 这里在计算完batchsize_list之后我们需要计算一下各个梯度的权重比例
                # 然后调用Distributed_Elastic_Sampler和相应的dataloader进行数据的划分
                self.grad_portion = batchsize_list / np.sum(batchsize_list)

                sampler_dict = \
                    {
                        'method': "uneven",
                        'partition_list': partition_list
                    }

                sampler = Distributed_Elastic_Sampler(dataset=dataset, shuffle=True,partition_strategy=sampler_dict)
                train_loader = DataLoader(dataset=dataset,
                                          batch_size=batchsize_list[self.dist_args.global_rank],
                                          shuffle=False,  # 这个值必须设置为false
                                          sampler=sampler,
                                          pin_memory=self.config.data["train_loader_pin_memory"],
                                          num_workers=self.config.data["train_loader_num_workers"]
                                          )
                return train_loader
            else:
                raise MyException(1,
                                  "Your choice of partition_method(" + partition_type + ") is wrong [neither manual or autobalanced!")
```

