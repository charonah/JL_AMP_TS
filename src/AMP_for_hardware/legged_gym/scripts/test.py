import torch
import numpy as np

a = torch.div(torch.arange(10), (10/5), rounding_mode='floor').to(torch.long)
print(a)
b = torch.ones(5)
    
# 定义两个PyTorch张量
a = torch.tensor([1, 2, 3, 4, 4, 5])
b = torch.tensor([4, 5, 6, 7, 8])

# 将PyTorch张量转换为NumPy数组
a_np = a.numpy()
b_np = b.numpy()

# 使用NumPy找到共同的元素
common_elements_np = np.intersect1d(a_np, b_np)

# 将结果转换回PyTorch张量
common_elements = torch.from_numpy(common_elements_np)

print(common_elements)

# 定义一个PyTorch张量
a = torch.tensor([1, 2, 3, 4, 2, 3, 5])

# 使用逻辑运算符找出等于2或者3的元素的位置
mask = (a == 2) | (a == 3)

# 使用torch.where获取满足条件的元素的下标
indices = torch.where(mask)[0]

print(indices)


# 定义两个张量
a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([1,4, 5, 6, 7, 8])

# 使用torch.isin检查a中的元素是否存在于b中
mask = ~torch.isin(a, b)

# 选择a中不在b中的元素
a_not_in_b = a[mask]

print(a_not_in_b)