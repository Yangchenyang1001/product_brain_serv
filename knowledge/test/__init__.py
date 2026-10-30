"""
测试代码目录

存放项目的单元测试和集成测试代码。
"""
import torch

print(f"{torch.cuda.is_available()}")
print(f"{torch.cuda.get_device_name()}")  # 必须安装GPU版本的对应PyTorch框架
