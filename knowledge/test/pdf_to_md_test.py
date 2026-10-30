"""
  @Author:桌角是小黑
  @Time:2026/10/8
  @Desc:
"""
import os
import subprocess
from pathlib import Path


# 方法2：显式设置关键环境变量（确保 MinerU 使用 ModelScope 离线模式）
os.environ['MINERU_MODEL_SOURCE'] = 'modelscope'
os.environ['MODELSCOPE_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("✅ 环境变量已配置")
print(f"   MINERU_MODEL_SOURCE = {os.environ.get('MINERU_MODEL_SOURCE')}")
print(f"   MODELSCOPE_OFFLINE = {os.environ.get('MODELSCOPE_OFFLINE')}")
# print(f"   HF_HOME = {os.environ.get('HF_HOME')}")
print()
proc = subprocess.Popen(
    args=["mineru", "-p", r"C:/Users/43779/Desktop/AIlearnrecord/shopkeeper_brain_docs/doc/万用表RS-12的使用.pdf",
          "-o", r"output/", "--backend", "pipeline"],
    stdout=subprocess.PIPE,  # 捕获标准输出
    stderr=subprocess.STDOUT,  # 合并错误到标准输出
    text=True,
    encoding="utf-8",
    errors="replace",  # 遇到乱码时替换
    bufsize=1  # 行缓冲，实时输出
)

for line in proc.stdout:
    print(line.rstrip())

return_code = proc.wait()
print(return_code)
