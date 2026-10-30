"""
  @Author:桌角是小黑
  @Time:2026/10/13
  @Desc:
"""
from modelscope import snapshot_download

local_dir = snapshot_download(model_id="BAAI/bge-m3",
                              local_dir=r"G:\AI\ai-models\modelscope_cache\models\BAAI")
print(local_dir)
