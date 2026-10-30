## 生成bge M3 的远程模型发布程序
【核心目标】
帮我生成一份 发布bge的远程模型服务代码
【已知环境】
已经把bgem3下载到了gpu服务器上, 目录地址/usr-data/models/BAAI/bge-m3
【要求】
1. 模型服务的远程接口的传入值和返回值  参考C:\Users\43779\Desktop\AIlearnrecord\shopkeeper_brain\knowledge\utils\client\ai_clients.py 
2. 生成的服务代码放在了 knowedge/deploy文件下
3. 使用uvcorn作为web服务框架 端口6006
4. 适用fastapi
5. 文件命名deploy_bgem3_service.py
6. 请求和响应和本地模型调用一致


## 生成bge reranker 的远程模型发布程序
【核心目标】
帮我生成一份 发布bge的远程模型服务代码
【已知环境】
已经把bgem3下载到了gpu服务器上, 目录地址/usr-data/models/BAAI/bge-reranker-large
【要求】
1. 模型服务的远程接口的传入值和返回值 参考C:\Users\43779\Desktop\AIlearnrecord\shopkeeper_brain\knowledge\utils\client\ai_clients.py
2. 生成的服务代码放在了 deploy文件下
3. 使用uvcorn作为web服务框架 端口6006
4. 适用fastapi
5. 文件deploy_bgereranker_service.py
6. 请求和响应和本地模型调用一致
