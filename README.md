# 安装依赖

python 3.11.13

```shell
uv sync
```

# 启动Clip服务

> https://clip-as-service.jina.ai/index.html

```shell
# 下载预训练模型并启动
JINA_LOG_LEVEL=DEBUG python -m clip_server
```

# 启动API服务

```shell
python main.py
```

http://localhost:8000/docs