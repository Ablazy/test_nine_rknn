FROM python:3.10-slim


WORKDIR /test_nine

# 复制项目文件
COPY . /test_nine

# 安装依赖
RUN pip install --no-cache-dir -r requirements_without_train.txt

# 暴露端口
EXPOSE 9646

# 启动服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9646"]