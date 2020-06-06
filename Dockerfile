# Base Images
#FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3
#FROM registry.cn-shanghai.aliyuncs.com/aliseccompetition/tensorflow:1.1.0-devel-gpu
#FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:latest-py3
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.1.0-cuda10.0-py3
ADD . /competition
WORKDIR /competition
RUN chmod 777 -R /competition
#RUN pip --no-cache-dir install  -r requirements.txt -i https://pypi.douban.com/simple
RUN pip --no-cache-dir install  -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
#RUN pip --no-cache-dir install  pyentrp
CMD ["sh", "run.sh"]
