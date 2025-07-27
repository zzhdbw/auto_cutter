# auto_cutter
一个使用VAD、ASR自动剪辑视频的工具

## 快速启动

### 环境安装

> conda create -n auto_cutter python=3.11 -y
>
> conda activate auto_cutter
>
> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt



### 参数说明

```sh
python -m src.transcribe \
  --input data/第二批-20250725 \
  --output output/第二批
```

input：待剪辑的视频所在的路径

output：结果输出的目录



### 执行

> sh run.sh





## 致谢

本项目参考了大量代码来自于：https://github.com/mli/autocut

