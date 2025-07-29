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
    --input data/第三批-20250728 \
    --output output/第三批 \
    --min_duration 0.5 \
    --max_duration 5
```

input：待剪辑的视频所在的路径

output：结果输出的目录  
  
min_duration：最小视频片段长度  
  
max_duration：最大视频片段长度

### 执行

> sh run.sh





## 致谢

本项目参考了大量代码来自于：https://github.com/mli/autocut

