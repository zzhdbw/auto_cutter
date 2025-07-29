export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
python -m src.transcribe \
    --input data/第三批-20250728 \
    --output output/第三批