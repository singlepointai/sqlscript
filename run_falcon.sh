export HUGGING_FACE_HUB_TOKEN="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"


nohup vllm serve "/data/falcon3-7b-instruct" \
     --tokenizer-mode auto \
     --load-format auto \
     --config-format auto \
     --max-model-len 3000 \
     --gpu_memory_utilization 0.90 \
     --cpu-offload-gb 1.5 \
     --seed 369 \
     --port 8000 \
     --host 0.0.0.0 \
    --dtype half \
     > falcon.log 2>&1 &


