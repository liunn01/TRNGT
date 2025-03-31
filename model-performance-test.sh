#!/bin/bash

#设置日志输出目录
mkdir ./vllm-server-log
mkdir ./benchamark-output-log

#设置变量
current_time=$(date +"%Y-%m-%d-%H-%M-%S")
model='/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1'
suffix="DeepSeek-R1"
port=8335
InputLen=200
OutputLen=2000
NumPrompts=128

echo "此次用于测试模型$model的性能,时间为: $current_time" |tee -a ./benchamark-output-log/BenchamarkOutput-$suffix-$current_time
echo "输入长度为$InputLen tokens" |tee -a ./benchamark-output-log/BenchamarkOutput-$suffix-$current_time
echo "输出长度为$OutputLen tokens" |tee -a ./benchamark-output-log/BenchamarkOutput-$suffix-$current_time

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

#启动vLLM instance
#    --max-num-seqs 128 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve $model \
    --port $port \
    -tp 8 \
    --max-num-batched-tokens 64000 \
    --num-speculative-token 1 \
    --trust-remote-code \
    --enable-reasoning \
    --reasoning-parser deepseek_r1   > ./vllm-server-log/$current_time 2>&1    &

# wait until vLLM instances are ready
wait_for_server $port

sleep 1

#测试模型性能 
echo "##############开始测试$model模型性能##############"

#设置最大并发大小
for i in {128,64,32,16,8,4}
do
	echo "######################################最大并发数为$i的性能测试结果--启动开始----###########################################" |tee -a ./benchamark-output-log/BenchamarkOutput-$suffix-$current_time
	python3 /deepseek/backup/vllm-main/benchmarks/benchmark_serving.py --backend vllm \
	--dataset-name random \
	--random-input-len $InputLen \
	--random-output-len $OutputLen \
	--model $model \
	--host "localhost" \
	--port $port \
	--num-prompts $NumPrompts \
	--seed 1100  \
	--max-concurrency $i \
	| grep -A 19 'Maximum request concurrency' |grep -v "^Mean"|grep -v "P99"|grep -v "tokens"|grep -v "Time" |grep -v "Inter"|tee -a ./benchamark-output-log/BenchamarkOutput-$suffix-$current_time 
	echo "######################################最大并发数为$i的性能测试--已经结束----###############################################" | tee -a ./benchamark-output-log/BenchamarkOutput-$suffix-$current_time
done

# Cleanup commands
pgrep python | xargs kill -9
pkill -f python

