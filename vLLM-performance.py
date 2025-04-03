import os
import sys
import time
import signal
import subprocess
import requests
import threading
from typing import List, Optional
from datetime import datetime
import psutil  # 确保导入psutil
import random  # 导入 random 模块
import csv  # 导入 csv 模块

# 强制使用无图形界面的后端
import matplotlib
matplotlib.use('Agg')  # 设置 matplotlib 后端为 Agg
import matplotlib.pyplot as plt  # 导入 matplotlib 模块


# 配置常量
LOG_DIRS = ["./vllm-server-log", "./benchamark-output-log"]
MODEL_PATH = "/local/models/DeepSeek-R1-Distill-Llama-8B"
PORT = 8335
INPUT_LEN = 20
OUTPUT_LEN = 20

class ProcessManager:
    """进程管理类，负责进程的启动、监控和清理"""
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.log_threads: List[threading.Thread] = []
        self._register_signals()

    def _register_signals(self):
        """注册信号处理"""
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)

    def start_process(self, cmd: List[str], log_file: str, prefix: str = "") -> subprocess.Popen:
        """启动进程并设置日志重定向"""
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 启动进程
        proc = subprocess.Popen(
            " ".join(cmd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        self.processes.append(proc)
        print(f"Started process: PID={proc.pid}, CMD={' '.join(cmd)}")

        # 启动日志记录线程
        def log_worker():
            try:
                with open(log_file, "a") as f:  # 使用追加模式
                    while proc.poll() is None:  # 检查进程是否已退出
                        line = proc.stdout.readline()
                        if line:
                            output = f"[{prefix}] {line.strip()}"
                            print(output)
                            f.write(f"{datetime.now().isoformat()} {output}\n")
            except Exception as e:
                print(f"Error in log worker: {e}")

        log_thread = threading.Thread(target=log_worker, daemon=True)
        log_thread.start()
        self.log_threads.append(log_thread)
        return proc

    def cleanup(self, signum=None, frame=None):
        """清理所有进程"""
        print("\n[Cleanup] Terminating processes...")
        
        # 终止子进程
        for proc in self.processes:
            try:
                print(f"Terminating process: PID={proc.pid}")
                proc.terminate()
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    print(f"Killing process: PID={proc.pid}")
                    proc.kill()
                except Exception as e:
                    print(f"Failed to kill process: {e}")
            finally:
                if proc.stdout:
                    proc.stdout.close()  # 确保释放资源

            # 强制终止子进程
            try:
                parent = psutil.Process(proc.pid)
                children = parent.children(recursive=True)  # 获取所有子进程
                for child in children:
                    print(f"Force killing child process: PID={child.pid}")
                    child.kill()
            except psutil.NoSuchProcess:
                pass

        # 清理残留进程
        self._kill_zombie_processes()
        self._kill_process_by_port(PORT)  # 通过端口杀掉进程
        self._kill_vllm_main_process(PORT)  # 新增：通过端口杀掉 vllm 主进程
        self._kill_gpu_processes()
        print("[Cleanup] Cleanup completed.")

    def _kill_zombie_processes(self):
        """清理残留的Python进程"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    if 'cmdline' in proc.info and proc.info['cmdline']:
                        cmdline = ' '.join(proc.info['cmdline'])
                        if 'vllm' in cmdline or 'benchmark_serving' in cmdline:
                            print(f"Force killing zombie process: PID={proc.info['pid']}, CMD={cmdline}")
                            proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def _kill_process_by_port(self, port: int):
        """通过端口查找并终止进程"""
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
                    pid = conn.pid
                    if pid:
                        proc = psutil.Process(pid)
                        print(f"Killing process listening on port {port}: PID={pid}, Name={proc.name()}")
                        proc.kill()
        except Exception as e:
            print(f"Failed to kill process on port {port}: {e}")

    def _kill_vllm_main_process(self, port: int):
        """通过端口查找并杀掉 vllm serve 主进程"""
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
                    pid = conn.pid
                    if pid:
                        proc = psutil.Process(pid)
                        print(f"Killing vllm serve main process: PID={pid}, Name={proc.name()}")
                        # 递归终止子进程
                        for child in proc.children(recursive=True):
                            print(f"Killing child process: PID={child.pid}, Name={child.name()}")
                            child.kill()
                        proc.kill()
                        return
        except Exception as e:
            print(f"Failed to kill vllm serve main process on port {port}: {e}")

    def _kill_gpu_processes(self):
        """通过nvidia-smi查找并终止占用显卡的进程"""
        try:
            # 使用nvidia-smi查找占用显卡的进程
            output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
            pids = output.decode().strip().split('\n')
            for pid in pids:
                if pid.strip():
                    pid = int(pid.strip())
                    try:
                        proc = psutil.Process(pid)
                        print(f"Force killing GPU process: PID={pid}, Name={proc.name()}")
                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
        except Exception as e:
            print(f"Failed to kill GPU processes: {e}")

def wait_for_server(port: int, timeout: int = 600) -> bool:
    """等待服务器启动"""
    print(f"Waiting for server on port {port}...")
    start_time = time.time()
    last_print = 0
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"http://localhost:{port}/v1/completions",
                timeout=5
            )
            if response.status_code == 405:
                print("\nServer is ready!")
                return True
        except (requests.ConnectionError, requests.Timeout):
            if time.time() - last_print > 30:
                print(".", end="", flush=True)
                last_print = time.time()
            time.sleep(1)
        except Exception as e:
            print(f"\nError checking server status: {str(e)}")
            break
    return False

def run_benchmark(pm: ProcessManager, config: dict, log_file: str):
    """执行基准测试"""
    cmd = [
        "vllm bench serve",
        "--dataset-name", "random",
        "--random-input-len", str(config['input_len']),
        "--random-output-len", str(config['output_len']),
        "--model", config['model'],
        "--host", "localhost",
        "--port", str(config['port']),
        "--num-prompts", str(config['num_prompts']),
        "--seed", str(config['seed']),
        "--max-concurrency", str(config['concurrency'])
    ]

    # 定义需要保留的行的正则表达式
    grep_patterns = [
        "Maximum request concurrency:",
        r"Successful requests:\s+[0-9]+",
        r"Benchmark duration \(s\):\s+[0-9.]+",
        r"Request throughput \(req/s\):\s+[0-9.]+",
        r"Output token throughput \(tok/s\):\s+[0-9.]+",
        r"Total Token throughput \(tok/s\):\s+[0-9.]+",
        r"Median TTFT \(ms\):\s+[0-9.]+",
        r"Median TPOT \(ms\):\s+[0-9.]+"
    ]

    # 构造过滤命令
    grep_cmd = " | ".join([
        " ".join(cmd),
        "grep -E --color=never '" + "|".join(grep_patterns) + "'"
    ])

    print(f"Running benchmark with concurrency={config['concurrency']}")
    print(f"Command: {grep_cmd}")

    try:
        # 使用 subprocess 直接执行过滤后的命令
        proc = subprocess.Popen(
            grep_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Benchmark failed with exit code {proc.returncode}: {stderr}")

        # 解析输出并格式化
        results = parse_benchmark_output(stdout)
        return results  # 返回解析后的结果
    except Exception as e:
        print(f"Error during benchmark with concurrency={config['concurrency']}: {e}")
        raise

def parse_benchmark_output(output: str) -> dict:
    """解析基准测试输出"""
    results = {}
    for line in output.splitlines():
        if "Maximum request concurrency:" in line:
            results["MaxConcurrency"] = line.split(":")[1].strip()
        elif "Successful requests:" in line:
            results["SuccessRequests"] = line.split(":")[1].strip()
        elif "Request throughput (req/s):" in line:
            results["RequestThroughput"] = line.split(":")[1].strip()
        elif "Output token throughput" in line:
            results["OutputTokenThroughput"] = line.split(":")[1].strip()
        elif "Total Token throughput" in line:
            results["TotalTokenThroughput"] = line.split(":")[1].strip()
        elif "Median TTFT" in line:
            results["TTFT"] = line.split(":")[1].strip()
        elif "Median TPOT" in line:
            results["TPOT"] = line.split(":")[1].strip()
    return results

def format_benchmark_results(results: dict) -> str:
    """格式化基准测试结果"""
    header = format_benchmark_header()
    row = format_benchmark_row(results)
    return header + "\n" + row

def format_benchmark_header() -> str:
    """生成基准测试结果的标题行"""
    return "{:<10} {:<10} {:<10} {:<15} {:<15} {:<10} {:<10}".format(
        "Max-Conc",  # Maximum-request-concurrency
        "Succ-Req",  # Successful-requests
        "Req/s",     # Request-throughput(req/s)
        "Out-Tok/s", # Output-token-throughput(tok/s)
        "Tot-Tok/s", # Total-Token-throughput(tok/s)
        "TTFT",      # Median-TTFT(ms)
        "TPOT"       # Median-TPOT(ms)
    )

def format_benchmark_row(results: dict) -> str:
    """生成基准测试结果的数据行"""
    return "{:<10} {:<10} {:<10} {:<15} {:<15} {:<10} {:<10}".format(
        results.get("MaxConcurrency", "N/A"),
        results.get("SuccessRequests", "N/A"),
        results.get("RequestThroughput", "N/A"),
        results.get("OutputTokenThroughput", "N/A"),
        results.get("TotalTokenThroughput", "N/A"),
        results.get("TTFT", "N/A"),
        results.get("TPOT", "N/A")
    )

def print_formatted_results(results: dict):
    """格式化并打印基准测试结果"""
    header = [
        "MaxConcurrency",  # 修改标题
        "SuccessRequests",
        "OutputTokenThroughput",
        "Total-Token-throughput",
        "TTFT",
        "TPOT"
    ]
    print("\t".join(header))  # 打印标题行
    row = [
        results.get("MaxConcurrency", "N/A"),  # 使用新的字段名称
        results.get("SuccessRequests", "N/A"),
        results.get("OutputTokenThroughput", "N/A"),
        results.get("Total-Token-throughput", "N/A"),
        results.get("TTFT", "N/A"),
        results.get("TPOT", "N/A")
    ]
    print("\t".join(row))  # 打印数据行

def plot_results(results_list, output_file="benchmark_results.png"):
    """根据基准测试结果生成图示"""
    # 提取数据
    max_conc = [int(result["MaxConcurrency"]) for result in results_list]
    req_per_sec = [float(result["RequestThroughput"]) for result in results_list]
    out_tok_per_sec = [float(result["OutputTokenThroughput"]) for result in results_list]
    tot_tok_per_sec = [float(result["TotalTokenThroughput"]) for result in results_list]
    ttft = [float(result["TTFT"]) for result in results_list]
    tpot = [float(result["TPOT"]) for result in results_list]

    # 设置图表大小
    plt.figure(figsize=(10, 6))

    # 绘制每个字段的折线图
    plt.plot(max_conc, req_per_sec, label="Req/s", marker="o")
    plt.plot(max_conc, out_tok_per_sec, label="Out-Tok/s", marker="o")
    plt.plot(max_conc, tot_tok_per_sec, label="Tot-Tok/s", marker="o")
    plt.plot(max_conc, ttft, label="TTFT", marker="o")
    plt.plot(max_conc, tpot, label="TPOT", marker="o")

    # 设置标题和标签
    plt.title("Benchmark Results", fontsize=16)
    plt.xlabel("Max-Conc", fontsize=12)
    plt.ylabel("Metrics", fontsize=12)

    # 设置纵轴刻度为每格 200
    plt.yticks(range(0, int(max(max(req_per_sec, out_tok_per_sec, tot_tok_per_sec, ttft, tpot)) + 200), 200))

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True, linestyle="--", alpha=0.6)

    # 保存图表为文件
    plt.savefig(output_file)
    print(f"Saved benchmark results plot to {output_file}")

    # 移除 plt.show()，避免在命令行中尝试显示图表

def main():
    # 初始化环境
    os.makedirs(LOG_DIRS[0], exist_ok=True)
    os.makedirs(LOG_DIRS[1], exist_ok=True)
    pm = ProcessManager()

    try:
        # 准备日志文件
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        server_log = os.path.join(LOG_DIRS[0], f"vllm-{current_time}.log")
        benchmark_log = os.path.join(LOG_DIRS[1], f"benchmark-{current_time}.log")
        csv_file = os.path.join(LOG_DIRS[1], f"benchmark-{current_time}.csv")  # CSV 文件路径
        plot_file = os.path.join(LOG_DIRS[1], f"benchmark-{current_time}.png")  # 图示文件路径

        # 启动vLLM服务
        vllm_cmd = [
            "CUDA_VISIBLE_DEVICES=0",
            "vllm", "serve", MODEL_PATH,
            "--port", str(PORT),
            "-tp", "1",
            "--max-num-batched-tokens", "64000",
            "--trust-remote-code"
        ]
        pm.start_process(vllm_cmd, server_log, "vLLM")

        # 等待服务启动
        if not wait_for_server(PORT, timeout=1200):
            raise RuntimeError("Server failed to start within timeout")

        # 写入 CSV 文件标题
        with open(csv_file, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "Max-Conc",  # Maximum-request-concurrency
                "Succ-Req",  # Successful-requests
                "Req/s",     # Request-throughput(req/s)
                "Out-Tok/s", # Output-token-throughput(tok/s)
                "Tot-Tok/s", # Total-Token-throughput(tok/s)
                "TTFT",      # Median-TTFT(ms)
                "TPOT"       # Median-TPOT(ms)
            ])

        # 写入日志文件标题
        with open(benchmark_log, "w") as f:
            f.write(format_benchmark_header() + "\n")

        # 运行基准测试
        config = {
            "model": MODEL_PATH,
            "port": PORT,
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
            "num_prompts": 0,
            "concurrency": None
        }

        results_list = []  # 用于存储所有基准测试结果

        for concurrency in [4, 8, 16, 32, 64, 128, 256]:
            print(f"\n{'='*40}")
            print(f"Starting concurrency test: {concurrency}")
            print(f"{'='*40}")
            
            # 动态生成一个随机种子
            random_seed = random.randint(10000, 99999)
            
            # 更新配置
            config['concurrency'] = concurrency
            config['num_prompts'] = concurrency * 5  # 设置 num-prompts 为 max-concurrency 的 5 倍
            config['seed'] = random_seed  # 将随机种子添加到配置中
            
            print(f"Using random seed: {random_seed}")
            print(f"Setting num-prompts to {config['num_prompts']} (5x max-concurrency)")
            
            # 执行基准测试
            results = run_benchmark(pm, config, benchmark_log)
            results_list.append(results)  # 收集结果

            # 写入日志文件数据行
            with open(benchmark_log, "a") as f:
                f.write(format_benchmark_row(results) + "\n")

            # 写入 CSV 文件数据行
            with open(csv_file, "a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    results.get("MaxConcurrency", "N/A"),
                    results.get("SuccessRequests", "N/A"),
                    results.get("RequestThroughput", "N/A"),
                    results.get("OutputTokenThroughput", "N/A"),
                    results.get("TotalTokenThroughput", "N/A"),
                    results.get("TTFT", "N/A"),
                    results.get("TPOT", "N/A")
                ])

        # 基准测试完成后生成图示
        plot_results(results_list, plot_file)

    except Exception as e:
        print(f"\n[Error] {str(e)}")
        sys.exit(1)
    finally:
        # 确保在任何情况下都调用清理方法
        pm.cleanup()

        # 检查是否还有残留的 Python 进程
        print("\n[Debug] Checking for remaining Python processes...")
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    if 'cmdline' in proc.info and proc.info['cmdline']:
                        cmdline = ' '.join(proc.info['cmdline'])
                        print(f"Remaining process: PID={proc.info['pid']}, CMD={cmdline}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

if __name__ == "__main__":
    main()
