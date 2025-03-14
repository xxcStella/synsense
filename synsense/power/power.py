# XXC, 2025-02-18
# power measurement on CPU and GPU (estimated)

import time
import subprocess

"""
User Functions:
- get_average_cpu_power(SUDO_PASSWORD, interval=1.0): 
    measure CPU power over "interval" period.
- get_gpu_power():
    measure current (instantaneous) power.
- get_average_gpu_power(interval=5, sampling_rate=0.5):
    measure GPU power over "interval" period with sampling rate of 0.5s.
- get_average_cpu_gpu_power(SUDO_PASSWORD, interval=5, sampling_rate=0.5):
    measure average CPU and GPU power separately over "interval" period with 
    sampling rate of 0.5s.
"""

def _run_sudo_command(command, SUDO_PASSWORD):
    """执行需要 sudo 权限的命令，并自动输入密码"""
    try:
        result = subprocess.run(
            f"echo {SUDO_PASSWORD} | sudo -S {command}",
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"
    
def _read_cpu_energy(SUDO_PASSWORD):
    """读取 CPU 总能耗(uJ),需要 sudo 权限"""
    energy_uj = _run_sudo_command("cat /sys/class/powercap/intel-rapl:0/energy_uj", SUDO_PASSWORD)
    return int(energy_uj) if energy_uj.isdigit() else None

def get_average_cpu_power(SUDO_PASSWORD, interval=1.0):
    """
    测量 `interval` 秒内的 CPU 平均功耗(W)
    Params:
        SUDO_PASSWORD: substitue it with your own Linux sudo password.
        interval (float): interval time, default=1 second.
    """
    energy_start = _read_cpu_energy(SUDO_PASSWORD)
    if energy_start is None:
        return "Error: 无法读取 CPU 能耗数据"

    time.sleep(interval)

    energy_end = _read_cpu_energy(SUDO_PASSWORD)
    if energy_end is None:
        return "Error: 无法读取 CPU 能耗数据"

    energy_diff = (energy_end - energy_start) / 1_000_000  # 转换为焦耳（J）
    power = energy_diff / interval  # 计算功率（W）
    return power

def get_gpu_power():
    """获取 GPU 实时功耗(W)"""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        power_values = output.strip().split("\n")  # 按行分割
        return float(power_values[0])  # 只取第一个 GPU 的功耗
    except Exception as e:
        return f"Error: {e}"
    
def get_average_gpu_power(interval=5, sampling_rate=0.5):
    """
    计算 `interval` 秒内的 GPU 平均功耗。
    sampling_rate 控制采样频率，例如 0.5 秒采样一次。
    """
    power_readings = []
    start_time = time.time()

    while (time.time() - start_time) < interval:
        power = get_gpu_power()
        if power is not None:
            power_readings.append(power)
        time.sleep(sampling_rate)  # 采样间隔

    if not power_readings:
        return "Error: 无法读取 GPU 功耗"

    avg_power = sum(power_readings) / len(power_readings)
    return avg_power

def get_average_cpu_gpu_power(SUDO_PASSWORD, interval=5, sampling_rate=0.5):
    """
    计算 `interval` 秒内 CPU 和 GPU 的平均功耗。
    - `sampling_rate` 控制采样频率，例如 0.5 秒采样一次
    """
    gpu_power_readings = []
    start_time = time.time()

    # 记录 CPU 初始能耗
    cpu_energy_start = _read_cpu_energy(SUDO_PASSWORD)
    if cpu_energy_start is None:
        return "Error: 无法读取 CPU 能耗数据"

    while (time.time() - start_time) < interval:
        # 获取 GPU 实时功耗
        gpu_power = get_gpu_power()
        if gpu_power is not None:
            gpu_power_readings.append(gpu_power)

        time.sleep(sampling_rate)

    # 记录 CPU 结束能耗
    cpu_energy_end = _read_cpu_energy(SUDO_PASSWORD)
    if cpu_energy_end is None:
        return "Error: 无法读取 CPU 能耗数据"

    # 计算 CPU 平均功耗
    energy_diff = (cpu_energy_end - cpu_energy_start) / 1_000_000  # 转换为焦耳（J）
    avg_cpu_power = energy_diff / interval  # 计算 CPU 平均功率（W）

    # 计算 GPU 平均功耗
    avg_gpu_power = sum(gpu_power_readings) / len(gpu_power_readings) if gpu_power_readings else 0

    return avg_cpu_power, avg_gpu_power
