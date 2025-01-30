import os
import platform
import multiprocessing
import psutil
import sys

def get_cpu_info():
    physical_cores = psutil.cpu_count(logical=False)
    total_cores = psutil.cpu_count(logical=True)
    try:
        cpu_freq = psutil.cpu_freq()
    except FileNotFoundError:
        cpu_freq = None
    return physical_cores, total_cores, cpu_freq

def get_system_info():
    uname = platform.uname()
    system_info = {
        "System": uname.system,
        "Node Name": uname.node,
        "Release": uname.release,
        "Version": uname.version,
        "Machine": uname.machine,
        "Processor": uname.processor
    }
    return system_info

def get_process_limits():
    try:
        import resource
        soft_proc_limit, hard_proc_limit = resource.getrlimit(resource.RLIMIT_NPROC)
        return soft_proc_limit, hard_proc_limit
    except ImportError:
        # resource module is not available on Windows
        return "N/A", "N/A"

def get_thread_limits():
    # Thread limits are not directly accessible via the resource module
    # Depending on the OS, you might need a different approach or library
    return "N/A", "N/A"

def main():
    print("===== System Information =====\n")
    
    # CPU Information
    physical_cores, total_cores, cpu_freq = get_cpu_info()
    print("---- CPU Information ----")
    print(f"Physical Cores: {physical_cores}")
    print(f"Total Cores (Logical): {total_cores}")
    if cpu_freq:
        print(f"Max Frequency: {cpu_freq.max:.2f}Mhz")
        print(f"Min Frequency: {cpu_freq.min:.2f}Mhz")
        print(f"Current Frequency: {cpu_freq.current:.2f}Mhz")
    else:
        print("CPU Frequency information not available.")
    print()

    # System Information
    sys_info = get_system_info()
    print("---- System Information ----")
    for key, value in sys_info.items():
        print(f"{key}: {value}")
    print()

    # Process Limits
    soft_proc_limit, hard_proc_limit = get_process_limits()
    print("---- Process Limits ----")
    print(f"Soft Process Limit: {soft_proc_limit}")
    print(f"Hard Process Limit: {hard_proc_limit}")
    print()

    # Thread Limits
    soft_thread_limit, hard_thread_limit = get_thread_limits()
    print("---- Thread Limits ----")
    print(f"Soft Thread Limit: {soft_thread_limit}")
    print(f"Hard Thread Limit: {hard_thread_limit}")
    print()

    # Maximum Number of Processes and Threads
    print("---- Maximum Number of Processes and Threads ----")
    print(f"Maximum Processes: {'Unlimited' if hard_proc_limit == -1 else hard_proc_limit}")
    print(f"Maximum Threads: {'Unlimited' if hard_thread_limit == -1 else hard_thread_limit}")
    print()

if __name__ == "__main__":
    main()