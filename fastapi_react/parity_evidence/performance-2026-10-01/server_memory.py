import json, psutil
ports = {8502: 'streamlit', 8000: 'fastapi', 5174: 'vite_dev'}
result = {'available_system_mb': psutil.virtual_memory().available / 1048576}
connections = psutil.net_connections(kind='tcp')
for port, name in ports.items():
    ids = {c.pid for c in connections if c.pid and c.status == 'LISTEN' and c.laddr.port == port}
    processes = {}
    for pid in ids:
        process = psutil.Process(pid)
        while True:
            processes[process.pid] = process
            parent = process.parent()
            if not parent or parent.name().lower() not in {'python.exe', 'pythonw.exe', 'node.exe'}:
                break
            process = parent
        for child in process.children(recursive=True):
            processes[child.pid] = child
    rss = private = cpu = 0
    for process in processes.values():
        try:
            memory = process.memory_full_info()
            rss += memory.rss
            private += getattr(memory, 'uss', 0)
            times = process.cpu_times()
            cpu += times.user + times.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    result[name] = {'pids': sorted(processes), 'process_count': len(processes), 'rss_mb': rss / 1048576, 'private_mb': private / 1048576, 'cpu_seconds': cpu}
print(json.dumps(result))
