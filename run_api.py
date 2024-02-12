import os
import subprocess
import socket

# todo- re-add these to be directly run via main.py, instead of running ```node api/api.js``` in the terminal

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    result = subprocess.run(['lsof', '-i', f':{port}'], stdout=subprocess.PIPE, text=True)
    for line in result.stdout.splitlines():
        if "LISTEN" in line and 'node' in line:
            pid = int(line.split()[1])
            os.kill(pid, 9)  # 9 is the SIGKILL signal

def start_node_app(port, script_path):
    if is_port_in_use(port):
        print(f"Port {port} is in use, killing process...")
        kill_process_on_port(port)
    print(f"Starting Node.js application on port {port}...")
    # subprocess.run(['node', script_path], stdout=subprocess.PIPE, text=True)
    process = subprocess.Popen(['node', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

