import os
import subprocess
import socket
import time
import requests


def get_free_port(start_port=8000):
    port = start_port
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', port))
            s.close()
            return port
        except OSError:
            port += 1
            
            
def check_if_vllm_running(port: int):
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        if response.status_code == 200:
            return True
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False
            
            
def wait_for_vllm(port: int):
    while True:
        if check_if_vllm_running(port):
            print(f"vllm server running on port {port}")
            return
        time.sleep(5)
            

class VLLMManager:
    def __init__(self, model_name: str, num_gpus: int = 1, port: int = 8000, vllm_process_outdir: str = None, seed: int = 42):
        self.model_name = model_name
        
        # checks if vllm is already running
        self.process = None
        if check_if_vllm_running(port):
            print(f"vllm server already running on port {port}")
            self.port = port
            self.vllm_url = f"http://localhost:{self.port}"
            return
        
        self.port = get_free_port(port)
        command = ['vllm', 'serve', model_name, '--tensor-parallel-size', str(num_gpus), '--port', str(self.port), '--seed', str(seed)]
        
        if vllm_process_outdir:
            os.makedirs(vllm_process_outdir, exist_ok=True)
            process_outfile = open(os.path.join(vllm_process_outdir, f"vllm_log.txt"), "w")
            self.process = subprocess.Popen(command, stdout=process_outfile, stderr=process_outfile)  # Start vllm serve in a subprocess
        else:
            self.process = subprocess.Popen(command)  # Start vllm serve in a subprocess
        
        wait_for_vllm(self.port)
        
        self.vllm_url = f"http://localhost:{self.port}"

    def __del__(self):
        if self.process:  # Ensure the subprocess is killed
            self.process.terminate()  # Terminate the subprocess
            self.process.wait()  # Wait for the subprocess to finish
            

if __name__ == "__main__":
    vllm = VLLMManager("Qwen/Qwen2.5-7B-Instruct", num_gpus=4, vllm_process_outdir="logs")
