import subprocess
import os
import signal
import platform

processes = []

def run_backend():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(__file__)
    with open(os.devnull, "w") as fnull:
        process = subprocess.Popen(
            ["python", "-m", "uvicorn", "code.deployment.api.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"],
            cwd=os.path.dirname(__file__),
            env=env,
            stdout=fnull,
            stderr=fnull
        )
    processes.append(process)
    print("Backend is running at: http://127.0.0.1:8000")

def run_frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), "code", "deployment", "app")
    with open(os.devnull, "w") as fnull:
        process = subprocess.Popen(
            ["python", "-m", "streamlit", "run", "main.py"],
            cwd=frontend_path,
            stdout=fnull,
            stderr=fnull
        )
    processes.append(process)
    print("Frontend is running at: http://127.0.0.1:8501")

def terminate_processes():
    print("\nTerminating all processes...")
    for process in processes:
        if platform.system() == "Windows":
            process.terminate()
        else:
            process.send_signal(signal.SIGINT)
        process.wait()
    print("All processes terminated.")

if __name__ == "__main__":
    try:
        run_backend()
        run_frontend()

        print("\nBackend and Frontend are running...")
        print("You can access the backend at: http://127.0.0.1:8000")
        print("You can access the frontend at: http://127.0.0.1:8501")
        
        input("\nPress Enter to terminate both.\n")
    finally:
        terminate_processes()