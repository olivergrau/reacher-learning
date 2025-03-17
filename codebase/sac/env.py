import time
import subprocess
from unityagents import UnityEnvironment
import gc

class BootstrappedEnvironment:
    def __init__(self, exe_path, worker_id=0, use_graphics=False, preprocess_fn=None, max_retries=5, retry_delay=2):
        """
        Initialize the Unity environment wrapper.
        """
        self.exe_path = exe_path
        self.worker_id = worker_id
        self.use_graphics = use_graphics
        self.preprocess_fn = preprocess_fn
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._closed = False

        try:
            self.env = UnityEnvironment(file_name=self.exe_path, no_graphics=not self.use_graphics, worker_id=self.worker_id)
        except Exception as e:
            print(f"[EnvWrapper] Failed to initialize UnityEnvironment (worker_id: {self.worker_id}): {e}")
            self.env = None
            raise

        self.brain_name = self.env.brain_names[0]

    def reset(self, train_mode=True):
        attempt = 0
        while attempt < self.max_retries:
            try:
                env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
                raw_state = env_info.vector_observations
                
                if self.preprocess_fn:
                    return self.preprocess_fn(raw_state)
                
                return raw_state
            except Exception as e:
                print(f"[EnvWrapper] Error during reset: {e}. Attempt {attempt + 1}/{self.max_retries}. Retrying in {self.retry_delay} seconds...")
                attempt += 1
                time.sleep(self.retry_delay)
        
        self.close()
        raise RuntimeError("Failed to reset Unity environment after maximum retries.")

    def step(self, actions):
        try:
            env_info = self.env.step(actions)[self.brain_name]
            raw_next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            
            if self.preprocess_fn:
                next_state = self.preprocess_fn(raw_next_state)
            else:
                next_state = raw_next_state
            
            return next_state, reward, done
        except Exception as e:
            print(f"[EnvWrapper] Error during step: {e}. Attempting to close environment.")
            self.close()
            raise

    def close(self):
        """
        Close the Unity environment if not already closed, and then attempt to wipe lingering processes.
        """
        if self.env is not None and not self._closed:
            try:
                self.env.close()
                self._closed = True
                self.env = None
                                
                gc.collect()

                print("[EnvWrapper] Waiting 5 seconds for closing the environment...")
                time.sleep(5)
                
                self._wipe_unity_processes()
                print("[EnvWrapper] Unity environment closed and processes wiped successfully.")
            except Exception as e:
                print(f"[EnvWrapper] Error while closing Unity environment: {e}")

    def _wipe_unity_processes(self):
        """
        Attempts to kill lingering Unity/Reacher processes from the OS.
        This method uses the 'pkill' command which is Linux-specific.
        """
        try:
            # Using pkill with the exe_path should kill any process that was started with that executable.
            subprocess.call(["pkill", "-f", self.exe_path])
            time.sleep(5)
            print("[EnvWrapper] Successfully wiped Unity processes from OS.")
        except Exception as e:
            print(f"[EnvWrapper] Error while wiping Unity processes: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()
