import time
from codebase.replay.replay_buffer import ReplayWrapper, Transition  # Replay buffer implementation

class ReplayProxy:
    def __init__(self, conn, max_retries=5, retry_delay=0.01):
        self.conn = conn
        self.cache = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def feed(self, exp):
        self.conn.send([ReplayWrapper.FEED, exp])

    def sample(self):
        retries = 0
        while retries < self.max_retries:
            self.conn.send([ReplayWrapper.SAMPLE, None])
            cache_id, data = self.conn.recv()
            if data is not None:
                # Update cache with new data and return the transition.
                self.cache = data
                return Transition(*data[cache_id])
            else:
                if self.cache is not None:
                    return Transition(*self.cache[cache_id])
                
                print("[ReplayProxy]: Insufficient data in sample, waiting for more data...")

                # Otherwise, wait a bit and retry.
                retries += 1
                time.sleep(self.retry_delay)
        raise RuntimeError(f"No cached data available for sampling after {self.max_retries} retries.")

# (Add other methods if needed, e.g., update_priorities)