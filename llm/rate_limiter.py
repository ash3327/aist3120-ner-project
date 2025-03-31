import time

class RateLimiter:
    def __init__(self, rate=1.0):
        """
        Initialize a rate limiter with specified calls per second.

        Args:
            rate (float): Maximum number of calls per second (default: 1.0)
        """
        self.rate = rate
        self.last_call_time = 0
        self.min_interval = 1.0 / rate if rate > 0 else 0

    def wait(self):
        """
        Wait if necessary to maintain the specified rate.
        """
        if self.rate <= 0:  # No rate limiting if rate is 0 or negative
            return
            
        current_time = time.time()
        elapsed_time = current_time - self.last_call_time
        
        if elapsed_time < self.min_interval:
            # Wait for the remaining time to maintain the rate
            sleep_time = self.min_interval - elapsed_time
            time.sleep(sleep_time)
            
        self.last_call_time = time.time()