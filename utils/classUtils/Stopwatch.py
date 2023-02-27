import time
import datetime

class Stopwatch:
    def __init__(self):
        self.start = time.perf_counter()

    def __str__(self):
        now = time.perf_counter() - self.start 
        timestamp = str(datetime.timedelta(seconds=now))
        timestamp = timestamp.split('.')[0]
        return timestamp