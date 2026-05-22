import torch 
import os 
import time 
from contextlib import contextmanager


@contextmanager
def timer(name): 
    if os.environ.get("DEBUG_TIMERS", "0") != "1":
        yield
        return
    t0 = time.time()
    yield
    if torch.cuda.is_available(): 
        torch.cuda.synchronize()
    print(f"{name} took {time.time() - t0:.2f}s")
