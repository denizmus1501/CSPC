import time
from decay import simulate, simulate_loop

N0 = 200000
lam = 0.1
dt = 0.05
steps = 200
seed = 0

#timing the pure-python version
start = time.perf_counter()
simulate_loop(N0, lam, dt, steps, seed)
loop_time = time.perf_counter() - start 

#timing the numpy version
start = time.perf_counter()
simulate(N0, lam, dt, steps, seed)
numpy_time = time.perf_counter() - start 

#calculate spped-up
speed_up = loop_time / numpy_time

print(f"pure python loop: {loop_time:.6f} seconds")
print(f"numpy version: {numpy_time:.6f} seconds")
print(f"numpy is {speed_up:.2f}x faster")
