import time


def perf_sleep(seconds):
    clk = time.perf_counter()
    if seconds > 5e-3:
        time.sleep(seconds - 5e-3)
    while (time.perf_counter() - clk) < seconds:
        continue
