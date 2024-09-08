import tkinter as tk
import threading
import time
import gc
import os


def worker_thread():
    while running:
        time.sleep(1)
        # 模拟一些工作
        print("Thread running...")


def on_closing():
    global running
    running = False

    # 等待线程终止
    thread.join()

    # 销毁Tkinter窗口
    root.destroy()

    # 强制进行垃圾回收
    gc.collect()

    # 退出Python解释器
    os._exit(0)


# 创建Tkinter窗口
root = tk.Tk()
root.protocol("WM_DELETE_WINDOW", on_closing)

# 启动工作线程
running = True
thread = threading.Thread(target=worker_thread)
thread.start()

# 运行Tkinter主循环
root.mainloop()
