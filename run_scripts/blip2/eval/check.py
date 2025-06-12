import sys

# 打印Python解释器路径，用于诊断
print(f"--- Python Executable: {sys.executable} ---")

# 下面是诊断核心代码
try:
    from lavis.common.registry import registry
    # 这一行是关键，它会触发所有内置任务的注册流程
    from lavis.tasks import *

    print("\n" + "="*40)
    print("LAVIS Successfully Initialized. Registered Tasks:")
    # 打印出所有已注册的任务名称列表
    print(sorted(list(registry.list_tasks())))
    print("="*40 + "\n")

except Exception as e:
    print(f"An error occurred during LAVIS initialization: {e}")