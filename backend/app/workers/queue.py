"""
简易任务队列占位：
- MVP阶段可用内存队列或后台任务管理AI自动标注(并发=2)与训练(单并发)
"""

class SimpleQueue:
    def __init__(self):
        self.jobs = []

    def enqueue(self, job):
        self.jobs.append(job)

    def pop(self):
        return self.jobs.pop(0) if self.jobs else None


auto_label_queue = SimpleQueue()
train_queue = SimpleQueue()

