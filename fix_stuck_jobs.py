"""修复卡住的训练任务"""
import sqlite3
from datetime import datetime

# 连接数据库
conn = sqlite3.connect('app.db')
cursor = conn.cursor()

# 查找所有 pending 或 running 状态的任务
cursor.execute("SELECT id, projectId, status, startedAt FROM TrainingJob WHERE status IN ('pending', 'running')")
stuck_jobs = cursor.fetchall()

print(f"找到 {len(stuck_jobs)} 个待处理/运行中的训练任务:")
for job in stuck_jobs:
    print(f"  Job ID: {job[0]}, Project ID: {job[1]}, Status: {job[2]}, Started: {job[3]}")

if stuck_jobs:
    choice = input("\n是否将这些任务标记为失败? (y/n): ").strip().lower()
    if choice == 'y':
        # 将所有卡住的任务标记为失败
        cursor.execute("""
            UPDATE TrainingJob
            SET status = 'failed',
                finishedAt = ?
            WHERE status IN ('pending', 'running')
        """, (datetime.utcnow().isoformat(),))
        conn.commit()
        print(f"已将 {cursor.rowcount} 个任务标记为失败")
    else:
        print("未做任何修改")
else:
    print("没有卡住的任务")

conn.close()
