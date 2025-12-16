"""调用 API 清理卡住的训练任务"""
import requests

# 假设后端运行在默认端口
backend_url = "http://localhost:8000"

try:
    response = requests.post(f"{backend_url}/training/jobs/cleanup-stuck")
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 成功清理 {result['cleaned']} 个卡住的任务")
        print(f"  消息: {result['message']}")
    else:
        print(f"✗ 清理失败: {response.status_code}")
        print(f"  {response.text}")
except requests.exceptions.ConnectionError:
    print("✗ 无法连接到后端服务")
    print("  请确保后端服务正在运行 (通常在 http://localhost:8000)")
except Exception as e:
    print(f"✗ 错误: {e}")
