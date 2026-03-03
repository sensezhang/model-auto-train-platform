# YOLO/RF-DETR 训练标注平台 - 部署指南

---

## 镜像说明

镜像托管在公司内部仓库 `hub.ikingtec.com/ai/`：

| 镜像 | 说明 |
|------|------|
| `hub.ikingtec.com/ai/yolo-backend:v1` | FastAPI 后端 + PyTorch + CUDA 12.4 |
| `hub.ikingtec.com/ai/yolo-frontend:v1` | React 前端 + Nginx |

**镜像包含：** OS + CUDA 运行时 + Python 依赖 + 代码
**镜像不含（全部挂载自数据盘）：** 图片、标注、训练权重、预训练权重缓存

---

## 本机：构建并推送镜像

```powershell
# 首次登录（只需一次）
docker login hub.ikingtec.com

# 构建并推送（在项目根目录执行）
.\build-push.ps1

# 指定版本号
.\build-push.ps1 -Version v2
```

---

## 服务器：首次部署

### 第一步：挂载数据盘

```bash
# 查看磁盘设备名
lsblk

# 挂载（/dev/sdb 换成实际设备名）
sudo mkdir -p /mnt/data
sudo mount /dev/sdb /mnt/data

# 设置开机自动挂载
sudo blkid /dev/sdb   # 记下 UUID
echo 'UUID=你的UUID /mnt/data ext4 defaults 0 2' | sudo tee -a /etc/fstab
sudo mount -a
```

### 第二步：创建数据目录

```bash
sudo mkdir -p /mnt/data/{datasets,datasets_coco,models,output,logs,pretrained,.torch,.cache/huggingface,.ultralytics}
sudo chmod -R 777 /mnt/data
```

### 第三步：上传预训练权重

在本机 PowerShell 执行：

```powershell
scp E:\PycharmProjects\yolo-train-test\rf-detr-medium.pth user@服务器IP:/mnt/data/pretrained/
scp E:\PycharmProjects\yolo-train-test\rf-detr-small.pth  user@服务器IP:/mnt/data/pretrained/
```

### 第四步：上传部署配置

在本机 PowerShell 执行：

```powershell
scp E:\PycharmProjects\yolo-train-test\deploy\docker-compose.yml user@服务器IP:/opt/yolo/
```

### 第五步：创建 .env 文件

```bash
mkdir -p /opt/yolo
vim /opt/yolo/.env
```

`.env` 内容：

```env
# 数据库
APP_DB_URL=mysql+pymysql://user:password@host:3306/dbname

# OSS
OSS_ENDPOINT=http://10.x.x.x:8009
OSS_BUCKET=your-bucket
OSS_ACCESS_KEY=your-key
OSS_SECRET_KEY=your-secret

# 自动标注 VLM
AUTOLABEL_BASE_URL=http://your-vlm-server/v1
AUTOLABEL_MODEL=glm-4v-9b
AUTOLABEL_API_KEY=your-api-key
```

### 第六步：拉取镜像并启动

```bash
cd /opt/yolo
docker compose pull
docker compose up -d
```

### 验证

```bash
# 查看容器状态
docker compose ps

# 查看后端日志
docker logs -f yolo-backend

# 测试接口
curl http://localhost/api/health
```

访问 `http://服务器IP` 能打开页面即部署成功。

---

## 日常运维

```bash
# 查看状态
docker compose ps

# 查看后端日志
docker logs -f yolo-backend

# 重启
docker compose restart

# 停止
docker compose down
```

---

## 版本更新

```powershell
# 本机：推新版本镜像
.\build-push.ps1 -Version v2
```

然后修改服务器 `/opt/yolo/docker-compose.yml` 中的镜像版本号（`v1` → `v2`），再执行：

```bash
cd /opt/yolo
docker compose pull
docker compose up -d
```

---

## 训练日志查看

```bash
# 子进程日志（含详细报错）
tail -f /mnt/data/models/training_logs/rfdetr_process_<job_id>.log

# 训练指标日志
cat /mnt/data/models/<project_id>/<job_id>/logs.txt
```

---

## 端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| 前端 + Nginx 反代 | 80 | 对外唯一入口 |
| 后端 API | 8000 | 容器内部，不对外暴露 |

---

## 常见问题

### GPU 在容器内不可用

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
# 失败则重装
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 端口 80 被占用

修改 `docker-compose.yml` 中 `"80:80"` 为 `"8080:80"`，再重启。
