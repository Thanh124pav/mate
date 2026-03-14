# Hướng dẫn sử dụng Docker cho MATE

## 1. Build Docker Image

### Cách 1: Sử dụng script tự động
```bash
chmod +x docker-build-push.sh
./docker-build-push.sh -u <your-dockerhub-username> -t latest
```

### Cách 2: Build thủ công
```bash
docker build -t <your-dockerhub-username>/mate:latest .
```

## 2. Test Image locally

```bash
# Test với CPU
docker run -it --rm <your-dockerhub-username>/mate:latest

# Test với GPU
docker run -it --rm --gpus all <your-dockerhub-username>/mate:latest

# Mount thư mục hiện tại vào container
docker run -it --rm --gpus all \
  -v $(pwd):/workspace/mate \
  <your-dockerhub-username>/mate:latest
```

## 3. Push lên Docker Hub

### Bước 1: Login vào Docker Hub
```bash
docker login
```
Nhập username và password của bạn.

### Bước 2: Push image
```bash
docker push <your-dockerhub-username>/mate:latest
```

### Bước 3 (Optional): Push thêm tag phiên bản
```bash
docker tag <your-dockerhub-username>/mate:latest <your-dockerhub-username>/mate:v1.0
docker push <your-dockerhub-username>/mate:v1.0
```

## 4. Hướng dẫn người khác sử dụng

### Pull image từ Docker Hub
```bash
docker pull <your-dockerhub-username>/mate:latest
```

### Chạy container
```bash
# Chạy với GPU support
docker run -it --gpus all \
  -v /path/to/your/data:/workspace/data \
  <your-dockerhub-username>/mate:latest

# Chạy training script
docker run -it --gpus all \
  -v $(pwd)/results:/workspace/results \
  <your-dockerhub-username>/mate:latest \
  python examples/ippo/camera_ippo.py
```

### Chạy với Weights & Biases
```bash
docker run -it --gpus all \
  -e WANDB_API_KEY=<your-wandb-key> \
  <your-dockerhub-username>/mate:latest
```

## 5. Docker Compose (Optional)

Tạo file `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mate:
    image: <your-dockerhub-username>/mate:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - WANDB_API_KEY=${WANDB_API_KEY}
    volumes:
      - ./results:/workspace/results
      - ./data:/workspace/data
    command: python examples/ippo/camera_ippo.py
```

Chạy với:
```bash
docker-compose up
```

## 6. Push lên các registry khác

### GitHub Container Registry
```bash
# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag và push
docker tag mate:latest ghcr.io/<username>/mate:latest
docker push ghcr.io/<username>/mate:latest
```

### Google Container Registry
```bash
# Tag và push
docker tag mate:latest gcr.io/<project-id>/mate:latest
docker push gcr.io/<project-id>/mate:latest
```

## 7. Tips

### Giảm kích thước image
- Sử dụng multi-stage build
- Xóa cache và temporary files
- Sử dụng `.dockerignore`

### Debug container
```bash
# Vào container đang chạy
docker exec -it <container-id> /bin/bash

# Xem logs
docker logs <container-id>

# Inspect container
docker inspect <container-id>
```

### Lưu image thành file
```bash
# Save
docker save -o mate.tar <your-dockerhub-username>/mate:latest

# Load
docker load -i mate.tar
```
