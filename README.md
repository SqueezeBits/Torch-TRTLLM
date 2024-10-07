# Ditto

## Setting up Docker container
### 1. Build a docker image
```
docker build -f docker/Dockerfile -t ditto:ubuntu24.04 .
```

### 2. Run a container
```
docker run --rm -it --gpus all -v `pwd`:/workspace/ditto -v /home/hdd/huggingface_models:/data ditto:ubuntu24.04 bash
```
