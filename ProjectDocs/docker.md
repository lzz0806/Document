# Docker

## 基础参数
- -i：交互式操作。
- -t：终端 （-it 同时使用可以让 docker 运行的容器实现"对话"的能力）
- -d：参数默认不会进入容器，想要进入容器需要使用指令 docker exec(后边会讲到)
- -p：端口映射主要用于通信

## 常见命令

### 从远程仓库拉取镜像
```
docker pull [域名：版本]
```
### 查看正在运行的容器
```
docker ps          # 查看正在运行的容器
docker ps -a       # 显示隐藏容器
docker ps | wc -l  # 查看总的容器数，包括 exited、created 状态的，数量需要-1
```
### 查看本地下载好的镜像
```
docker images
```

### 删除镜像
```
docker rmi [image name or image ID]
docker rmi [image name or image ID] --f  # 强力删除
```
### 停止容器
```
docker stop [容器名]
```
### 删除容器
```
docker rm [容器名]
```
### 容器与宿主机之间相互复制文件
```
docker cp [容器内部文件路径] [目标位置]
docker cp [容器外部文件路径] [容器内目标位置]
```
### 运行镜像生成容器
```
docker run --name [容器名] -p [映射到的port]:[需要映射的port] -itd [镜像名] /bin/bash run.sh  # -itd表示启动，但不进入容器
```

### 标签
打标签，或者换标签，注意如果需要删除之前的 tag，需要使用 rmi 删除 tag，不要删除 imageID
```
docker tag [旧的仓库名：tag][新的仓库名：tag]
```


### build
```
```

### commit
```
```
### 导出镜像
```
docker save -o [image ID] xxx.tar
```

### 加载镜像
```
docker load < xxx.tar
```


### Dockerfile

## 用法案例

