# Docker

## 2.1 基础参数
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

### docker非root用户不用sudo执行docker命令

1. 创建名为docker的组，如果之前已经有该组就会报错，可以忽略这个错误：
```shell
sudo groupadd docker
```

2. 将当前用户加入组docker：
```bash
sudo gpasswd -a [user_name] docker
```

3. 重启docker服务(生产环境请慎用)：
```bash
sudo systemctl restart docker
```

4. 添加访问和执行权限：
```shell
sudo chmod a+rw /var/run/docker.sock
```

5. 操作完毕，验证一下，现在可以不用带sudo了：

### 
1. [Ubuntu安装完docker引擎后，在创建容器的时候指定 --gpus all，出现报错如下](https://www.cnblogs.com/booturbo/p/16318627.html)
2. [docker mount](https://zhuanlan.zhihu.com/p/667272282)
3. [docker镜像源](https://blog.csdn.net/llc580231/article/details/139979603)