# Conda

## 基础命令

### 创建环境
```
conda create -n [env_name] python=3.9
```

### 进入环境
```
conda activate [env_name]
```

### 删除环境
```
conda detictive [env_name]
```

### 临时换源
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple [package_name]
```

### 永久换源
```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

