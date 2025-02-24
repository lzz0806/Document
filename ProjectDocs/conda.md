# Conda

## 基础命令
```bash
1. 创建环境
conda create -n [env_name] python=3.9
2. 进入环境
conda create -n [env_name] python=3.9
3. 激活环境
conda activate [env_name]
4. 删除环境
conda remove -n [env_name] --all
5. pip 临时换源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple [package_name]
6. 永久换源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
7. 查看 conda config
conda config --show
8. 设置 conda config
conda config --set auto_activate_base True
9. 打包 conda 环境
conda-pack -n isaaclab -o isaaclab.tar.gz
```
