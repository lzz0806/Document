# Linux 基础
## Linux 基础指令
## 问题总结
### Ubuntu 系统安装无法进入安装界面
```bash
# 1.在 “try and install Ubuntu” 处按 e 进入编辑模式
# 2.将 quiet splash --- 中的 “---” 改为 “nomodeset”
# 3.按 F10 保存后，等待重启即可进入 Ubuntu 安装界面
```
### Ubuntu 系统安装后重启黑屏
```bash
# 1.重启电脑，在启动开始时按住 “shift”，进入 advanced options for Ubuntu 模式
# 2.选择 recover mode，进入 root shell
# 3.选择 root 确认后输入帐号密码
# 4.sudo vi /etc/default/grub
# 5.找到“quiet splash”，改为 “quiet splash nomodeset”，保存退出
# 6.先 sudo update-grub 再执行 reboot 重启电脑
```