## Git笔记

### 1.安装Git

Debian/Ubuntu Linux

> 确定是否已安装：git
>
> 安装命令：sudo apt-get install git
>
> 旧版本安装命令：sudo apt-get install git-core 

Mac OS X

>方法一：安装homebrew，然后通过homebrew安装Git，具体方法请参考homebrew的文档： http://brew.sh/ 
>
>方法二：直接从AppStore安装Xcode，Xcode集成了Git，不过默认没有安装，你需要运⾏ Xcode，选择菜单“Xcode”->“Preferences”，在弹出窗⼝中找到“Downloads”，选择“Command Line Tools”，点“Install”就可
>以完成安装了。 

Windows：

> 官网下载git安装即可，配置用户名及邮箱地址
>
> git config --global user.name "Your Name" 
>
> git config --global user.email "email@example.com" 

### 2.Windows认证

> 1.ssh-keygen -t rsa -C "username"  (username为你git上的用户名)
>
> 2.找到bash反馈路径文件夹，使用记事本打开id_rsa.pub并复制全部内容
>
> 3.进入github的setting中的SSH里复制到key生成ssh key
>
> 4.在bash.exe中输入ssh -T git@github.com

### 3.Windows上传至Github库

> 1.首先在Github上创建repository
>
> 2.本地创建文件夹并进入Git Bash
>
> 3.初始化：git init
>
> 4.添加：git add README.md
>
> 5.提交：git commit -m "commit" （其中"commit"为该文件备注）
>
> 6.连接：git remote add origin git@github.com:jm199504/NNN.git（其中NNN为repository名）
>
> 7.上传：git push -u origin master

### 4.其他命令

> 4.1 查看本地与Github repository目录差异：git status 
>
> 例如：
>
> 图1
>
> 解释：本地新建了Network-Optimizer文件夹，而Github repository中不存在该文件夹，建议添加&提交
>
> 4.2 查看文件差异：git diff