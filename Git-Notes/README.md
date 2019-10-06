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

> #### 4.1 查看本地与Github repository目录差异：git status 
>
> 例如：
>
> <img src="https://github.com/jm199504/Other-Notes/blob/master/Git-Notes/images/1.png" width="500">
>
> 解释：本地新建了Network-Optimizer文件夹，而Github repository中不存在该文件夹，建议添加&提交
>
> #### 4.2 查看文件内具体差异：git diff
>
> 例如：
>
> <img src="https://github.com/jm199504/Other-Notes/blob/master/Git-Notes/images/2.png" width="500">
>
> 解释：本地修改了Git-Notes/README.md内容，其中红色表示删除行内容，绿色表示添加行内容
>
> #### 4.3 查看日志
>
> 多行显示一次更新：git log（提交历史）
>
> 单行显示一次更新：git log --pretty=oneline
>
> #### 4.4 回滚至历史版本
>
> git reset --hard HEAD^ 
>
> 注意：HEAD^表示前次版本；HEAD^^表示前前版本；HEAD~N表示前N次版本
>
> 使用git reflog 查看每次提交的commit_id（命令历史）
>
> git reset --hard commit_id

> #### 4.5 删除Github上某一个文件夹
> git pull origin master      拉取远程仓库文件夹及文件
>
> dir                         查看有文件夹及文件
>
> git rm -r --cached a        删除a文件夹
>
> git commit -m 'del a'       提交及备注


其他资料：

 <img src="https://github.com/jm199504/Other-Notes/blob/master/Git-Notes/images/3.png">


 <img src="https://github.com/jm199504/Other-Notes/blob/master/Git-Notes/images/4.jpg">


参考来源：廖雪峰Git教程.pdf
