吴保宙-作业git提交-操作_wubaozhou_20211201

git pull
git add . --（ 代表将所有文件都提交）
该命令作用就是将我们需要提交的代码从工作区添加到暂存区，就是告诉git系统，我们要提交哪些文件，之后就可以使用git commit命令进行提交了。
git commit -m “吴保宙提交作业目录_20211201”  --git commit 主要是将暂存区里的改动给提交到本地的版本库。
git push --最后一步将本地版本库的分支推送到远程服务器上对应的分支了


教程链接 https://www.cnblogs.com/jinqi520/p/10384225.html
https://juejin.cn/post/6844903598522908686#heading-2
https://git-scm.com/book/zh/v2
https://mp.weixin.qq.com/s?__biz=MzAxNjYyMzUzNQ==&mid=100006558&idx=4&sn=8c7862c7217fc05f4d6eea1a77bd2fa5&chksm=1bf0a85e2c872148d752632f7409971b08ecb825f2644027736056948573cc737b57acc4d7b3#rd


至此就已经将Repository拉到本地了，但是本地的仓库只和自己github上的远程仓库建立了连接，没有和源仓库建立链接，
如果还想和源仓库建立链接，可以如下命令：
git remote add upstream https://github.com/michael0420/BADOU-AI-Tsinghua.git
git remote add upstream https://github.com/jinqi520/chapter7.git

日常提交操作命令
git pull
git add .
git commit -m "119-吴保宙-提交作业目录_2021-1201-1536"
git push

