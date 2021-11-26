

![img](https://images2018.cnblogs.com/blog/1476067/201808/1476067-20180830103917637-1345060579.png)

 

#### 一、windows下，打开命令行，重新生成一个jupyter配置文件

- ```
  jupyter notebook --generate-config　　
  ```

   

 

#### 二、修个配置文件

- 找到这个新生成的文件：Windows: `C:\Users\USERNAME\.jupyter\jupyter_notebook_config.py` 

- 搜索 NotebookApp.allow_password_change，改为：NotebookApp.allow_password_change=False ，**去掉注释**

 

 

#### 三、回到windows命令行，运行jupyter notebook password

- 

  ```
  C:\Windows\System32>jupyter notebook password
  Enter password:             #键入密码，不显示的
  Verify password:            #再次重复键入密码
  [NotebookPasswordApp] Wrote hashed password to C:\Users\用户\.jupyter\jupyter_notebook_config.json     #密码生成的一串sha1（即是
  NotebookPasswordApp），写入到这个文件
  ```

  

#### 四、在 jupyter_notebook_config.py 配置文件中找到 “c.NotebookApp.password“，插入刚生成的那个密码sha1，效果如下：去掉注释

- ```
  c.NotebookApp.password = 'NotebookPasswordApp'
  ```

   

> 注意去掉代码行前的空格

#### 五、配置外网访问

在 jupyter_notebook_config.py 中找到下面的行，取消注释**就是把这几行代码最前面的#号去掉**并修改。

```
c.NotebookApp.ip``=``'*'`    `#在所有的网卡接口上开启服务` `c.NotebookApp.port ``=``8888` `#可自行指定一个端口, 访问时使用该端口7777` `c.NotebookApp.allow_remote_access ``=` `True` `#允许远程
```

　　

**注：如果购买的是阿里云的服务器，或者腾讯云的服务器，一定要在控制台里面的安全组里添加相对应的端口，另外windows服务器的话，一定要在防火墙里也添加相对应的端口，否则会造成无法访问。**

> **修改jupyter notebook默认工作路径**
>
> ```
> 在jupyter_notebook_config.py文件中查找c.NotebookApp.notebook_dir，把前面的注释符号#号去掉，然后把后面的路径改成自己想设置成的路径，如下所示：
> c.NotebookApp.notebook_dir ``=` `'D:\\JupyterProject'
> ```
>
> 　　

#### 六、重启 Jupyter



#### 参考

1.  [关于jupyter notebook密码设置](https://www.cnblogs.com/honway/p/9559324.html)

2.  [设置 jupyter notebook 外网远程访问](https://www.cnblogs.com/pychina/articles/12122710.html)

   

