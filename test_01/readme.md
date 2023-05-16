#配置pytest命令行运行参数
   [pytest]
    addopts = -s ... # 空格分隔，可添加多个命令行参数 -所有参数均为插件包的参数配置测试搜索的路径
    testpaths = ./scripts  # 当前目录下的scripts文件夹 -可自定义
#配置测试搜索的文件名称
    python_files = test*.py 
#当前目录下的scripts文件夹下，以test开头，以.py结尾的所有文件 -可自定义
配置测试搜索的测试类名
    python_classes = Test_*  
 
   #当前目录下的scripts文件夹下，以test开头，以.py结尾的所有文件中，以Test开头的类 -可自定义
配置测试搜索的测试函数名
  
    python_functions = test_*
 
#当前目录下的scripts文件夹下，以test开头，以.py结尾的所有文件中，以Test开头的类内，以test_开头的方法 -可自定义
