### 创建虚拟环境

```bash
python -m venv .venv
```

### 安装依赖

```bash
.\.venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r .\requirements.txt
```

* 修改 `Pyinstaller`：

  将 `hook-pyecharts.py` 复制到 `submodule\nav_analysis\.venv\Lib\site-packages\PyInstaller`目录下

### 打包

```bash
.\.venv\Scripts\activate
Pyinstaller -F -i .\Ptools_B.ico .\main.py
```

### 运行

双击 `main.exe`或者运行

```bash
.\dist\main.exe 
```
