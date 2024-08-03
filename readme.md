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

### 打包

```bash
.\.venv\Scripts\activate
Pyinstaller -F -i .\Ptools_B.ico .\main.py
```

