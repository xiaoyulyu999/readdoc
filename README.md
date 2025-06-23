# 1. 创建虚拟环境（你可以把 venv 换成你想要的文件夹名）
python3 -m venv venv

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 安装 Sphinx 和主题
pip install sphinx sphinx-pdj-theme


| 层级   | 用法     | 示例字符              |
| ---- | ------ | ----------------- |
| 一级标题 | 文档标题   | `=`               |
| 二级标题 | 章节或副标题 | `-`               |
| 三级标题 | 小节     | `^`、`~`、`"`（任选其一） |
| 四级标题 | 小小节    | `'` 或 `+`（任选）     |

# 4. Copy paste button
 pip install sphinx-copybutton    
 
# 5. code tabs
pip install sphinx_code_tabs