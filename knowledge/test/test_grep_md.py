"""
  @Author:桌角是小黑
  @Time:2026/10/9
  @Desc: 正则表达式匹配
"""
import re

md_content = "这是一段文字 ![图片描述](images/photo.jpg) 后面的文字"

# 匹配所有图片
pattern = r"!\[(.*?)\]\((.*?)\)"
matches = re.findall(pattern, md_content)
# [('图片描述', 'images/photo.jpg')]

# 匹配特定图片文件名
# 在一段文本中，精准地把包含特定图片名字（img_name）的完整 Markdown 图片标签（![...](...)）给揪出来。
image_filename = "photo.jpg"
specific_pattern = r"!\[.*?\]\(.*?" + re.escape(image_filename) + r".*?\)"

# 1. re.compile(...) 作用：预编译正则。因为这个正则可能要在几十万字的 Markdown 里循环匹配，提前编译好能极大地提升 Python 的执行速度
# 2. r"!\[" 作用：匹配 Markdown 图片的开头 ![
# 3. .*? 作用：匹配中括号里的“图片描述（Alt Text） 代表任意字符，* 代表 0 个或多个，最关键的是这个 ?。它叫“非贪婪模式”，意思是**“见好就收，遇到下一个条件就立刻停止”**。如果不加 ?，它会像贪吃蛇一样一口气吃到文章最后的一个 ]，导致把两张不同的图片连带着中间的文字全部吞掉。
# 4. \]\( 作用：匹配中间的转折部分 ](。同样，这两个符号在正则里都有特殊含义，必须加 \ 转义。
# 5. .*? 作用：匹配括号里面、图片名字前面的路径（比如 /images/temp/）。
# 6. re.escape(img_name) 强行安全注入图片名。 假设 img_name 是 "pic.png"。如果不加 re.escape，直接拼进去，正则会把里面的 . 当成“任意字符”
# 7. .*? 作用：匹配图片名字后面可能跟着的杂七杂八的东西（比如鼠标悬停提示词 "这是头像"）。
# 8. \) 作用：匹配 Markdown 语法的最后一个右括号 )，