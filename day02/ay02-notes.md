# Day 2 笔记

## 文件读写
- open(文件名, 模式, encoding="utf-8")
- 模式：w 覆盖、r 读、a 追加
- with 自动关闭文件

## 异常处理
- try / except / finally
- 为什么需要：API 调用会出错

## 模块
- os.getcwd() 当前目录
- os.listdir() 列出文件
- json.dumps/loads 字符串转换
- json.dump/load 文件转换
- datetime.now().strftime() 格式化时间

## API 调用
- client.chat.completions.create()
- model、messages 必填
- response.choices[0].message.content 取回答

## 多轮对话
- messages 列表保存历史
- 每轮 append user 和 assistant
- 上下文 = 每次把历史全发给模型
- len(messages) 验证历史增长

## 对话保存
- json.dump(messages, f, ensure_ascii=False, indent=2)
- ensure_ascii=False 保中文
- indent=2 保格式

## 踩的坑
1. 中文编码报错 → sys.stdout 设 utf-8
3. create() 漏参数
4. 密钥不能硬编码，后面用环境变量