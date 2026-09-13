# 练习1：基本异常处理
try:
    num = int(input("请输入一个数字："))
    result = 10 / num
    print("结果是：", result)
except ValueError:
    print("你输入的不是数字！")
except ZeroDivisionError:
    print("不能除以0！")
except Exception as e:
    print("出错了：", e)
finally:
    print("程序结束")

# 练习2：文件读取异常
try:
    with open("不存在的文件.txt", "r", encoding="utf-8") as f:
        content = f.read()
except FileNotFoundError:
    print("文件不存在！")