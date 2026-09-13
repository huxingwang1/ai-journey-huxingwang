#练习一：打招呼
name=input("你叫什么名字：")
print("你好"+name+"!欢迎来到AI世界")

#联系二：判断年龄
age=int(input("请输入你的年龄"))
if age<18:
    print("您还未成年")
elif age<60:
    print("你是成年人")
else:
    print("你是老年人")
