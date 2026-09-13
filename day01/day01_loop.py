#练习一：打印一到十
for i in range(1,11):
    print (i)

#练习二：遍历数组
fruits=["苹果","香蕉","橙子"]
for fruit in fruits:
    print("我喜欢吃",fruit)

#练习三：猜数字游戏
import random
secret=random.randint(1,10)
guess=0
while guess!=secret:
    guess=int(input("请随便输入一个数字"))
    if guess>secret:
        print("输入的数字太大了")
    elif guess<secret:
        print("输入的数字太小了")
    else:
        print("你猜对了")


