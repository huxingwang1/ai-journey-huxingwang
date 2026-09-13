# 练习1：写文件
with open("test.txt", "w", encoding="utf-8") as f:
    f.write("Hello, AI World!\n")
    f.write("这是第二行\n")
    f.write("这是第三行\n")

print("写入完成")

# 练习2：读文件
with open("test.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print("文件内容：")
    print(content)

# 练习3：按行读
with open("test.txt", "r", encoding="utf-8") as f:
    for line in f:
        print("行：", line.strip()) 

# 练习4：追加
with open("test.txt", "a", encoding="utf-8") as f:
    f.write("这是追加的一行\n")

print("追加完成")