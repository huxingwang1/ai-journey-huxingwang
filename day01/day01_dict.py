
#学生管理系统
students={}
def add_student(name,score):
    students[name]=score
    print(f"已添加{name},成绩{score}")

def show_all():
    for name,score in students.items():
        print(f"{name}:{score}")

def average():
    if len(students)==0:
        return 0
    return sum(students.values())/len(students)
#测试
add_student("张三",90)
add_student("李四",85)
add_student("王五",95)
show_all()
print("平均分是：",average())
