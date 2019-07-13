# 打开or创建一个csv文件

# 设置newline，否则两行之间会空一行

csv_file = open('student.csv','w',newline='')

try:
    # 获取csv写入对象
    writer = csv.writer(csv_file)
    # 写入行数据
    writer.writerow(('姓名','学号','年龄','性别'))
    for i in range(0,10):
        writer.writerow((i,i+3,i*2,i+5))
except Exception as e:
    print(e)
finally:
    # 关闭IO流
    csv_file.close()

# 打开csv文件并保存到txt
with open('student.csv','r') as f:
    reader = csv.reader(f)
    rows = [row for row in reader]
    for row in rows:
        print(row)
        for index,item in enumerate(row):
            # a表示追加模式，这里使用w，写完第一行在写第二行时候会覆盖第一行
            with open('student.txt','a') as f:
                if index == len(row)-1:
                    f.write(item+'\n')
                else:
                    f.write(item+'  ')
    # 关闭文件流
    f.close()