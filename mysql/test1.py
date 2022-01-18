import pymysql
import pandas as pd
import tkinter
from tkinter import Tk, Scrollbar, Frame
from tkinter.ttk import Treeview

#打开数据库连接
conn = pymysql.connect(host='127.0.0.1',user= "root",password="123456",port= 3306,db='test1')
# conn.select_db('test1')
cursor = conn.cursor()
SQL = 'SELECT * FROM tb_emp2;'
cursor.execute(SQL)
results = cursor.fetchall()
df = pd.DataFrame([[ij for ij in i] for i in results])
df.rename(columns={0: 'speed', 1: 'temperature', 2: 'others1'},inplace=True)

print(results)
print(df)   # 将元组转换成了DF格式并显示出。
cursor.close()
conn.close()

# 界面展示
window = tkinter.Tk()
window.title('Data Acquisition System')
window.geometry('1300x800')

#使用Treeview组件实现表格功能
frame = Frame(window)
frame.place(x=0, y=10, width=480, height=280)

#滚动条
scrollBar = tkinter.Scrollbar(frame)
scrollBar.pack(side=tkinter.RIGHT, fill=tkinter.Y)

#Treeview组件，6列，显示表头，带垂直滚动条
tree = Treeview(frame, columns=('c1', 'c2', 'c3'),show="headings",yscrollcommand=scrollBar.set)

tree.pack(side=tkinter.LEFT, fill=tkinter.Y)

#Treeview组件与垂直滚动条结合
scrollBar.config(command=tree.yview)


df_col = df.columns.values
tree["columns"] = (df_col)
counter = len(df)
print(df_col)
print(counter)
df=df.values
#generating for loop to create columns and give heading to them through df_col var.
for x in range(len(df_col)):
    print(df_col[x])
    tree.column(x, width=150, anchor='center')
    tree.heading(x, text=df_col[x])
#generating for loop to print values of dataframe in treeview column.
    for i in range(counter):
        print(df[i][x])
        tree.insert('', i, values=(df[i][x]))


window.mainloop()





