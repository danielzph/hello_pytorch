import pymssql

# 打开数据库连接 这里的host='.'也可用本机ip或ip+端口号
conn = pymssql.connect(host="192.168.191.3",user= "sa",password= "123456", database="017-1", charset='utf8' )
# 使用cursor()方法获取操作游标
cursor = conn.cursor()
# SQL 查询语句
sql = "SELECT TOP 5 * FROM Table_1 "
try:
   # 执行SQL语句
   cursor.execute(sql)
   # 获取所有记录列表
   results = cursor.fetchall()
   print(results)
   # print(results.)
except:
   print('连接失败')
# 关闭数据库连接
conn.close()


