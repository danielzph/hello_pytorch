
# nums=input('input_nums:')
# nums=[1,2,3,5]
# target=input('input_target:')
# target=6

# def twoSum(nums: list,target: int):
#     hashtable=dict()
#     for i,num in enumerate(nums):
#         if target-num in hashtable:
#             return [hashtable[target-num],i]
#         hashtable[nums[i]]=i
#     return []
# print(twoSum(nums,target))









# hashtable=dict()
# hashtable[2]=1
# hashtable[3]=2
# hashtable.get(4,5)
# print(hashtable.get(3,5))
# print(hashtable)
# print(hashtable.values())
# print(hashtable.keys())
# print(hashtable[3])
# print(hashtable.values())

# import collections
# words = ["cat","bt","hat","tree","cha"]
# chars = "atach"
# def countCharacters(words: list(), chars: str):
#         ch=collections.Counter(chars)
#         len_sum=0
#         for word in words:
#             wo=collections.Counter(word)
#             for i in wo:
#                 if ch[i] < wo[i]:
#                     break
#             else:
#                 len_sum+=len(word)
#         return len_sum
#
# print(countCharacters(words, chars))


# a=[1,5,6,8,9,6,5,7,4,5,6,9,7]
# # del a[0]
# print(a[:1])

# n,m=2,3
# dp = [[0] * 5 for _ in range(5)]
# Matrix = [[0]*n]*m
# print(dp)
# print(Matrix)

# a=[1,2,3]
# print(a[-1])

# list1=input().strip().split(' ')
# print(list1)

# a=list(map(int,input().strip().split(' ')))
# print(a)

# print(2015&2014)
# print(2014&2013)
# print(2012&2011)
# print(2008&2007)
# print(2000&1999)
# print(1984&1983)


# while 1:
#     l=list(map(int,input().strip().strip()))

# queue = []
# #入队
# queue.append(1)
# print(queue)
# queue.append(2)
# print(queue)
# queue.append(5)
# print(queue)
# #出队
# queue.pop(0)
# print(queue)
#
# stack = []
# #入栈
# stack.append(1)
# print(stack)
# stack.append(2)
# print(stack)
# stack.append(5)
# print(stack)
# #查看栈顶元素
# top = stack[-1]
# print('栈顶元素为:',top)
# #出栈
# stack.pop()
# print(stack)

# while True:
#     n,m,a,b=map(int,input().strip().split(' '))
#     list1=list(map(int,input().strip().split(' ')))
#     if n-m>=2:
#         if a<b:
#             for i in range(0, m):
#                 if a <= list1[i] <= b:
#                     out = 'YES'
#                 else:
#                     out = 'NO'
#                     break
#         elif a>b:
#             for i in range(0, m):
#                 if b <= list1[i] <= a:
#                     out = 'YES'
#                 else:
#                     out = 'NO'
#                     break
#         print(out)
#     elif n-m==1:
#         if a in list1 or b in list1:
#             print('YES')
#         else:
#             print('NO')
#     elif n-m==0:
#         if a in list1 and b in list1:
#             print('YES')
#         else:
#             print('NO')

# t=2
# while t:
#     t=t-1
#     print(1)

# while True:
#     try:
#         s1 = list(map(int,input().split()))
#         s2 = list(map(int,input().split()))
#         n, m, a, b = s1
#         s2.sort()
#         print(a)
#         print(s2)
#     except:
#         break
#
# a=[[ 0 for _ in range(10)] for _ in range(10)]
# print(a)
# import numpy as np
# a=[[0,1,3,4,5],[2,5,6,8,7],[5,3,6,9,7]]
# # a=np.array(a)
# # print(a.shape)
# print(len(a))
# print(max(a))



# a=[1,5,4,6,7,8,9]
# del a[3]
# print(a)


# a=None
# if not a:
# #     a=1
# c=5
# r=3
# dp= [[0] * c for _ in range(r)]
# # dp=[[0]*5]*3
# print(dp)


# a=list(map(int,input().strip().split(' ')))
# N,M=a[0],a[1]
# A=[]
# if N==0 and M==0:
#     print(0)
# for i in range(N):
#     A.append(list(map(int,input().strip().split(' '))))
# dp= [[0] * M for _ in range(N)]
# dp[0][0]=A[0][0]
#
# for i in range(1, N):
#     dp[i][0] = dp[i - 1][0] + A[i][0]
# for j in range(1, M):
#     dp[0][j] = dp[0][j - 1] + A[0][j]
# for i in range(1, N):
#     for j in range(1, M):
#         dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + A[i][j]
#
# print(dp[N - 1][M - 1])


# class ListNode:
#     def __init__(self,val=0,next=None):
#         self.val=val
#         self.next=next


#
# def QuickSort(a:list):
#     # 设计一个实现分区的函数
#     def partition(arr,left,right):
#         cir=left
#         while left<right:
#             while left < right and arr[right] >= arr[cir]:
#                 right-=1
#             while left < right and arr[left] <= arr[cir]:
#                 left+=1
#             (arr[left],arr[right])=(arr[right],arr[left])
#         (arr[left], arr[cir]) = (arr[cir], arr[left])
#         index=left
#         return index
#     # 快排，用到了递归
#     def quicksort(arr,left,right):
#         if left>=right:
#             return 0
#         mid=partition(arr,left,right)
#         quicksort(arr,left,mid-1)
#         quicksort(arr,mid+1,right)
#     n=len(a)
#     if n<=1:
#         return a
#     quicksort(a,0,n-1)
#     return a
# x=list(map(int,input('请输入待排序数列，并以' '隔开：\n').strip().split(' ')))
# arr=QuickSort(x)
# print('排列结果如下：')
# for k in arr:
#     print(k,end=' ')




# a=[1,3,5,4,6,9,7,8,33,2]
# b=[]
# b.append(a)
# print(b)


import sys
workstations=list(map(int,input().split(' ')))
sterilizers=list(map(int,input().split(' ')))
workstations.sort()
sterilizers.sort()
j,res=0,0
for i in range(len(workstations)):
    if j+1 <len(sterilizers) and workstations[i]<=sterilizers[j+1]:
        res=max(res,abs(sterilizers[j]-workstations[i]))
    elif j+1 == len(sterilizers):
        res=max(res,abs(sterilizers[j]-workstations[i]))
        j+=1
    else:
        j+=1
sys.stdout.write(str(res))



