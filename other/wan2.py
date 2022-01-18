# import math
# x2=math.log(0.5,0.88)
# print(x2)

# import numpy
# In=input()
# points=input()
# for point in points:
#     x=point[0] -1
#     y=point[1] -1
#     if x-1 or y-1<0:
#         In[y+1,x]= (In[y+1,x]+1)%2
#         In[y,x+1]=(In[y,x+1]+1)%2
#     elif

#
# mo={}
# nums=[2, 4, 1, 2, 7, 8, 4,11,54,8,6,9,77,8,56,2,4,2,4,5]
# for i in range(1,len(nums)-1):
#     if nums[i] >=nums[i-1] and nums[i] >=nums[i+1]:
#         mo[nums[i]]= i
# print(mo[max(mo.keys())])

# a=3
# print(1<=a<=9)

# s='1233456'
# a=len(s)
# s=list(s)
# print(a)
# print(int(''.join(s[5:7])))


# s=input()
# list_s=list(s)
# if len(s)==1:
#     print('Likes')
# elif 'A'<=list_s[0]<='Z' and 'A'<=list_s[-1]<='Z':
#     for i in range(1,len(s)):
#         if list_s[i]==list_s[i-1]:
#             print('Dislikes')
#             break
#         else:
#             print('Likes')

# s=input()
# list_s=list(s)
# cost=0
# if len(s)==1:
#     print(0)
# elif list_s[0]!='?':
#     for i in range(1,len(s)):
#         if list_s[i]==list_s[i-1]:
#             cost+=1
#         elif list_s[i]=='?':
#             if list_s[i-1]=='A':
#                 list_s[i]='B'
#             elif list_s[i-1]=='B':
#                 list_s[i]='A'
# elif list_s[0]=='?':
#     j=1
#     while j<len(s):
#         if list_s[j-1]!='?':
#             if list_s[j] == list_s[j-1]:
#                 cost += 1
#             elif list_s[j] == '?':
#                 if list_s[j-1] =='A':
#                     list_s[j] = 'B'
#                 elif list_s[j-1] == 'B':
#                     list_s[j] = 'A'
#         j+=1
# print(cost)


# N=int(input())
# a=list(map(int,input().strip().split()))
# a1,a2,a3=a[0],a[1],a[2]
# for i in range(3,len(a)):
#     if a1^a[i]>=a2^a[i] and a1^a[i]>=a3^a[i]:
#         a1=a1^a[i]
#     elif a2^a[i]>=a1^a[i] and a2^a[i]>=a3^a[i]:
#         a2=a2^a[i]
#     elif a3^a[i]>=a1^a[i] and a3^a[i]>=a2^a[i]:
#         a3=a3^a[i]
# print(a1+a2+a3)




