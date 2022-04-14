# import sys
# workstations=list(map(int,input().split(' ')))
# sterilizers=list(map(int,input().split(' ')))
# workstations.sort()
# sterilizers.sort()
# j,res=0,0
# for i in range(len(workstations)):
#     if 1<j+1<len(sterilizers) and workstations[i]<=sterilizers[j+1]:
#         res=max(res,abs(sterilizers[j]-workstations[i]))
#         if res%2==0: res=res/2
#         else: res=int(res/2)+1
#     elif j==0:
#         res = max(res, abs(sterilizers[j] - workstations[i]))
#     elif j+1 == len(sterilizers):
#         res=max(res,abs(sterilizers[j]-workstations[i]))
#     else:
#         j+=1
# sys.stdout.write(str(res))

import sys

n,m=map(int,input().strip().split(' '))
mes=[]
for _ in range(m*m):
    mes.append(list(map(str,input().split(' '))))
mes_t,mes_m=[],[]
for i in range(len(mes)):
    if (i+1)%2==0:
        mes_m=mes_m+mes[i]
    else:
        mes_t=mes_t+mes[i]
# print(mes_t)
# print(mes_m)
d={}
for j in range(len(mes)):
    for k in range(len(mes[j])):
        if mes[j][k] in d.keys():
            break
        # t
        x,y=0,0
        for z in range(len(mes_t)):
            if mes_t[z]==mes[j][k]:
                x+=1
        for c in range(len(mes_m)):
            if mes_m[c]==mes[j][k]:
                y+=1
        d[mes[j][k]]=3*x+y
d=sorted(d.items(), key=lambda e:e[1], reverse=True)

out=[]
for l in range(n):
    o=d[l]
    out.append(o[0]+' ')
print(''.join(out))






