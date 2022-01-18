import sys
workstations=list(map(int,input().split(' ')))
sterilizers=list(map(int,input().split(' ')))
workstations.sort()
sterilizers.sort()
j,res=0,0
for i in range(len(workstations)):
    if 1<j+1<len(sterilizers) and workstations[i]<=sterilizers[j+1]:
        res=max(res,abs(sterilizers[j]-workstations[i]))
        if res%2==0: res=res/2
        else: res=int(res/2)+1
    elif j==0:
        res = max(res, abs(sterilizers[j] - workstations[i]))
    elif j+1 == len(sterilizers):
        res=max(res,abs(sterilizers[j]-workstations[i]))
    else:
        j+=1
sys.stdout.write(str(res))