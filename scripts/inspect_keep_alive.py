import sys
path = r'C:\Users\gmalb\Downloads\f1Analysis\.github\workflows\keep-alive.yml'
with open(path,'rb') as f:
    data = f.read()
print('length', len(data))
# show bytes around 1350
for i in range(1330,1370):
    print(i, hex(data[i]), chr(data[i]) if data[i]<128 else '?')
