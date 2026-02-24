path=r'C:\Users\gmalb\Downloads\f1Analysis\.github\workflows\keep-alive.yml'
with open(path,'rb') as f:
    data=f.read()
for i,b in enumerate(data):
    if b>127:
        print(i,hex(b))
