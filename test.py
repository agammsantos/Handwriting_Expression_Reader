import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os, os.path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib as jb


gambar1=Image.open('./extracted_images/2/2_100001.jpg')
gambar2=Image.open('./static/test/test.png')
# print(gambar1)
# print(gambar2)
# print(type(gambar1))
# print(gambar1.size)
# print(gambar2.size)
# print(list(gambar1.getdata()))
# print(list(gambar2.getdata()))

grey = gambar2.convert('L')
# bw = grey.point(lambda x: 0 if x<128 else 255, '1')
# bw.save("./static/test/grey.jpg")
# print(gambar1)
# print(gambar3)
# print(list(gambar3.getdata()))
# print(gambar1.size)
# print(gambar3.size)

gambar = grey.resize((45, 45))
gambartest=np.array(gambar.getdata())
# gambar4.show()
# print(gambartest)

i=0
img = []
while i<10:
    path=str(i)
    path = "C:/Users/agamm/Purwadhika/datasets/handwritemath/extracted_images/"+path
    valid_images = [".jpg",".png",".tga"]
    j=1
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            j+=1
            continue
        img.append(list(Image.open(os.path.join(path,f)).getdata()))
        j+=1
        if j>2500:
            break
    i+=1
img=np.array(img)

target=np.zeros((25000),dtype=int)
i=2500
j=2
while i < 25000:
    target[i:2500*j]=np.full((2500),j-1)
    i+=2500
    j+=1

# print(target)
# print(target[70000])
# print(target[72000])
# print((len(target)))

# print(len(img))
# print(img[0])
# print(len(img[0]))

data=pd.DataFrame(np.array(img))
data['target']=target
# data['len']=data['features'].apply(len)
# print(data.head())
# print(data[data['len']!=2025])

x=data.drop(['target'],axis='columns')
y=data['target']
xtr,xts,ytr,yts=train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=253 
    )

# print(xtr)
# print(xts)
# print(ytr)
# print(yts)

model=LogisticRegression(solver='liblinear',multi_class='auto')
model.fit(xtr,ytr)
print(model.score(xts,yts)*100,'%')
print(yts[0])
print(model.predict([gambartest]))

jb.dump(model,'modeldigit')