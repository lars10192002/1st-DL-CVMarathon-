# 1st-DL-CVMarathon-

## init env 

```
1.conda create --name cnn python=3.7

2.conda activate cnn

3.pip install keras==2.2.4

4.pip install tensorflow==1.14.0
```

## Note

```
##輸入照片尺寸==28*28*1
##都用一層，288個神經元

##建造一個一層的CNN層
classifier=Sequential()

##Kernel size 3*3，用32張，輸入大小28*28*1
classifier.model.add(Convolution2D(32, 3, 3, input_shape = (28, 28, 1)))
'''32張Kernel，大小為3*3，輸入尺寸為28*28*1'''
##看看model結構
print(classifier.summary())
'''理解輸出Total params為何==320'''

##建造一個一層的FC層
classifier=Sequential()
##輸入為28*28*1攤平==784
inputs = Input()'''輸入尺寸為28*28*1'''
##CNN中用了(3*3*1)*32個神經元，我們這邊也用相同神經元數量
x=Dense()(inputs)'''使用288個神經元'''
model = Model(inputs=inputs, outputs=x)
##看看model結構
print(model.summary())
'''理解輸出Total params為何==226080'''

##大家可以自己變換設定看看參數變化

```

