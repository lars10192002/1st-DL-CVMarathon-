```

from keras.models import Sequential \#用來啟動 NN

from keras.layers import Conv2D \# Convolution Operation

from keras.layers import MaxPooling2D \# Pooling

from keras.layers import Flatten

from keras.layers import Dense \# Fully Connected Networks

from keras.layers import GlobalAveragePooling2D

```

```
input_shape = (32, 32, 3)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding='same',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))  ##pooling_size=2,2 strides=2,2 輸出feature map 大小為多少？
# 計算參考  https://zhuanlan.zhihu.com/p/29119239
# Params = (kernel_size * input_channels + 1) * kernel numbers=((3* 3)*3+1)*32 = 896
# Pooling_output = (Input+2*padding-Kernel_Size) /Stride+1= (32 + 2*1 - 3) / 2 + 1 = 16

model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))##pooling_size=2,2 strides=2,2 輸出feature map 大小為多少？
# Params = (kernel_size * input_channels + 1) * kernel numbers=((3* 3)*32+1)*64 = 18496
# Pooling_output = (Input+2*padding-Kernel_Size) /Stride+1= (16 + 2*1 - 3) / 2 + 1 = 8

model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(1,1), strides=(1,1)))##pooling_size=1,1 strides=1,1 輸出feature map 大小為多少？
# Params = (kernel_size * input_channels + 1) * kernel numbers=((3* 3)*64+1)*128 = 73856
# Pooling_output = (Input+2*padding-Kernel_Size) /Stride+1= (8 + 2*1 - 3) / 1 + 1 = 8


model.add(Conv2D(10, kernel_size=(3, 3), padding='same'))
model.add(Flatten()) ##Flatten完尺寸如何變化？
# Params = (kernel_size * input_channels + 1) * kernel numbers=((3* 3)*128+1)*10 = 11530
# Pooling_output = (Input+2*padding-Kernel_Size) /Stride+1= (8 + 2*1 - 3) / 1 + 1 = 8

model.add(Dense(units=28, input_shape=input_shape)) ##全連接層使用28個units
model.summary()

```


```
# none Flatten ,use GlobalAveragePooling2D
input_shape = (32, 32, 3)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding='same',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))  ##pooling_size=2,2 strides=2,2 輸出feature map 大小為多少？

model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))  ##pooling_size=2,2 strides=2,2 輸出feature map 大小為多少？

model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(1, 1)))  ##pooling_size=1,1 strides=1,1 輸出feature map 大小為多少？

model.add(Conv2D(10, kernel_size=(3, 3), padding='same'))
# model.add(Flatten()) ##Flatten完尺寸如何變化？
model.add(GlobalAveragePooling2D()) #關掉Flatten，使用GlobalAveragePooling2D，完尺寸如何變化？

model.add(Dense(28)) ##全連接層使用28個units

model.summary()

```