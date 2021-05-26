#!/usr/bin/env python
# coding: utf-8

# In[50]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt

import os


# In[51]:


len([name for name in os.listdir('Desktop/data/test/') for name in os.listdir('Desktop/data/test/' + name)])


# In[52]:


len([name for name in os.listdir('Desktop/data/train/') for name in os.listdir('Desktop/data/train/' + name)])


# In[53]:


img_width, img_height = 150, 150

train_data_dir = 'Desktop/data/train'
validation_data_dir = 'Desktop/data/test'
nb_train_samples = len([name for name in os.listdir('Desktop/data/train/') for name in os.listdir('Desktop/data/train/' + name)]) #1500
nb_validation_samples = len([name for name in os.listdir('Desktop/data/test/') for name in os.listdir('Desktop/data/test/' + name)]) #150

batch_size = 8

print(K.image_data_format())

if K.image_data_format() == 'channels_first':  #check chanel
    input_shape = (3, img_width, img_height)  
else:
    input_shape = (img_width, img_height, 3)  #chanel last


# สร้าง model CNN

# In[54]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))      #ทำให้ไม่ใช่liner
model.add(MaxPooling2D(pool_size=(2, 2)))     #การหาpixel ที่สำคัญ

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(15)) #class 
model.add(Activation('softmax'))


# In[55]:


model.summary()


# In[60]:


model.compile(loss='categorical_crossentropy', #categorical_crossentropy สำหรับหลายclass 
                      optimizer='adam',# rmsprop or adagrad or  #adam หาจุดต่ำสุดของerror
                      metrics=['accuracy']) #ความถูกต้อง


# In[61]:


train_datagen = ImageDataGenerator(
    rescale=1. / 255,      #ลดสเกล
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)    #ทำให้dataมีความหลากหลาย

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical') #binary # categorical

print(train_generator.class_indices)


# In[62]:


test_datagen = ImageDataGenerator(rescale=1. / 255)  #เป็นการรีสเกล

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical') #binary # categorical 


print(validation_generator.class_indices)


# In[63]:


epochs = 200   #รอบน้อยไป จำนวนรอบที่ weigh

history =model.fit_generator(    #เก็บค่าที่เทรนไว้
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,  #stepเข้าไปกี่รูปกี่รอบ
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)    #เทรนต่อรอบจะฟีตรูปเข้าไปกี่รูป คือจำนวนbatch_size


# In[68]:


model.save('plant.model')


# In[69]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()


# # Predict

# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


# In[3]:


img_width, img_height = 150, 150

model = load_model('plant.model')


# In[4]:


mypath = 'Desktop/data/predict/'           #ที่อยู่ของfloderที่จะเทส


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#print(onlyfiles)

# predicting images
for file in onlyfiles:
    #print(file)
    img = image.load_img(mypath+file, target_size=(img_width, img_height)) #150x150
    
    imgplot = plt.imshow(img)
    plt.show()
    
    
    x = image.img_to_array(img)     #แปลงเป็น array
    x /= 255.0
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict(images)
    #classes = model.predict_classes(images)
    #print(classes)
    #classes = classes[0][0]
    
    if np.argmax(classes) == 0:
        #print("%s , %.2f" % ("Artichoke(GreenGlobe)", (classes[0][np.argmax(classes)]*100)))
        #print(file + ": " + 'Artichoke(GreenGlobe')
        print("พืช : Artichoke (Greenglobe)") 
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")    
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 100:0:0") 
        print("เมื่อเปรียบเทียบกับแสงจาก : Natural light")      
        print("เร่งการเจริญเติบโตด้าน : ความยาวลำต้น")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 8")      
        print("PPFD = 236.54 umol/s/m2")   
    
    elif np.argmax(classes) == 1:
        #print("%s ,%.2f" % ("Artichoke(cardoon)", (classes[0][np.argmax(classes)]*100)))
        #print(file + ": " + 'Artichoke(cardoon)')
        print("พืช : Artichoke (Cardoon)") 
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")    
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 100:0:0") 
        print("เมื่อเปรียบเทียบกับแสงจาก : Natural light")      
        print("เร่งการเจริญเติบโตด้าน : ความยาวของลำต้น")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 8")      
        print("PPFD = 236.54 umol/s/m2")   
        
    elif np.argmax(classes) == 2:
        #print("%s ,%.2f" % ("Artichoke(violette)", (classes[0][np.argmax(classes)]*100)))
        #print(file + ": " + 'Artichoke(violette)')
        print("พืช : Artichoke (Violetto)") 
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")    
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 100:0:0") 
        print("เมื่อเปรียบเทียบกับแสงจาก : Natural light")      
        print("เร่งการเจริญเติบโตด้าน : ความยาวลำต้น")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 8")      
        print("PPFD = 236.54 umol/s/m2")   
          
    elif np.argmax(classes) == 3:
        #print("%s ,%.2f" % ("Chinese Cabbage", (classes[0][np.argmax(classes)]*100)))
        #print(file + ": " + 'Chinese Cabbage')
        print("พืช : Chinese cabbage") 
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 0:0:100") 
        print("เมื่อเปรียบเทียบกับแสงจาก : Fluorescent")      
        print("เร่งการเจริญเติบโตด้าน : น้ำหนักสด")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 12")      
        print("PPFD = 80 umol/s/m2") 
         
          
    elif np.argmax(classes) == 4:
        #print("%s ,%.2f" % ("Chinese Kale", (classes[0][np.argmax(classes)]*100)))
        #print(file + ": " + 'Chinese Kale')
        print("พืช : Chinese kale")  
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")      
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 88.88:0:11.11") 
        print("เมื่อเปรียบเทียบกับแสงจาก : White(control)")      
        print("เร่งการเจริญเติบโตด้าน : ความยาวลำต้นและน้ำหนักสด")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 12")      
        print("PPFD = 120 umol/s/m2")   
                 
    elif np.argmax(classes) == 5:
        #print("%s ,%.2f" % ("Dill", (classes[0][np.argmax(classes)]*100)))
        #print(file + ": " + 'Dill')
        print("พืช : Dill")  
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")      
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 93:0:7") 
        print("เมื่อเปรียบเทียบกับแสงจาก : High pressure")      
        print("เร่งการเจริญเติบโตด้าน : ความยาวลำต้น")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 24")      
        print("PPFD = 100 umol/s/m2")  
          
    elif np.argmax(classes) == 6:
        #print("%s ,%.2f" % ("Lettice green oak", (classes[0][np.argmax(classes)]*100)))
        #print(file + ": " + 'Lettice green oak')
        print("พืช : Lettice green oak") 
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 75:0:25") 
        print("เมื่อเปรียบเทียบกับแสงจาก : sunlight")      
        print("เร่งการเจริญเติบโตด้าน : ความยาวลำต้น")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 16")      
        print("ตัวแปรควบคุมอื่น ๆ : Hydroponics system") 
          
    elif np.argmax(classes) == 7:
        #print("%s ,%.2f" % ("Lettuce", (classes[0][np.argmax(classes)]*100)))
        #print(file + ": " + 'Lettuce')
        print("พืช : Lettuce")  
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 88.88:0:11.11") 
        print("เมื่อเปรียบเทียบกับแสงจาก : Fluorescent")      
        print("เร่งการเจริญเติบโตด้าน : น้ำหนักสด")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 12")      
        print("PPFD = 200 umol/s/m2")  
          
    elif np.argmax(classes) == 8:
        #print("%s ,%.2f" % ("Myrtus communis", (classes[0][np.argmax(classes)]*100)))
        #print(file + ": " + 'Myrtus communis')
        print("พืช : Myrtus communis") 
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 0:0:100") 
        print("เมื่อเปรียบเทียบกับแสงจาก : Control")      
        print("เร่งการเจริญเติบโตด้าน : น้ำหนักสด")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 24")      
        print("PPFD = 35 umol/s/m2")  
              
    elif np.argmax(classes) == 9:
        #print("%s ,%.2f" % ("Oncidium", (classes[0][np.argmax(classes)]*100)))
        #print(file + ": " + 'Oncidium')
        print("พืช : Oncidium")  
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")  
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 100:0:0") 
        print("เมื่อเปรียบเทียบกับแสงจาก : Fluorecent")      
        print("เร่งการเจริญเติบโตด้าน : ความยาวลำต้น")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 24")      
        print("PPFD = 50 umol/s/m2") 
          
    elif np.argmax(classes) == 10:
        #print("%s ,%.2f" % ("Potato", (classes[0][np.argmax(classes)]*100)))
       # print(file + ": " + 'Potato')
        print("พืช : Potato")
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")      
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 100:0:0") 
        print("เมื่อเปรียบเทียบกับแสงจาก : Fluorescent")      
        print("เร่งการเจริญเติบโตด้าน : ความยาวลำต้นและน้ำหนักสด")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 16")      
        print("PPFD = 100 umol/s/m2")   
              
    elif np.argmax(classes) == 11:
        #print("%s ,%.2f" % ("Red Oakleaf", (classes[0][np.argmax(classes)]*100)))
        #print(file + ": " + 'Red Oakleaf')
        print("พืช : Red oakleaf")  
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 80:0:20") 
        print("เมื่อเปรียบเทียบกับแสงจาก : Fluorescent")      
        print("เร่งการเจริญเติบโตด้าน : น้ำหนักสด")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 16")      
        print("PPFD = 200 umol/s/m2")  
          
    elif np.argmax(classes) == 12:
        #print("%s ,%.2f" % ("Tomato", (classes[0][np.argmax(classes)]*100)))
        #print(file + ": " + 'Tomato')
        print("พืช : Tomato ")   
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")     
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 66:0:33") 
        print("เมื่อเปรียบเทียบกับแสงจาก : Fluorescent")      
        print("เร่งการเจริญเติบโตด้าน : ความยาวลำต้นและน้ำหนักสด")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 24")   
          
    elif np.argmax(classes) == 13:
        #print("%s ,%.2f" % ("Vanilla", (classes[0][np.argmax(classes)]*100)))
        #print(file + ": " + 'Vanilla')
        print("พืช : Vanilla") 
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 50:0:50") 
        print("เมื่อเปรียบเทียบกับแสงจาก : Fluorescent")      
        print("เร่งการเจริญเติบโตด้าน : น้ำหนักสด")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 16")      
        print("PPFD = 25 umol/s/m2")
        
    else:
        #print("%s , %.2f" % ("Water morning Glory", (classes[0][np.argmax(classes)]*100)))
        #print(file + ": " + 'Water morning Glory')
        print("พืช : Water morning glory ")   
        print("ผลข้อมูลการควบคุมการเร่งการเจริญเติบโตของพืชที่ดีที่สุด")
        print("อัตราส่วนแสงเสริมที่ดีที่สุดในการเร่งการเจริญเติบโต R:G:B : 100:0:0") 
        print("เมื่อเปรียบเทียบกับแสงจาก : Fluorescent")      
        print("เร่งการเจริญเติบโตด้าน : ความยาวลำต้น")      
        print("จำนวนชั่วโมงในการเปิดต่อวัน : 24")      
        print("ตัวแปรควบคุมอื่น ๆ : Drain aquaponics system") 
        
    print('--------------------------------------------------------------')

