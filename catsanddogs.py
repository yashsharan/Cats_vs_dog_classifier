
# coding: utf-8

# In[2]:


import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import os

TRAIN_DIR='/home/yash/Deep_learning_projects/cats_vs_dogs_data/train'
TEST_DIR='/home/yash/Deep_learning_projects/cats_vs_dogs_data/test'
IMG_SIZE=50
LR=1e-3

MODEL_NAME='dogsvscats-{}-{}.model'.format(LR,'CNN_basicsv5')


# In[3]:


def label_img(img):
    word_label=img.split('.')[-3]
    if word_label == 'cat':
        return [1,0]
    elif word_label == 'dog':
        return [0,1]
    


# In[4]:


def train_data():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label=label_img(img)
        path=os.path.join(TRAIN_DIR,img)
        #print(path)
        #img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        train_data.append([np.array(img),np.array(label)])
    shuffle(train_data)
    np.save('trained_data.npy',train_data)
    return train_data
        


# In[5]:


def test_data():
    testing_data= []
    for img in tqdm(os.listdir(TEST_DIR)):
        path=os.path.join(TEST_DIR,img)
        img_num=img.split('.')[0]
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))        
        testing_data.append([np.array(img),img_num])
    np.save('test_data.npy',testing_data)
    return test_data

        


# In[6]:


trained_data=train_data()


# In[8]:


import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf

tf.reset_default_graph()

convnet=input_data(shape=[None,IMG_SIZE,IMG_SIZE,1], name='input')

convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)


convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1024,activation='relu')
convnet=dropout(convnet,0.8)

convnet=fully_connected(convnet,2,activation='softmax')
convnet=regression(convnet,optimizer='adam',learning_rate=LR,loss='categorical_crossentropy',name='targets')

model=tflearn.DNN(convnet,tensorboard_dir='log')


# In[9]:


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded')


# In[9]:


train=trained_data[:-500]
test=trained_data[-500:]


# In[10]:


X=np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y=[i[1] for i in train]


test_x=np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y=[i[1] for i in test]



# In[13]:


model.fit({'input':X},{'targets':Y},n_epoch=3,validation_set=({'input':test_x},{'targets':test_y}),
	snapshot_step=500,show_metric=True,run_id=MODEL_NAME)


# In[14]:


model.save(MODEL_NAME)


# In[23]:





# In[10]:


import matplotlib.pyplot as plt

#test_data_arr=test_data()
test_data_arr=np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data_arr[:12]):
    
    img_num=data[1]
    img_data=data[0]
    
    y=fig.add_subplot(3,4,num+1)
    
    orig=img_data
    data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    
    model_out=model.predict([data])[0]
    if np.argmax(model_out) == 1:img_label="Dog"
    else : img_label="Cat"
    
    y.imshow(orig,cmap='gray')
    plt.title(img_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()


# In[29]:


with open('submission-file.csv','w') as f:
    f.write('id,label\n')


# In[30]:


with open('submission-file.csv','a') as f:
    for data in tqdm(test_data_arr):
        img_num=data[1]
        img_data=data[0]  
        orig=img_data
        data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)    
        model_out=model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




