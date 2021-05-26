#!/usr/bin/env python
# coding: utf-8

# In[180]:


from selenium import webdriver
from bs4 import BeautifulSoup
import urllib.request
import time
import os
from tqdm import tqdm          #tagbar โหลดกี่%


# In[181]:


print("What do you want to download?")
download = input()


# In[182]:


try:
    os.mkdir(download)
except OSError:
    print ("Creation of the directory %s failed" % download)


# In[183]:


site = 'https://www.google.com/search?tbm=isch&q=' + download


# In[184]:


driver = webdriver.Firefox(executable_path="Desktop/geckodriver-v0.26.0-win64/geckodriver.exe")


# In[185]:


driver.get(site)


# scorebar

# In[186]:


i = 0

while i < 1:
    # for scrolling page
    driver.execute_script("window.scrollBy(0,document.body.scrollHeight)")

    try:
        # for clicking show more results button
        driver.find_element_by_xpath("/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[5]/input").click()
    except Exception as e:
        pass
    time.sleep(5)
    i += 1


# In[187]:


soup = BeautifulSoup(driver.page_source, 'html.parser')

# closing web browser
driver.close()


# In[188]:


img_tags = soup.find_all("img", class_="rg_i")
#print('img_tags: ' + str(img_tags))
count = 0
for i in tqdm(img_tags):
    # print(type(i))
    # print(i)
    # if i.get('src') and i.get('data-src'):
    #     print(i['src'])
    try:
        if i.get('src'):
            # passing image urls one by one and downloading
            urllib.request.urlretrieve(i['src'], './' + download + '/' + str(count) + ".jpg")
            print(i)
        elif i.get('data-src'):
            urllib.request.urlretrieve(i['data-src'], './' + download + '/' + str(count) + ".jpg")
            print(i)
        count += 1
        print("Number of images downloaded = " + str(count), end='\r')
    except Exception as e:
        pass


# In[ ]:





# In[ ]:





# In[ ]:




