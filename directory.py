import os

os.chdir('C:/Users/Rahul Gupta/Desktop/project folder/image_classification_assessment/val/non-medical')


i=1
for file in os.listdir():
    src=file
    dst="0"+"_"+str(i)+".jpg"
    os.rename(src,dst)
    i+=1
