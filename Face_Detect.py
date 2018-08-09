#import required libraries
#import OpenCV library
import cv2
#import matplotlib library
# import matplotlib.pyplot as plt
#importing time library for speed comparisons of both classifiers
import time,os
from random import randint
# get_ipython().run_line_magic('matplotlib', 'inline')

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_faces(img_name, f_cascade, colored_img, scaleFactor = 1.1):
    #just making a copy of image passed, so that passed image is not changed
    img_copy = colored_img.copy()
    
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)#,minSize=(25,25),flags=0)
    
    i = 1
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        sub_img=img_copy[y:y+h,x:x+w]
        cv2.imwrite('Extracted/'+ img_name + '_' + str(i) +".jpg",sub_img)
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        i+=1
        
    return img_copy

def main():

    #load cascade classifier training file for haarcascade
    haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

    #load another image
    if not "Extracted" in os.listdir("."):
        os.mkdir("Extracted")
    root_dir = 'data/'
    files = os.listdir(root_dir)
    suffix_arr = ['jpg','png']
    for file in files:
        file_name, file_suffix = file.split('.')
        if file_suffix not in suffix_arr:
            continue
        print(file)
        img = cv2.imread(root_dir+file)

        #call our function to detect faces
        faces_detected_img = detect_faces(file_name, haar_face_cascade, img, scaleFactor = 1.1)

        #conver image to RGB and show image
        #plt.imshow(convertToRGB(faces_detected_img))

if __name__ == "__main__":
    main()
