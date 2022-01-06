import cv2
import os


nw = 100

CHARS = ' .,-_`~+=*^)(&%$#@'
CHARS = sorted(CHARS, reverse=True)
inp = input()
cap = cv2.VideoCapture(f'{inp}_output.mp4')
os.system('cls')
while cap.isOpened:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    nh = int(h / (2 * w) * nw)

    img = cv2.resize(img, (nw, nh))

    for row in img:
        for pixel in row:
            i = int(pixel / 256 * len(CHARS))
            print(CHARS[i], end='')
        print()
    os.system('cls')
    #print("\x1b[H", end='')
    #print("\x1B[J")

#img = cv2.imread('project.jpg', 0)
#image = cv2.imread('project.jpg', 0) #flag 0은 흑백이라 z값이 사라짐.
#print("\x1B[H")
#print("\x1b[2J", end='')

#h, w = image.shape

#nh = int(h / (2*w) * nw)

#image = cv2.resize(image, (nw, nh))

#for row in image:
#    for pixel in row:
#        i = int(pixel/ 256 * len(CHARS))
#        print(CHARS[i], end='')
#    print()
