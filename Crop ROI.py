
import cv2
import numpy as np  

class CoordinateStore:
    def __init__(self):
        self.points = []
        self.ix,self.iy = -1,-1

    def draw_rectangle(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ix,self.iy = x,y
            # self.points.append((self.ix,self.iy))
            # self.points.append((self.ix,self.iy))
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.rectangle(img,(self.ix,self.iy),(x,y),(0,255,0),1)
            self.points.append([self.ix,self.iy,x,y])


coordinateStore1 = CoordinateStore()

img = cv2.imread('test/equations.jpg')
copy = img.copy()
cv2.namedWindow('image')
cv2.setMouseCallback('image',coordinateStore1.draw_rectangle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()


for i in range(len(coordinateStore1.points)):
    img = cv2.imread('test/equations.jpg')
    copy = img.copy()
    roi = copy[coordinateStore1.points[i][1]:coordinateStore1.points[i][3], coordinateStore1.points[i][0]:coordinateStore1.points[i][2]]
    cv2.imshow("ROI", roi)
    cv2.imwrite("ROI/"+str(i)+".jpg",roi)
    cv2.waitKey(0)