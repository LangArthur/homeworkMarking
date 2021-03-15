# import numpy as np
# import cv2 as cv2

# class ResultPoints():
#     def __init__(self):
#         self.points = []
    
# drawing = False # true if mouse is pressed
# mode = True # if True, draw rectangle. Press 'm' to toggle to curve
# ix,iy = -1,-1
# # mouse callback function
# def draw_rectangle(event,x,y,flags,param):
#     global ix,iy,drawing,mode
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix,iy = x,y
#     # elif event == cv2.EVENT_MOUSEMOVE:
#     #     if drawing == True:
#     #         if mode == True:
#     #             cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#     #         else:
#     #             cv2.circle(img,(x,y),5,(0,0,255),-1)
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
       

# img = cv2.imread('test/test4.jpg')
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_rectangle)
# while(1):
#     cv2.imshow('image',img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == ord('m'):
#         mode = not mode
#     elif k == 27:
#         break
# cv2.destroyAllWindows()

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

#instantiate class
coordinateStore1 = CoordinateStore()


# Create a black image, a window and bind the function to window
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


# print ("Selected Coordinates: ")
# cv2.rectangle(copy, coordinateStore1.points[0], coordinateStore1.points[1], (0, 255, 0), 2)
# cv2.imshow("image", copy)
# cv2.waitKey(0)
# roi = copy[coordinateStore1.points[0][1]:coordinateStore1.points[1][1], coordinateStore1.points[0][0]:coordinateStore1.points[1][0]]
# cv2.imshow("ROI", roi)
# cv2.waitKey(0)
for i in range(len(coordinateStore1.points)):
    print(i)
    img = cv2.imread('test/equations.jpg')
    copy = img.copy()
    roi = copy[coordinateStore1.points[i][1]:coordinateStore1.points[i][3], coordinateStore1.points[i][0]:coordinateStore1.points[i][2]]
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)