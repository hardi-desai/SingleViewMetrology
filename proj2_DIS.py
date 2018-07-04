
import numpy as np
import math
import os
import cv2

#Reading the image
img = cv2.imread('box.png')
img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )

#Create default parametrization LSD
lsd = cv2.createLineSegmentDetector(0)

#Detect lines in the image
lines = lsd.detect(img)[0]#Position 0 of the returned tuple are the detected lines
thresholded_lines = []
slope = []

#Thresholding lines using length and slope
for line in lines:
    for everyline in line:
        if abs(everyline[0] - everyline[2])>125 or abs(everyline[1] - everyline[3])>125:
            thresholded_lines.append(everyline)
            slope.append((everyline[1] - everyline[3])/(everyline[0] - everyline[2]))

slope = np.array(slope)
indexs = np.argsort(slope)

# print('indexs')
# print(indexs)

#sorting lines according to slope
sorted_lines = []
for i in range(0,len(thresholded_lines)):
    sorted_lines.append(list(thresholded_lines[indexs[i]]))

#Selecting the reference X, Y and Z lines
Selected_lines_indexes = [0,2,3,4,10,11]
Selected_lines = []
for i in range(0,len(Selected_lines_indexes)):
    Selected_lines.append(sorted_lines[Selected_lines_indexes[i]])

def homogeneous_coordinates(points):
    points.append(1)
    return points

#Defining points in homogeneous_coordinates
points = []
for line in Selected_lines:
    points.append(homogeneous_coordinates(line[0:2]))
    points.append(homogeneous_coordinates(line[2:4]))
# print(points)

#Defining points in X ,Y and Z plane
X_points = [points[4],points[5],points[6],points[7]]
Y_points = [points[8],points[9],points[11],points[10]]
Z_points = [points[1],points[0],points[2],points[3]]

def construct_lines(points):
    lines= []
    for point in points[:-1]:
        next_point = points[points.index(point)+1]
        # print(points,'points')
        # print(point,'point')
        # print(next_point,'next_point')
        lines.append(np.cross(point,next_point))
    return lines

#Constructing lines in X ,Y and Z plane in homogeneous coordinate System
lines_x = construct_lines(X_points)
lines_y= construct_lines(Y_points)
lines_z = construct_lines(Z_points)

# print(lines_x)
# print(lines_y)
# print(lines_z)

thresholded_lines = np.array(thresholded_lines)
Selected_lines = np.array(Selected_lines)

# #Draw detected lines in the image
all_lines_img = lsd.drawSegments(img,lines)
thresholded_lines_img = lsd.drawSegments(img,thresholded_lines)
selected_lines_img = lsd.drawSegments(img,Selected_lines)


# Show image
cv2.imwrite('Original.png',img)
cv2.imwrite("LSD_all_lines.png",all_lines_img)
cv2.imwrite("thresholded_lines.png",thresholded_lines_img)
cv2.imwrite("Selected_Lines.png",selected_lines_img )


#Calculation of vanishing points
vanishing_points = []
vanishing_points.append(np.cross(lines_x[0],lines_x[2])) #1 - x
vanishing_points.append(np.cross(lines_y[0],lines_y[2])) #2 - y
vanishing_points.append(np.cross(lines_z[0],lines_z[2])) #3 - z

for i in range(len(vanishing_points)):
    for j in range(len(vanishing_points[i])):
        vanishing_points[i][j] = vanishing_points[i][j]/vanishing_points[i][-1]

print(vanishing_points,'vanishing_points')

# Defining reference points and world origin
ref_x = points[6]
ref_y = points[9]
ref_z = points[2]
World_Origin = points[7]

def distances(x1,x2):
    return [np.sqrt(math.pow(x1[0] - x2[0],2))+ np.sqrt(math.pow(x1[1] - x2[1],2)) ]

#Calculating refernce distance
ref_axis = []
ref_axis.append (distances(World_Origin,ref_x)[0]/2)#1-x
ref_axis.append (distances(World_Origin,ref_y)[0]/2)#2-y
ref_axis.append (distances(World_Origin,ref_z)[0]/2)#3-z

print(ref_axis,'ref_axis')

# Calculating Scaling Constants
x,resid_x,rank_x,s_x = np.linalg.lstsq(np.reshape( (vanishing_points[0] - ref_x).T,(3,-1)),np.reshape(np.array([a - b for a, b in zip(ref_x, World_Origin)]).T,(3,-1)))
y,resid_y,rank_y,s_y = np.linalg.lstsq(np.reshape( (vanishing_points[1] - ref_y).T,(3,-1)),np.reshape(np.array([a - b for a, b in zip(ref_y, World_Origin)]).T,(3,-1)))
z,resid_z,rank_z,s_z = np.linalg.lstsq(np.reshape( (vanishing_points[2] - ref_z).T,(3,-1)),np.reshape(np.array([a - b for a, b in zip(ref_z, World_Origin)]).T,(3,-1)))

# print(x,'x')
# print(y,'y')
# print(z,'z')

a_x = (x/ref_axis[0]).ravel()
a_y = (y/ref_axis[1]).ravel()
a_z = (z/ref_axis[2]).ravel()

print(a_x,'a_x')
print(a_y,'a_y')
print(a_z,'a_z')

#Calculating Projection Matrix
Projection_Matrix = [vanishing_points[0] * a_x , vanishing_points[1] * a_y, vanishing_points[2] * a_z, World_Origin]
Projection_Matrix = np.array(Projection_Matrix).T

print('Projection_Matrix')
print(Projection_Matrix)

#Calculation of Homography Matrix
hxy = Projection_Matrix[:,[0,1,3]]
hxz = Projection_Matrix[:,[0,2,3]]
hyz = Projection_Matrix[:,[1,2,3]]

hxy[0,2]=hxy[0,2] + 20
hxy[1,2]=hxy[1,2] + 20

hyz[0,2]=hyz[0,2] - 120
hyz[1,2]=hyz[1,2] + 20

hxz[0,2]=hxz[0,2] + 40
hxz[1,2]=hxz[1,2] - 100

print(hxy,'hxy')
print(hxz,'hxz')
print(hyz,'hyz')

#Appling a perspective transformation - XY plane  to original image.
img_dst_xy = cv2.warpPerspective(img, hxy, (img.shape[0], img.shape[1]), flags=cv2.WARP_INVERSE_MAP)
cv2.imwrite('hxy_fin.jpg',img_dst_xy)

#Appling a perspective transformation - YZ plane  to original image.
img_dst_yz = cv2.warpPerspective(img, hyz, (img.shape[0], img.shape[1]), flags=cv2.WARP_INVERSE_MAP)
cv2.imwrite('hyz_fin.jpg',img_dst_yz)

#Appling a perspective transformation - XZ plane  to original image.
img_dst_xz = cv2.warpPerspective(img, hxz, (img.shape[0], img.shape[1]), flags=cv2.WARP_INVERSE_MAP)
cv2.imwrite('hxz_fin.jpg',img_dst_xz)
