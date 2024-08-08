#!/usr/bin/env python3
from detect import *
import numpy as np
import rospy
from std_msgs.msg import String

#extract the x coordinates of the bounding boxes
def get_angles(boxes):
    counter=0
    x_distance=[]
    for i in boxes:
        counter=counter+1
    for i in range(counter):
        x_coords=boxes[i]
        x_distance.append(x_coords[0])
    
    # extracts the width of the bounding boxes
    widths=[]
    for j in range(counter):
        x_widths=boxes[j]
        widths.append(x_widths[2])
    
    # find all the x coordinate of the boxes(where there are obstacles)
    x_coords=[]
    for h in range(counter):
        x_coords.append(x_distance[h])
        x_coords.append(x_distance[h]+widths[h])
    
    # to set for camera We USE!!!!!!
    # degrees, pixels.. We use ZED CAMERA
    cam_angle_horizontal= 62.2 # SETTING!!
    image_width=3280 ##PIXEL
    cam_angle_rad=np.radians(cam_angle_horizontal/2)
    f=(image_width/2)/(np.sin(cam_angle_rad))
    x_angles=[]
    for k in range(counter*2):
        x=x_coords[k]
        if x<image_width/2:
            c=np.sqrt((x**2)+(f**2)-(2*x*f*np.cos(np.radians(90-(cam_angle_horizontal/2)))))
            angle_x=np.degrees(np.arcsin(((image_width/2)-x)/c))
            x_angles.append(-1*angle_x)
        else:
            c = np.sqrt(((image_width-x)**2)+(f**2)-(2*(image_width-x)*f*np.cos(np.radians(90-(cam_angle_horizontal/2)))))
            angle_x=np.degrees(np.arcsin(((x-(image_width/2))/c)))
            x_angles.append(angle_x)
    print(x_angles)

    #extracts the y coordinates of the bounding boxes
    y_distance=[]
    for i in range(counter):
        y_coords=boxes[i]
        y_distance.append(y_coords[i])
    #extracts the widths of the bounding boxes
    heights=[]
    for j in range(counter):
        y_heights=boxes[j]
        heights.append(y_heights[3])
        
    # find all the x coordinates of the boxes (where there are obstacles)
    y_coords=[]
    for h in range(counter):
        y_coords.append(y_distance[h])
        y_coords.append(y_distance[h]+heights[h])
    # to set for camera you use
    # degrees, pixels
    cam_angle_vertical=45.5 #degrees
    image_height=2464
    cam_angle_rad = np.radians(cam_angle_vertical/2)
    g = (image_height/2)/(np.sin(cam_angle_rad))
    y_angles=[]
    for k in range(counter*2):
        y=y_coords[k]
        if y<image_height/2:
            c=np.sqrt((y**2)+(g**2)-(2*y*g*np.cos(np.radians(90-(cam_angle_vertical/2)))))
            angle_y=np.degrees(np.arcsin(((image_height/2)-y)/c))
            y_angles.append(-1*angle_y)
        else:
            c=np.sqrt(((image_height-y)**2)+(g**2)-(2*(image_height-y)*g*np.cos(np.radians(90-(cam_angle_vertical/2)))))
            angle_y=np.degrees(np.arcsin(((y-(image_height/2))/c)))
            y_angles.append(angle_y)
    print(y_angles)
    
    # in a 16 degrees range of 0
    angle_orientation=0
    in_between_boxes=False
    if counter!=0:
        for l in range(counter+1):
            if l%2 ==0 and angle_orientation>x_angles[l] and angle_orientation < x_angles[l+1]:
                #find which direction to turn to
                between_boxes=True
                best_avoid = min((x_angles[l:l+2]), key=abs)
                for m in range(counter):
                    if best_avoid<x_angles[m+1] and best_avoid>x_angles[m]:
                        best_avoid=max((x_angles[l:l+2]), key=abs)
                    else:
                        for j in range(counter*2):
                            if best_avoid>x_angles[j]:
                                angle_orientation=best_avoid+8
                            else:
                                angle_orientation=best_avoid-8
                            return (angle_orientation)
            elif (abs(x_angles[l])-8):
                angle_orientation=0
            
            else:
                best_avoid=min((x_angles[l:l+2]), key=abs)
                for m in range(counter):
                    if best_avoid<x_angles[m+1] and best_avoid>x_angles[m]:
                        best_avoid=max((x_angles[l:l+2]), key=abs)
                    else: 
                        for j in range(counter*2):
                            if best_avoid>x_angles[j]:
                                angle_orientation=best_avoid+8
                            else: 
                                angle_orientation=best_avoid-8
                            return(angle_orientation)
        else:
            angle_orientation=0
            return(angle_orientation)
        
def direction_publisher(direction):
    pub=rospy.Publisher('avoidance_direction', String, queue_size=10)
    rospy.init_node('direction_publisher', anonymous=True)
    rate=rospy.Rate(20) # 20Hz
    while not rospy.is_shutdown():
        rospy.loginfo(direction)
        pub.publish(direction)
        rate.sleep()
if __name__=='__main__':
    try:
        boxes=[...]
        direction=get_angles(boxes)
        direction_publisher(direction)
    except rospy.ROSInternalException:
        pass