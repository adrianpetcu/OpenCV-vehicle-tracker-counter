import cv2
import datetime
import paho.mqtt.client as mqtt
from tracker import *

tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("rtsp://172.28.95.99:8554/stream") #match the local highway stream
mqtt_client = mqtt.Client("detector")
mqtt_client.connect("127.0.0.1") #connect to local mqtt

object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold = 30)
count = 0
count2 = 0
counted_id=[]
while True:
    ret, frame = cap.read()
    
    y=455
    x=950
    h=950
    w=950
    
    if ret == True:
        roi = frame[y:y+h, x:x+w].copy() #take the middle lane of the road
        
        mask = object_detector.apply(roi)
        _,mask = cv2.threshold(mask,254,255, cv2.THRESH_BINARY)
        contours,_ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >5000:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(roi,(x,y),(x+w,y+h), (0,255,0),3)
                detections.append([x,y,w,h])
                
        cv2.line(roi,(0,350),(950,350),(0,0,255),1)
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x,y,w,h,ids= box_id
            centerx = round(x+w/2)
            centery = round(y+h/2)
            cv2.circle(roi,(centerx,centery),2,(0,0,255),5)
            if centerx > 212 and centerx < 252:
                if ids not in counted_id:
                    mqtt_client.publish(str(datetime.datetime.now()) + ":new_car")    #publish basic new car event
                    count2 +=1
                    counted_id.append(ids)
            cv2.line(roi,(0,350),(950,350),(0,0,255),2)
            count = ids
            
        
        cv2.putText(roi,str(count2),(10,25), cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        cv2.imshow("ROI", roi)
        cv2.imshow("Feed", frame)
        cv2.imshow("Mask",mask)

        key = cv2.waitKey(40)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
