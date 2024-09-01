import cv2

def facebox(facenet,frame):
    f_heigh=frame.shape[0]
    f_width=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame,1.0,(227,227),[104,117,123],swapRB=False)
    facenet.setInput(blob)
    detection=facenet.forward()
    # loop to the predicted value
    # and pass to the boundry box
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*f_width)
            y1=int(detection[0,0,i,4]*f_heigh)
            x2=int(detection[0,0,i,5]*f_width)
            y2=int(detection[0,0,i,6]*f_heigh)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
    return frame,bboxs

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"

genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

 # we are giving command to deep neu. net. to read the pre-trained 
facenet=cv2.dnn.readNet(faceModel,faceProto)

gendernet=cv2.dnn.readNet(genderModel,genderProto) 

# Categories of distribution
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# agelist = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
#       '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderlist = ['Male', 'Female']

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    frame,bboxs=facebox(facenet,frame)
    for bbox in bboxs:
        face=frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        blob=cv2.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB=False)
        
        gendernet.setInput(blob)
        gender_pred=gendernet.forward()
        gender=genderlist[gender_pred[0].argmax()]

     

        label="{}".format(gender)
        cv2.putText(frame,label,(bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow("Age_gender",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    

# first we need to read our model 
# create a blob
# put the blob in side input
# 
# create a rectangle upon our boundry boxes