import cv2
import numpy as np

# set confidence level
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = 'Yolo/coco.names'
classNames = []
# extract the objects name to list
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


modelConfiguration = 'Yolo/yolov3.cfg'
modelWeights = 'Yolo/yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]* wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))


    # localize object
    indices = cv2.dnn.NMSBoxes(bbox, confs,confThreshold,nmsThreshold)

    for i in indices:
        # i = i[0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]

        # find position
        loc=""
        if x>=320:
            loc="on right"
        elif x+w>320:
            loc="on mid"
        else:
            loc="on left"

        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),1)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}% {loc}',
                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,255),2)


