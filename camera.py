from yolo import *
import cv2

whT = 320

cap = cv2.VideoCapture(0)
def captureAndFind():
    while True:
        success, img = cap.read()

        # draw a line in middle
        cv2.line(img,(int(img.shape[1]/2),0),(int(img.shape[1]/2),int(img.shape[0])),(0,0,0),1)

        blob = cv2.dnn.blobFromImage(img, 1/255,(whT,whT),[0,0,0],crop=False)
        net.setInput(blob)

        layerNames = net.getLayerNames()
        outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(outputNames)
        findObjects(outputs,img)

        cv2.imshow('Image', img)
        if cv2.waitKey(1)==ord('e'):
            break

    cv2.destroyAllWindows()


