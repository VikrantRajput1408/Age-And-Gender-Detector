import cv2
import numpy 
import tensorflow as tf

faceProto = "trained__models/opencv_face_detector.pbtxt"
faceModel = "trained__models/opencv_face_detector_uint8.pb"

ageFinder = tf.keras.models.load_model("Models/age.h5")
GenderFinder = tf.keras.models.load_model("Models/gender.h5")

## Varibale 
pred_per_frame = 60
num_per_image = 3

def faceBox(faceNet, frame):
    height = frame.shape[0]
    width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    # print(detection)
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > 0.7:
            x1 = int(detection[0,0,i,3] * width)
            y1 = int(detection[0,0,i,4] * height)
            x2 = int(detection[0,0,i,5] * width)
            y2 = int(detection[0,0,i,6] * height)
            bboxs.append([x1,y1, x2, y2])
            cv2.rectangle(frame, (x1-5,y1-5), (x2+5,y2+5), (169,81,120), 5)
            cv2.rectangle(frame, (x1-8,y1-55), (x2+8,y1+5), (169,81,120), -1)
    return frame, bboxs


#========================== make for multiple =========================#

faceNet = cv2.dnn.readNet(faceModel, faceProto)

age = ["Age"] * num_per_image
gender = ["Gender"] * num_per_image
video = cv2.VideoCapture(0)
K = 1
pred_age = [0] * num_per_image
pred_gender = [0] * num_per_image
while True:
    ret, frame = video.read()
    frame, bboxs = faceBox(faceNet, frame)
    for i, (x1, y1, x2, y2) in enumerate(bboxs):
        img = frame[y1:y2, x1:x2 , 1]/255
        try:
            K += 1
            img = cv2.resize(img, (48,48), interpolation = cv2.INTER_AREA)
            img = img.reshape(1, 48, 48, 1)
            pred_age[i] += int(ageFinder.predict(img)[0])
            pred_gender[i] += GenderFinder.predict(img)[0][0]
            cv2.putText(frame, age[i]+", "+gender[i], (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        except:
            continue

    # print(pred_age, pred_gender)

    if K % pred_per_frame == 0:
        K = 0
        for j in range(num_per_image):
            age[j] = str(pred_age[j] // pred_per_frame)

            if pred_gender[j] > 25.0:
                gender[j] = 'female'
            elif 0.5 < pred_gender[j] <= 25.0 :
                gender[j] = 'male'
        pred_age = [0] * num_per_image
        pred_gender = [0] * num_per_image
        
    # print(age, gender)
    cv2.imshow("Video",frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    

video.release()
cv2.destroyAllWindows()