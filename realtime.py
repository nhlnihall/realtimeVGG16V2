import numpy as np
import cv2
import fps
import caffe

caffe.set_mode_cpu()

age_net_pretrained='./age_net.caffemodel'
age_net_model_file='./deploy_age.prototxt'

gender_net_pretrained='./gender_net.caffemodel'
gender_net_model_file='./deploy_gender.prototxt'

#age_list=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100']
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Female','Male']

age_net = cv2.dnn.readNetFromCaffe(age_net_model_file, age_net_pretrained)
gender_net = cv2.dnn.readNetFromCaffe(gender_net_model_file, gender_net_pretrained)

#mean_filename='./mean.binaryproto'
#proto_data = open(mean_filename, "rb").read()
#a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
#mean= caffe.io.blobproto_to_array(a)[0]

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')

fpsWithTick = fps.fpsWithTick()

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #print faces
    i=0
    for (x, y, w, h) in faces:
        i=i+1

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #roi_gray = gray[y:y + h, x:x + w]
        cropped_img = img[y:y + h, x:x + w]
        #cv2.imshow('face', cropped_img)

        blob = cv2.dnn.blobFromImage(cropped_img, scalefactor=1.0, size=(256, 256), swapRB=False)
        age_net.setInput(blob)
        detections = age_net.forward()

        print 'predicted age:{}'.format(i), age_list[detections.argmax()]

        # detections = age_net.blobs['data'].data[...]
        #age = np.dot(detections, np.arange(0, 101))
        #print 'predicted age:{}'.format(i),round(age)


        gender_net.setInput(blob)
        predict = gender_net.forward()

        print 'predicted gender:{}'.format(i), gender_list[predict[0].argmax()]
        #cv2.putText(img, age, (x-20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    fps_output = str(fpsWithTick.get())
    cv2.putText(img, "fps = " + fps_output, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    # Display the resulting frame
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()