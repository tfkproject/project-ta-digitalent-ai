import cv2
import numpy as np

lowerRed = np.array([0,100,100], dtype = "uint8")
upperRed = np.array([20,255,255], dtype = "uint8")

lowerGreen = np.array([33,80,40], dtype = "uint8")
upperGreen = np.array([102,255,255], dtype = "uint8")

video_capture= cv2.VideoCapture(0)
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (0, 255, 255)

# KNN
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('../dataset1.csv')
attributes = ['a','b','c','d','e','f','g','h','i','j','k','l','m']
df.columns = attributes
df.head()
X_Trainb = (df.values[:,0:11]) #mengambil feature pada data set
Y_Train = (df.values[:,12]) #mengambil label pada data set
from sklearn import preprocessing
X_Traint = preprocessing.MinMaxScaler(feature_range=(0, 3))
X_Train = X_Traint.fit_transform(X_Trainb)
from sklearn.model_selection import train_test_split #(KALO MAU VALIDASI)
X_training, X_testing, y_training, y_testing = train_test_split(X_Train, Y_Train, test_size=0.3) # 70% training and 30% test
#VALIDASI
from sklearn.neighbors import KNeighborsClassifier
knnVal = KNeighborsClassifier(n_neighbors=7)
knnVal.fit(X_training, y_training)
y_pred = knnVal.predict(X_testing)
from sklearn import metrics
# Akurasi dataset
print("Classification report for classifier %s:\n%s\n" 
     % (knnVal, metrics.classification_report(y_testing, y_pred)))
print("Confusion matrix:\n%s" %metrics.confusion_matrix(y_testing,y_pred))
print("Accuracy:",metrics.accuracy_score(y_testing, y_pred))
# membuat model
knnModel =  KNeighborsClassifier(n_neighbors=7)
knnModel.fit(X_Train, Y_Train)
# End KNN

while True:
    ret, img=video_capture.read()

    #convert BGR to HSV
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # buat Mask untuk warna merah
    mask_merah=cv2.inRange(imgHSV,lowerRed,upperRed)
    # buat Mask untuk warna hijau
    mask_hijau=cv2.inRange(imgHSV,lowerGreen,upperGreen)
    #morphology merah
    maskOpen_merah=cv2.morphologyEx(mask_merah,cv2.MORPH_OPEN,kernelOpen)
    maskClose_merah=cv2.morphologyEx(maskOpen_merah,cv2.MORPH_CLOSE,kernelClose)
    #morphology hijau
    maskOpen_hijau=cv2.morphologyEx(mask_hijau,cv2.MORPH_OPEN,kernelOpen)
    maskClose_hijau=cv2.morphologyEx(maskOpen_hijau,cv2.MORPH_CLOSE,kernelClose)

    maskFinal_merah=maskClose_merah
    maskFinal_hijau=maskClose_hijau
    im1,conts1,h1 = cv2.findContours(maskFinal_merah.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    im2,conts2,h2 = cv2.findContours(maskFinal_hijau.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    # draw object yang warna merah
    cv2.drawContours(img,conts1,-1,(255,0,0),3)
    # draw object yang warna hijau
    cv2.drawContours(img,conts2,-1,(0,255,0),3)

    # ubah mask (yang mana hitam putih) kembali ke warna 
    merah_saja = cv2.bitwise_and(img, img, mask = mask_merah)
    hujau_saja = cv2.bitwise_and(img, img, mask = mask_hijau)

    # ambil nilai max ny saja (artinya nilai yang hanya ada warna nya saja)
    out_merah = np.max(merah_saja, 0)
    out_hijau = np.max(hujau_saja, 0)
    # buang nilai 0 yang artinya pixel hitam, karna kita hanya butuh yang ada warna
    value_merah = out_merah[~(out_merah==0).all(1)]
    value_hijau = out_hijau[~(out_hijau==0).all(1)]
    # ambil nilai paling rendah
    min_red = np.min(value_merah, 0)
    min_green = np.min(value_hijau, 0)
    # ambil nilai paling tinggi
    max_red = np.max(value_merah, 0)
    max_green = np.max(value_hijau, 0)

    a,b,c = min_red
    d,e,f = min_green
    g,h,i = max_red
    j,k,l = max_green
    
    # lanjut ke input/pembanding knn
    data_baru = np.array([a,b,c,d,e,f,g,h,i,j,l])
    data_baru = data_baru.reshape(1, -1)
    minmax = preprocessing.MinMaxScaler(feature_range=(0, 3))
    data_live = minmax.fit_transform(data_baru)
    prediksi = knnModel.predict(data_live)

    # block object yang warna merah
    for i in range(len(conts1)):
        x,y,w,h=cv2.boundingRect(conts1[i])
        cv2.putText(img, str(prediksi),(x,y+h),fontFace, fontScale, fontColor)
    
    cv2.imshow("Output",img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
