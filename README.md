# 手勢辨識

手勢辨識0~5，資料部份使用opencv擷取自己的二值化手勢，再利用資料擴增法方式來增加數據

## training
訓練時需查看顯卡佔存以免佔存爆滿或是降低batch_size，請注意降低batch_size會影響神經網路學習特徵會影響準確度

### resize與手勢標籤
resize為256*256
training_data存放data與class映射

    def create_training_data():
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            for img in tqdm(os.listdir(path)):
                try:
                    img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE) 
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                    training_data.append([new_array, class_num]) 
                except Exception as e:  
                    pass
### 數據洗牌
避免數據集中一起進行洗牌以免影響到神經網路學習
    import random

    random.shuffle(training_data)

    X = []
    y = []
    for features,label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    
### 轉換成one-hot encoding 
    from sklearn.preprocessing import OneHotEncoder
    values = np.array(y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = y.reshape(len(y), 1)
    y = onehot_encoder.fit_transform(integer_encoded)
    
    
### test、train 比例
設定traing 與 test為8:2

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=0.2, random_state=1)
        
        
### 查看資料label與手勢資料
        
    for X,y in test_gen:
    print(X.shape, y.shape)
    
    plt.figure(figsize=(16,16))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.axis('off')
        plt.title('Label: %d' % np.argmax(y[i]))
        img = np.uint8(255*X[i,:,:,0])
        plt.imshow(img)
    break
  
### 計算準確度
準確度還算不錯高達98%因為訓練資料是二值化圖片，在手勢無奇怪姿勢下訓練效果都不錯

      y_pred = np.int32([np.argmax(r) for r in model.predict(X_test)])
      y_test = np.int32([np.argmax(r) for r in y_test])
      match = (y_test == y_pred)
      print(np.sum(match)*100/match.shape[0])
    
## pred.ipynb

### prection 資料
    from keras.models import load_model
    import cv2

    model = load_model('handmodel.h6')

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    img = cv2.imread('C:\\Users\\tomto\\Desktop\\hand1\\data\\FIVE\\FIVE_13.png')
    img = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    img = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    classes = model.predict_classes(img)
    print(img)
    print(classes)
    
    
    
## opencv_pred.ipynb

### 結合opencv調用模型進行手勢辨識
trackbar可設定threshold數值來調整最好的數據給模型進行高準確度辨識

removeBG前後景分割減少雜訊

    def removeBG(frame):
        fgmask = bgModel.apply(frame, learningRate=learningRate)
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        return res
        
### 數據處理
把thresh做resize處裡辨識後用putText顯示結果

        thresh = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE)).tolist()
        thresh = np.array(thresh, dtype='int8').reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        CATEGORIES = ["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
        pred = CATEGORIES[np.argmax(model.predict(thresh))]

        cv2.putText(frame,str(pred),(0,40),cv2.FONT_HERSHEY_COMPLEX,1,(0, 255, 0),1)
        
        
 ### 辨識展示
 https://youtu.be/zTjkvJQfy5s
 
 ## 膚色偵測ycrcb
 前後景分割法可能會受到光線影響，因此可用HSV手部趨勢來取得降低光線影響
 
     #膚色偵測ycrcb(動態)
    from keras.models import load_model
    import numpy as np
    from sklearn.model_selection import train_test_split
    %matplotlib inline
    import matplotlib.pyplot as plt
    import cv2

    # load model
    model = load_model('model_finger_2.h6')
    # summarize model.
    model.summary()



    width, height = 300, 300 #設置拍攝窗口大小
    x0,y0 = 300, 100 #設置選取位置

    cap = cv2.VideoCapture(0) #開攝像頭
    cap.set(10, 200)  

    threshold = 60  # 二值化阈值

        #以3*3的模板進行均值濾波

    def binaryMask(frame, x0, y0, width, height):
        cv2.rectangle(frame,(x0,y0),(x0+width, y0+height),(0,255,0)) #畫出截取的手勢框圖
        roi = frame[y0:y0+height, x0:x0+width] #獲取手勢框圖
        res,skin = skinMask(roi) #進行膚色檢測

        cv2.imshow('skin',skin)#二值化

        kernel = np.ones((2,2), np.uint8) #設置卷積核
        erosion = cv2.erode(res, kernel) #腐蝕操作

        IMG_SIZE = 128


        skin = cv2.resize(skin, (IMG_SIZE, IMG_SIZE)).tolist()
        skin = np.array(skin, dtype='int8').reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        CATEGORIES = ["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
        pred = CATEGORIES[np.argmax(model.predict(skin))]

        cv2.putText(frame,str(pred),(0,40),cv2.FONT_HERSHEY_COMPLEX,1,(0, 255, 0),1)



        dilation = cv2.dilate(erosion, kernel)#膨脹操作
        return res

    ##########方法一###################
    ##########BGR空間的手勢識別#########
    def skinMask(roi):
        YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #轉換至YCrCb空間

        (y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
        cr1 = cv2.GaussianBlur(cr, (5,5), 0)
        _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Ostu處理
        res = cv2.bitwise_and(roi,roi, mask = skin)
        return res,skin

    if __name__ == "__main__":
        while(1):
            ret, frame = cap.read() #讀取攝像頭的內容
            frame = cv2.flip(frame, 2)

            cv2.putText(frame," ",(0,40),cv2.FONT_HERSHEY_COMPLEX,1,(0, 255, 0),1)

            binaryMask(frame, x0, y0, width, height) #取手勢所在框圖並進行處理

            cv2.imshow('pred', frame)  

            key = cv2.waitKey(1) & 0xFF#按鍵判斷並進行一定的調整
            #按'j''l''u''j'分別將選框左移，右移，上移，下移
            #按'q'鍵退出錄像
            if key == ord('i'):
                y0 += 5
            elif key == ord('k'):
                y0 -= 5
            elif key == ord('l'):
                x0 += 5
            elif key == ord('j'):
                x0 -= 5
            if key == ord('q'):
                break
            cv2.imshow('frame', frame) #播放攝像頭的內容
        cap.release()
        cv2.destroyAllWindows() #關閉所有窗口
 ### 辨識展示
 https://youtu.be/o9Pyc0tuJOQ
 
 ## 前後景去除找尋最大區域
前後景分割會受光線影響而產生雜訊降低辨識結果或是辨識錯誤，目前已知手部面積是最大的無庸置疑所以可以找尋最大面積來進行雜訊的處理，目前三種方法此方法效果是最好的

二值化原圖
![near](https://github.com/JED-4a6g0109/gesture/blob/main/Original_image.jpg)  

找尋最大面積去除雜訊
![near](https://github.com/JED-4a6g0109/gesture/blob/main/Max_Area.jpg)  


    from keras.models import load_model
    import time
    import numpy as np
    from sklearn.model_selection import train_test_split
    import cv2
    %matplotlib inline
    import matplotlib.pyplot as plt

    model = load_model('model_finger_2.h6')

    cap_region_x_begin = 0.5  
    cap_region_y_end = 0.8
    threshold = 60  
    blurValue = 41  
    bgSubThreshold = 100
    learningRate = 0
    isBgCaptured = 0
    triggerSwitch = False
    image_count=0
    pred=''

    def printThreshold(thr):
        print("! Changed threshold to " + str(thr))


    def removeBG(frame):
        fgmask = bgModel.apply(frame, learningRate=learningRate)
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        return res

    camera = cv2.VideoCapture(0)   
    camera.set(10, 200)  
    cv2.namedWindow('trackbar') 
    cv2.resizeWindow("trackbar", 640, 200)  
    cv2.createTrackbar('threshold', 'trackbar', threshold, 100, printThreshold)

    while camera.isOpened():
        ret, frame = camera.read()
        threshold = cv2.getTrackbarPos('threshold', 'trackbar') 
        # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2YCrCb)
        frame = cv2.bilateralFilter(frame, 5, 50, 100) 
        frame = cv2.flip(frame, 1) 
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),(frame.shape[1], int(cap_region_y_end * frame.shape[0])), (0, 0, 255), 2)
        cv2.putText(frame," ",(0,40),cv2.FONT_HERSHEY_COMPLEX,1,(0, 255, 0),1)

        if isBgCaptured == 1:
            img = removeBG(frame) 
            img = img[0:int(cap_region_y_end * frame.shape[0]),int(cap_region_x_begin * frame.shape[1]):frame.shape[1]] 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)       
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
            if cv2.waitKey(33) == ord('a'):
                name="image"+str(image_count)+".png"
                cv2.imwrite(name, thresh)
                print(name)
                image_count+=1

                img = cv2.imread(name)
                binarization = np.zeros(img.shape[:2],np.uint8)

                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

                plt.imshow(binary)
                plt.show()

                contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                area = [(contour, cv2.contourArea(contour)) for contour in contours]
                max_contour = max(area, key=lambda x: x[1])[0]

                cv2.drawContours(binarization, [max_contour], -1, (255,255,0), -1)

                cv2.imwrite(name, binarization)
                plt.imshow(binarization)
                plt.show()
                load_img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
                print(name)
                IMG_SIZE = 128
                load_img = cv2.resize(load_img, (IMG_SIZE, IMG_SIZE)).tolist()
                load_img = np.array(load_img, dtype='int8').reshape(-1, IMG_SIZE, IMG_SIZE, 1)
                CATEGORIES = ["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
                pred = CATEGORIES[np.argmax(model.predict(load_img))]
                print(pred)
        cv2.putText(frame,str(pred),(0,40),cv2.FONT_HERSHEY_COMPLEX,1,(0, 255, 0),1)    
        cv2.imshow('original', frame)
        k = cv2.waitKey(10)

        if k == 27:
            break
        elif k == ord('b'):
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold,False)
            isBgCaptured = 1
            print('切換')
            
            
  ### 辨識展示
  
  https://www.youtube.com/watch?v=28jCay5YX6U&feature=youtu.be&ab_channel=%E6%B4%AA%E5%B4%87%E6%81%A9
