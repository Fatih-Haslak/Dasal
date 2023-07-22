###################################################################
# "dasal_rotation"
###################################################################
# @Description: "İki frame arasında ki özellikleri eşleyerek ve bu özellik noktalarını mesafeye göre sıralayarak 
#                %x lik bir kısımları arasında bir lineer doğru çıkarımı yapılmaktadır. Knn kümeleme yöntemi ile. 
#                Bu doğruların birbirine olan açıları ile iki frame arası dönme farkını yaklaşımsal olarak bulmaktadır."
#  @Note : "Framelerin resize değeri, noktaların % kaçlık kısımlarının alınacağının ve feature dedectior'un parametlerine
#           dikkat edilmedilir. Bu parametreler algoritmanın yaklaşımsal sonuç değerinde önemli yere sahiptir. "


#   Version 0.0.1:  "Feature Dedection ve İlgili alanların sınıflandırılması "
#                   ...
#                   02 AĞUSTOS 2022 Sal, 13:00 - "Fatih HAŞLAK,Mucahid KARAAGAC"
#   Version 0.0.2:  "İlgili alanların kordinat sisteminde nokta olarak belirtilip
#                    en uygun line bulunması ve tüm noktaların mın max'ında ki açı değerinin
#                    yaklaşıksal olarak hesaplanması"
#                   ...
#                   3 Ağustos 2022 Çar, 08:00 - "Fatih HAŞLAK,Mucahid KARAAGAC"

#   Version 0.0.3:  ""
#                   ...
#                    Ağustos 2022 , 08:30 - "Fatih HAŞLAK,Mucahid KARAAGAC"


import math
from numpy.linalg import lstsq
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import ransac,LineModelND
from skimage.transform import ProjectiveTransform, AffineTransform
import time
from shapely.geometry import Point #version 1.8.4 last version
from shapely.geometry import LineString
import shapely as sp
start=time.time()

#r"C:\Users\90546\Desktop\dasal_task\map_tile_matcing_with_camera\feature_matching\images_deneme\10_metre_saat_yönü_drone_sabit1.jpg",
sayac=0
n_samples=0

while(sayac<=(1285-30)): 
  #filename ="10_metre_saat_yonu_dasal" +str(sayac)+ ".jpg" #framelerin hangi isimlendirmeyle belirleneceğini yazın str(n_samples)          31+n_samples
  name1="C:/Users/90546/Desktop/dasal_task/map_tile_matcing_with_camera/feature_matching/images_deneme/100_metre_saat_yonu_drone_sabit/100_metre_saat_yonu_drone_sabit"+str(31+n_samples)+".jpg"
  name2="C:/Users/90546/Desktop/dasal_task/map_tile_matcing_with_camera/feature_matching/images_deneme/100_metre_saat_yonu_drone_sabit/100_metre_saat_yonu_drone_sabit"+str(61+n_samples)+".jpg"
  n_samples+=10
  sayac+=1
  chos = cv2.imread(name1,1)
  #chos = cv2.resize(chos,(400,400))
  chos=chos[240:840,660:1260]
  cho = cv2.imread(name2, 1)
  cho=cho[240:840,660:1260]

  #cho = cv2.resize(cho,(400,400))

  deg=0 #kaç derece döndüreceksin
  deg_1=0


  cho = imutils.rotate(cho,deg)
  chos = imutils.rotate(chos,deg_1)

  cv2.imshow("1",chos)
  cv2.imshow("2",cho)
    
  print("Shape of İmages",chos.shape,cho.shape)
  print("Input Degree of frame 1 and 2 : ",deg,deg_1)

  chos_d=chos.copy()
  cho_d=cho.copy()


  dedector = cv2.ORB_create(nfeatures = 10000) #Orb feature dedector olusturucu
  descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
  kpts1 = dedector.detect(chos, None)
  kpts2 = dedector.detect(cho, None)
  kp1, des1 = descriptor.compute(chos, kpts1)
  kp2, des2 = descriptor.compute(cho, kpts2)

  bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True) #brute force matcher ile noktaları eşleştirme
  matches = bf.match(des1, des2) #eşlenmiş noktalar


  good_matches=[]

  good_matches = sorted(matches, key = lambda x: x.distance) #mesafeye göre sırala
  good_matches=good_matches[:int(len(good_matches)/2)]
  src_pts = np.float32([ kp1[match.queryIdx].pt for match in good_matches ] ).reshape(-1, 2) #source image pointler x,y
  dst_pts = np.float32([ kp2[match.trainIdx].pt  for match in good_matches ] ).reshape(-1, 2) #hedef image pointler x1,y1


  img_match = cv2.drawMatches(chos, kp1, cho, kp2, good_matches[:], None, flags = 2)
  cv2.imshow("ilk",img_match)

  print("SRC_pts",len(src_pts))
  print("DST_pts",len(dst_pts))

  list_p1=[]
  list_p2=[]

  for point in src_pts: #source pointlerimin inlier noktalarını al ve list_p1'e at.
    list_p1.append((point[0],point[1]))
  for point in dst_pts:  #hedef pointlerimin inlier noktalarını al ve list_p2'e at.
    list_p2.append((point[0],point[1]))
  

  lenght=int(len(list_p1))
  if(lenght<50):
    print("Eşleşme yok")
    #exit()

  print("Data uzunluğum point sayisi",lenght)


  point_cloud_data=lenght #bu datamın yüzde x ini kadar al

  point_p1=list_p1[0:][0:point_cloud_data] #point p1 in içine at ve arraye çevir
  point_p2=list_p2[0:][0:point_cloud_data] #point p1 in içine at ve arraye çevir

  point_p1=np.array(point_p1).astype(np.float64)
  point_p2=np.array(point_p2).astype(np.float64)

  print("Point_cloud_data_len :",point_cloud_data)

  x=point_p1[0:,0] # 1.fonksiyonun x datası
  y=600-point_p1[0:,1] # 1.fonskyionun y datası (İNTvalues=shape)

  x1=point_p2[0:,0] # 2.fonksiyonun x datası
  y1=600-point_p2[0:,1] #2.fonksyıonun y datası

  fig1, ax1= plt.subplots()

  plt.scatter(x, y,color="red")
  plt.scatter(x1, y1,color="blue")

  fig, ax = plt.subplots()

  ax.set(xlim=(0, 600), xticks=np.arange(0,0),
        ylim=(0, 600), yticks=np.arange(0,0))
        

  data =np.column_stack([x, y])
  data2 = np.column_stack([x1, y1])


  #fine tunıng yapılacak
  model_robust, inliers,_ = ransac(data, LineModelND, min_samples=int(len(data)*0.5), 
                                residual_threshold=40  , max_trials=1000,random_state=42,stop_probability=1)

  
  model_robust2, inliers2,_s = ransac(data2, LineModelND, min_samples=int(len(data)*0.5),
                                residual_threshold=40, max_trials=1000,random_state=42,stop_probability=1)


  line_y_robust = model_robust.predict_y(x)


  line_y_robust2 = model_robust2.predict_y(x1)


  #doğru denklemleri
  #bana doğrunun o y eksenınde ki x değeri gerekli yani doğrunun ben o noktadak i y değerinin bilmekteyim x i lazım


  #line_y_robust

  plt.scatter(x[inliers], y[inliers])
  plt.scatter(x1[inliers2], y1[inliers2])
  depo=[]

  indeks=np.where(inliers==True)
  indeks2=np.where(inliers2==True)
  indeks=indeks[0]
  indeks2=indeks2[0]

  #print("İndeks 1 ",indeks)
  print(" ")
  #print("indeks 2" ,indeks2)

  count=0

  for i in indeks:
    for a in indeks2:
      if(i==a):
        count+=1
        depo.append(i)

  print("1.data uzunluk",len(inliers))
  print("2.data uzunluk ",len(inliers2))
   
  A = np.vstack([x, np.ones(len(x))]).T
  m, c = np.linalg.lstsq(A, line_y_robust, rcond=None)[0]
  #plt.plot(x, m*x + c, 'r', label='Fitted line')
  # denklem 1    m*x+c-y=0
  A = np.vstack([x1, np.ones(len(x1))]).T
  m1, c1 = np.linalg.lstsq(A, line_y_robust2, rcond=None)[0]
  
  ### baslangıc 1.line

  point = Point(x[depo[0]], y[depo[0]])
  
  dist =LineString( [ (min(x),line_y_robust[np.argmin(x)]),(max(x),line_y_robust[np.argmax(x)]) ]).project(point)
  
  print("dist",dist)
  
  baslangic=list(LineString( [(min(x),line_y_robust[np.argmin(x)]),(max(x),line_y_robust[np.argmax(x)]) ]).interpolate(dist).coords)

  ### bitiş 1.line
  point2=Point(x[depo[-1]], y[depo[-1]])
  
  dist_2 = LineString( [ (min(x),line_y_robust[np.argmin(x)]),(max(x),line_y_robust[np.argmax(x)]) ]  ).project(point2)
 
  bitis=list(LineString( [(min(x),line_y_robust[np.argmin(x)]),(max(x),line_y_robust[np.argmax(x)]) ]).interpolate(dist_2).coords)
  ######
  # noktanın eğimi
  slope=((baslangic[0][1]) - y[depo[0]])/(baslangic[0][0]-x[depo[0]])
  slope2=((bitis[0][1]) - y[depo[-1]])/(bitis[0][0]-x[depo[-1]])
  print("eğim",slope*m,slope2*m)


  # 
  ### baslangıc 2.line
  point3=Point(x1[depo[0]], y1[depo[0]])
  dist_3 = LineString( [ (min(x1),line_y_robust2[np.argmin(x1)]),(max(x1),line_y_robust2[np.argmax(x1)])  ]).project(point3)
  baslangic_1=list(LineString( [(min(x1),line_y_robust2[np.argmin(x1)]),(max(x1),line_y_robust2[np.argmax(x1)]) ]).interpolate(dist_3).coords)
  
  ###### bitis 2.line
  point4=Point(x1[depo[-1]], y1[depo[-1]])
  dist_4 = LineString( [ (min(x1),line_y_robust2[np.argmin(x1)]),(max(x1),line_y_robust2[np.argmax(x1)])  ]).project(point4)
  bitiş_1=list(LineString( [(min(x1),line_y_robust2[np.argmin(x1)]),(max(x1),line_y_robust2[np.argmax(x1)]) ]).interpolate(dist_4).coords)
  ######
  slope=((baslangic_1[0][1]) - y1[depo[0]])/(baslangic_1[0][0]-x1[depo[0]])
  slope2=((bitiş_1[0][1]) - y1[depo[-1]])/(bitiş_1[0][0]-x1[depo[-1]])
  print("eğim2",slope*m1,slope2*m1)
  
  print("Baslangic 1.Line",baslangic[0])
  print("Bitiş 1.Line",bitis[0])
  print(" ")
  print("Baslangic 2.Line ",baslangic_1[0])
  print("Bitiş 2.line",bitiş_1[0])


  if(len(depo)<2):
    nokta_1=(x[0],line_y_robust[0])
    nokta_2=(x[1],line_y_robust[1])
    vector1=np.array([nokta_1,nokta_2])
    print("Başlangic1,Bitis1",nokta_1,nokta_2)

    nokta_3=(x1[0],line_y_robust2[0])
    nokta_4=(x1[1],line_y_robust2[1])
    vector2=np.array([nokta_3,nokta_4])
    print("Başlangic 2, Bitiş 2",nokta_3,nokta_4)
    print("UYARIIII EŞLESEN CIKAMADI")
    # x leri y leri cıkar kenara koy

  else: 
    nokta_1=baslangic[0]
    nokta_2=bitis[0]
    vector1=np.array([nokta_1,nokta_2])
    print("Başlangic1,Bitis1",nokta_1,nokta_2)

    nokta_3=baslangic_1[0]
    nokta_4=bitiş_1[0]
    vector2=np.array([nokta_3,nokta_4])
    print("Başlangic 2, Bitiş 2",nokta_3,nokta_4)


     
  ax.plot(x, line_y_robust, color="black")
  plt.scatter(x[inliers],y[inliers],color="red")
  plt.scatter(x1[inliers2],y1[inliers2],color="blue")
  ax.plot(x1, line_y_robust2,color="green")

  plt.scatter(
      vector2[0,0],vector2[0,1],
      s=700, marker='*',
      c='yellow', 

      )

  plt.scatter(
      vector2[1,0],vector2[1,1],
      s=700, marker='*',
      c='black'

      )

  plt.scatter(
      vector1[0,0],vector1[0,1],
      s=700, marker='+',         
      c='yellow'

      )

  plt.scatter(
      vector1[1,0],vector1[1,1],
      s=700, marker='+',
      c='black'

      )
  plt.scatter(x[depo[0]],y[depo[0]],s=250,marker="+",c="black")
  plt.scatter(x[depo[-1]],y[depo[-1]],s=250,marker="*",c="black")
  plt.scatter(x1[depo[0]],y1[depo[0]],s=250,marker="+",c="green")
  plt.scatter(x1[depo[-1]],y1[depo[-1]],s=250,marker="*",c="green")

  print()

  def quadrant(va,degree):
    if(va[0]>0 and va[1]>0):#bölge 1
      return degree
    elif(va[0]>0 and va[1]<0): #bölge 4
      return degree+360
    else:
      return degree+180 #bölge 3


  def ang(point_p1, point_p2):
      # Get nicer vector form
            #inital point x   end point x          inital point y  end point y
      vA = [(point_p1[0,0]-point_p1[1,0]), (point_p1[0,1]-point_p1[1,1])] #(p1 ilk x - p1 son x), (p1 ilk y p1 son y)
      
      #vA type=list  #örnek [31, -98] Rows: 2 
      vB = [(point_p2[0,0]-point_p2[1,0]), (point_p2[0,1]-point_p2[1,1])]  #(p2 ilk x - p2 son x), (p2 ilk y p2 son y)
      
      print("vector A",vA)
      print("vector B",vB)
      
      ilk=math.degrees(math.atan(vA[1]/vA[0])) #karşı bölü komşu
      son=math.degrees(math.atan(vB[1]/vB[0]))
      print("ilk degree",quadrant(vA,ilk))
      print("son degree",quadrant(vB,son))
      degg = (quadrant(vA,ilk) - quadrant(vB,son) ) #quadrant ile bölge tahmini yap ve dereceyi ayarla
      
      if(degg<0):
        degg += 360
        print("kuçuk then 0")
    
      return degg 

  deger=ang(vector1,vector2) #fonksiyonu cagır
  print("Degree of real",deger)
  print("{}. ve {}. resimler ".format(abs(n_samples+21),abs(n_samples+51)))

  counter=0
  for i in point_p1:
    image = cv2.circle(chos_d, (int(point_p1[counter,0]),int(point_p1[counter,1])), 5, (250,250,146), -1)
    counter+=1
    if(counter==point_cloud_data):
      break
  
  counter=0
  for i in point_p2:
    image1 = cv2.circle(cho_d, (int(point_p2[counter,0]),int(point_p2[counter,1])), 5, (250,0,146), -1)
    counter+=1
    if(counter==point_cloud_data):
      break
  end=time.time()



  inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]] # new source points
  inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers2]]# old source points
  placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(len(depo))]
  image3 = cv2.drawMatches(chos, inlier_keypoints_left, cho, inlier_keypoints_right,placeholder_matches, None,flags=2)
  plt.show()
  cv2.imshow("Circle_Image",image)
  cv2.imshow("Circle2_Image",image1)
  cv2.imshow("Circle22_Image",image3)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.waitKey(0)
    cv2.destroyAllWindows()
