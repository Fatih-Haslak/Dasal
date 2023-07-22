
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
#                   02 AĞUSTOS 2022 Sal, 13:00 - "Fatih HAŞLAK"

#   Version 0.0.2:  "İlgili alanların kordinat sisteminde nokta olarak belirtilip
#                    en uygun line bulunması ve tüm noktaların mın max'ında ki açı değerinin
#                    yaklaşıksal olarak hesaplanması"
#                   ...
#                   3 Ağustos 2022 Çar, 08:00 - "Fatih HAŞLAK"

#   Version 0.0.3:  ""
#                   ...
#                    Ağustos 2022 , 08:30 - "Fatih HAŞLAK"


import math
from numpy.linalg import lstsq
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import time

fig, ax = plt.subplots() #plotun şeklini ayarlama

ax.set(xlim=(0, 400), xticks=np.arange(1, 1),
       ylim=(0, 400), yticks=np.arange(1, 1))


chos = cv2.imread("nestle.jpg", 1)
chos = cv2.resize(chos,(400,400))

cho = cv2.imread("nestle.jpg", 1)
cho = cv2.resize(cho,(400,400))

deg=0 #kaç derece döndüreceksin
deg_1=0
chop=cho.copy()
cho = imutils.rotate(cho,deg)
chos = imutils.rotate(chos,deg_1)

if deg not in [0, 90, 180, 270, -90, -180, -270]:
            
  shape = (chop.shape[0], chop.shape[1])
  cho = cv2.resize(cho, shape, interpolation=cv2.INTER_AREA)
   

print("Input Degree of frame 1 and 2 : ",deg,deg_1)

chos_d=chos.copy()
cho_d=cho.copy()

sift = cv2.xfeatures2d.SIFT_create() #Orb feature dedector olusturucu
kp1, des1 = sift.detectAndCompute(chos, None)
kp2, des2 = sift.detectAndCompute(cho, None)
FLAN_INDEX_KDTREE = 0
index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
search_params = dict (checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch (des1, des2,k=2)[0]



good_matches=[]
uz=int(len(matches)/2) #toplam eşlenmiş nokta sayısının yarısı
matches1=matches[0::2] #karşılatırmak icin tek ve çift olarak böldüm index olarak #0 2 4 6
matches2=matches[1::2]                                                            #1 3 5 7

good_matches = sorted(matches, key = lambda x: x.distance) #mesafeye göre sırala

#print(len(good_matches)) #eşlesen toplam nokta
if(len(good_matches)==0):
  print("Eşleşme yok")
  exit()

src_pts = np.float32([ kp1[match.queryIdx].pt for match in good_matches ] ).reshape(-1, 2) #source image pointler x,y
dst_pts = np.float32([ kp2[match.trainIdx].pt  for match in good_matches ] ).reshape(-1, 2) #hedef image pointler x1,y1

start = time.time()

model, inliers,residual = ransac( #min_samples kadar data arasında doğru çizerekten inliers bakıyor resıdual thresholda gore

          (src_pts, dst_pts),
          AffineTransform, min_samples=4,
          residual_threshold=1, max_trials=100,random_state=0
      )

print(model)
print()
print("Residual",residual)
end = time.time()
print("Time cost",end-start)
n_inliers = np.sum(inliers)#toplam nokta sayım


inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]] # new source points
inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]# old source points
placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)] #eşleşenler

list_p1=[]
list_p2=[]
sy=0
#src_pts ve dst_pst sıralı ve karşılıklı bir şekilde duruyor.
syc=0

for point in src_pts[inliers]: #source pointlerimin inlier noktalarını al ve list_p1'e at.
  list_p1.append((point[0],point[1]))
for point in dst_pts[inliers]:  #hedef pointlerimin inlier noktalarını al ve list_p2'e at.
  list_p2.append((point[0],point[1]))

lenght=int(len(list_p1))
print("Data uzunluğum point sayisi",lenght)
point_cloud_data=int((lenght/100)*40) #bu datamın yüzde x ini kadar al

point_p1=list_p1[0:][0:point_cloud_data] #point p1 in içine at ve arraye çevir
point_p2=list_p2[0:][0:point_cloud_data] #point p1 in içine at ve arraye çevir

point_p1=np.array(point_p1).astype(np.int64)
point_p2=np.array(point_p2).astype(np.int64)

print("Point_cloud_data_len :",point_cloud_data)

x=point_p1[0:,0] # 1.fonksiyonun x datası
y=400-point_p1[0:,1] # 1.fonskyionun y datası (400=shape)

x1=point_p2[0:,0] # 2.fonksiyonun x datası
y1=400-point_p2[0:,1] #2.fonksyıonun y datası

plt.scatter(x, y)
plt.scatter(x1, y1)

from sklearn.cluster import KMeans

knn=KMeans(n_clusters=2, random_state=0) #etiketsiz kümele algoritması
knn.fit(point_p1)#point p1 için fitle
center=knn.cluster_centers_ #best match oldugu ıcın start end aynı oldu duzeltılcek
center_1=np.array( [center[0,0],400-center[0,1]] )    #start points   
center_2=np.array( [center[1,0],400-center[1,1]] )    #end points

plt.scatter(#merkez nokta koyma
    knn.cluster_centers_[:, 0], 400-knn.cluster_centers_[:, 1],
    s=125, marker='*',
    c='black', edgecolor='black',
    label='centroids')

print(" ")

knn2=KMeans(n_clusters=2, random_state=0)
knn2.fit(point_p2) #point p2 için fit et
center2=knn2.cluster_centers_ #center noktalarım

center_3=np.array( [center2[0,0],400-center2[0,1]] )    #start points         
center_4=np.array( [center2[1,0],400-center2[1,1]] )    #end points  

#iki noktası bilinen doğrunun denklemi
x_coords_1, y_coords_1 = (center[0,0],center[1,0]),(400-center[0,1],400-center[1,1])
A = np.vstack([x_coords_1,np.ones(len(x_coords_1))]).T
m, c = lstsq(A, y_coords_1,rcond=-1)[0]
print("Line Solution is 1 y1 = {m}x + {c}".format(m=m,c=c))
plt.plot(point_p1[0:,0], m*point_p1[0:,0]+c)#plot et

#iki noktası bilinen doğrunun denklemi
x_coords_2, y_coords_2 = (center2[0,0],center2[1,0]),(400-center2[0,1],400-center2[1,1])
A = np.vstack([x_coords_2,np.ones(len(x_coords_2))]).T
m_1, c_1 = lstsq(A, y_coords_2,rcond=-1)[0]
print("Line Solution is 2  y2 = {m}x + {c}".format(m=m_1,c=c_1))
plt.plot(point_p2[0:,0], m_1*point_p2[0:,0]+c_1)#plot et

plt.scatter(
    knn2.cluster_centers_[:, 0], 400-knn2.cluster_centers_[:, 1],
    s=125, marker='*',
    c='yellow', edgecolor='black',
    label='centroids'
    )

print("center1=",center_1,center_2)
print("center2=",center_3,center_4)

vector1=np.array([center_1,center_2]) #centerları yaz kontrol edelim
vector2=np.array([center_3,center_4])

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
    
    degg = (quadrant(vA,ilk) - quadrant(vB,son) ) #quadrant ile bölge tahmini yap ve dereceyi ayarla
    
    if(degg<0):
      degg += 360

    return degg 



deger=ang(vector1,vector2) #fonksiyonu cagır
print("Degree of real",deger)


counter=0
for i in point_p1:
  image = cv2.circle(chos_d, (int(point_p1[counter,0]),int(point_p1[counter,1])), 5, (250,250,146), -1)
  counter+=1
  if(counter==point_cloud_data):
    break
 
counter=0
for i in point_p2:
  image = cv2.circle(chos_d, (int(point_p2[counter,0]),int(point_p2[counter,1])), 5, (250,0,146), -1)
  counter+=1
  if(counter==point_cloud_data):
    break

####
cv2.imshow("Circle_Image",image)

image3 = cv2.drawMatches(chos, inlier_keypoints_left, cho, inlier_keypoints_right, placeholder_matches, None,flags=0)
cv2.imshow('Matches zort', image3)
cv2.waitKey(3)


plt.show()

cv2.waitKey(0)
