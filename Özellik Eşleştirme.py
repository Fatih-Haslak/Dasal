import cv2
import matplotlib.pyplot as plt

# ana görüntüyü içe aktar
chos = cv2.imread("chocolates.jpg", 1)
cv2.imshow("1",chos)

# aranacak olan görüntü
cho = cv2.imread("nestle.jpg", 1)

cv2.imshow("2",cho)

# orb tanımlayıcı

# köşe-kenar gbi nesneye ait özellikler

orb = cv2.ORB_create()

# anahtar nokta tespiti
kp1, des1 = orb.detectAndCompute(cho, None)
kp2, des2 = orb.detectAndCompute(chos, None)

print(des1.shape)

# bf matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# noktaları eşleştir
matches = bf.match(des1, des2)

# mesafeye göre sırala
matches = sorted(matches, key = lambda x: x.distance)

# eşleşen resimleri görselleştirelim

img_match = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags = 4)
cv2.imshow("3",img_match)

cv2.waitKey(0)
# sift
