#from _typeshed import ReadableBuffer
import os
import numpy as np
from PIL import Image

print("当前工作路径为： {}".format(os.getcwd()))
oriImage = Image.open(r'test.PNG', 'r')
imgArray = np.array(oriImage)
print(imgArray.shape)
#print(imgArray)

R = imgArray[:,:,0]
G = imgArray[:,:,1]
B = imgArray[:,:,2]
A = imgArray[:,:,3]

print(R)
print('*****************************************')
print(G)
print('*****************************************')
print(B)
print('*****************************************')
print(A)

def imgCompress(channel, percent):
    U, sigma, VT = np.linalg.svd(channel)
    m = U.shape[0]
    n = VT.shape[0]
    rechannel = np.zeros((m,n))
    for k in range(len(sigma)):
        rechannel = rechannel + sigma[k] * np.dot(U[:,k].reshape(m,1), VT[k,:].reshape(1,n))
        if float(k) / len(sigma) > percent:
            rechannel[rechannel < 0] = 0
            rechannel[rechannel > 255] = 255
            break
    return np.rint(rechannel).astype("uint8")


U, sigma, VT = np.linalg.svd(R)
m = U.shape[0]
n = VT.shape[0]
#print(m)
#print(n)
#print(np.zeros((m,n)))
#print(U[:,5])

for p in [0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    reR = imgCompress(R,p)
    reG = imgCompress(G,p)
    reB = imgCompress(B,p)
    reA = imgCompress(A,p)
    reI = np.stack((reR, reG, reB, reA), 2)
    Image.fromarray(reI).save("{}".format(p)+"img.png")

































