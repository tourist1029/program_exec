import numpy as np
from scipy import linalg

scoreData = np.mat([
[5,2,1,4,0,0,2,4,0,0,0],
[0,0,0,0,0,0,0,0,0,3,0],
[1,0,5,2,0,0,3,0,3,0,1],
[0,5,0,0,4,0,1,0,0,0,0],
[0,0,0,0,0,4,0,0,0,4,0],
[0,0,1,0,0,0,1,0,0,5,0],
[5,0,2,4,2,1,0,3,0,1,0],
[0,4,0,0,5,4,0,0,0,0,5],
[0,0,0,0,0,0,4,0,4,5,0],
[0,0,0,4,0,0,1,5,0,0,0],
[0,0,0,0,4,5,0,0,0,0,3],
[4,2,1,4,0,0,2,4,0,0,0],
[0,1,4,1,2,1,5,0,5,0,0],
[0,0,0,0,0,4,0,0,0,4,0],
[2,5,0,0,4,0,0,0,0,0,0],
[5,0,0,0,0,0,0,4,2,0,0],
[0,2,4,0,4,3,4,0,0,0,0],
[0,3,5,1,0,0,4,1,0,0,0]    
])


def cossim(vec1, vec2):
    dotprod = float(np.dot(vec1.T, vec2))
    normprod = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return 0.5 + 0.5 * (dotprod/normprod)

def eatscore(scoredata, scoredatarc, userindex, itemindex):
    n = np.shape(scoredata)[1]
    simsum = 0
    simsumscore = 0
    for i in range(n):
        userscore = scoredata[userindex, i]
        if userscore == 0 or i == itemindex:
            continue
        sim = cossim(scoredatarc[:,i], scoredatarc[:,itemindex])
        simsum = float(simsum + sim)
        simsumscore = simsumscore + userscore * sim
    if simsum == 0:
        return 0
    return simsumscore/simsum

#print(np.shape(scoreData)[0])  #0是读取矩阵的行
#print(np.shape(scoreData)[1])  #1是读取矩阵的列

#print(np.shape(scoreData))

U, sigma, VT = np.linalg.svd(scoreData)
print(sigma)
sigmasum = 0
k_num = 0

for k in range(len(sigma)):
    sigmasum = sigmasum + sigma[k] * sigma[k]
    if float(sigmasum) / float(np.sum(sigma ** 2)) > 0.9:
        k_num = k+1
        break;

sigma_k = np.mat(np.eye(k_num) * sigma[:k_num])
scoreDataRC = sigma_k * U.T[:k_num, :] * scoreData
n = np.shape(scoreData)[1]

#test
print("the scoreDataRC shape is : {}".format(np.shape(scoreDataRC)))

userindex = 17

for i in range(n):
    userscore = scoreData[17, i]
    if userscore != 0:
        continue
    print("index:{}, score:{}".format(i, eatscore(scoreData,scoreDataRC, userindex, i)))





















