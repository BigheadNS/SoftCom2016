import cv2
import numpy as np
from scipy import ndimage
from vector import distance, pnt2line
import time
import os


# Bounding Rectangle :)
def getDigitEdges(img):
    dim=img.shape
    north=0
    south=0
    east=0
    west=0
    for r in range(0,dim[0]):
        for c in range(0, dim[1]):
            if north==0:
                if(img[r,c]==1):
                    north=r
            if south==0:
                if (img[dim[0] - 1 - r, c] == 1):
                    south=dim[0] - 1 - r

    for c in range(0,dim[1]):
        for r in range(0, dim[0]):
            if west==0:
                if(img[r,c]==1):
                    west=c
            if east==0:
                if (img[r, dim[1] - 1 - c] == 1):
                    east=dim[1] - 1 - c

    return north,south,west,east

#pozicioniranje cifre u gornji levi ugao
def transform_img(img):
    n,s,w,e=getDigitEdges(img)
    ret = (np.zeros((28,28))/255.0)>0.5
    if(img.shape[0]==0 or img.shape[1]==0):
        return ret
    ret[0:s-n+2, 0:e-w+2] = img[n-1:s+1, w-1:e+1]
    return ret


def calcLineCoeff(x1, y1, x2, y2):
    a = np.array([[x1,1],[x2,1]])
    b = np.array([y1,y2])
    [k, n] = np.linalg.solve(a,b)
    #print 'k=' + str(k) + ' n=' + str(n)
    return [k, n]


linesEndpoints = []


def calcLinesEndpoints(label, lines):
    groupedLines = [[], []]
    for i in range(len(label)):
        line = lines[i][0]
        # line = lines[i]
        groupedLines[int(label[i])].append(line)

    for j in range(len(groupedLines)):
        minval = (-1, -1)
        maxval = (10000, 10000)
        lines = groupedLines[j]
        for p in lines:
            if (p[1] > minval[1]):
                minval = (p[0], p[1])
            if (p[3] < maxval[1]):
                maxval = (p[2], p[3])

                # if (p[0]>minval[0]):
                #    minval = (p[0], p[1])
                # if (p[0]<maxval[0]):
                #    maxval = (p[0], p[1])

        # print maxval,minval
        global linesEndpoints
        linesEndpoints.append([minval, maxval])

    # izbrisi nepostojecu liniju
    if linesExpected != len(linesEndpoints):
        linesEndpoints.pop()

    print 'linesEndpoints: ' + str(linesEndpoints)

    return linesEndpoints

#preko hough transformacije dobavljas gornju i donju tacku linije
def getLinesEndpointsH(proba, linesExp):
    gray = cv2.cvtColor(proba, cv2.COLOR_BGR2GRAY)
    #gray = cv2.dilate(gray, kernel)
    #gray = cv2.erode(gray, kernel)
    edges = cv2.Canny(gray, 70, 130, apertureSize=5)
    # cv2.imwrite("edges.png", edges)

    # 30 10 inicijalno
    minLineLength = 30
    maxLineGap = 10

    # 50 odlicno
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength, maxLineGap)

    #frm = np.zeros((proba.shape[0], proba.shape[1], 3), np.uint8)

    coeffs=[]
    for cnt in range(len(lines)):
        x1,y1,x2,y2 = lines[cnt][0]
        kn = calcLineCoeff(x1, y1, x2, y2)
        coeffs.append(kn)
        #cv2.line(frm, (x1, y1), (x2, y2), (0, 255, 0), 2)

    coeffs = np.float32(coeffs)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(coeffs, linesExp, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    linesEndpoints=calcLinesEndpoints(label, lines)

    #for line in linesEndpoints:
    #    cv2.circle(frm, line[0], 5, (255, 255, 0), 2)
    #    cv2.circle(frm, line[1], 5, (255, 255, 0), 2)

    #cv2.circle(frm, gore, 2, (255, 0, 255), 1)
    #cv2.circle(frm, dole, 2, (255, 0, 255), 1)
    #cv2.imwrite("lines.png", frm)
    #return dole,gore
    return linesEndpoints

#preko inRange dobavljas pocetnu i krajnju tacku linije
def getLinesEndpoints(img):
    lowerB = np.array([150, 0, 0])
    upperB = np.array([255, 0, 0])
    lowerB = np.array(lowerB, dtype="uint8")
    upperB = np.array(upperB, dtype="uint8")
    maska = cv2.inRange(img, lowerB, upperB)
    img = 1.0 * maska

    # img = cv2.dilate(img, kernel)  # cv2.erode(img0,kernel)
    # img = cv2.dilate(img, kernel)

    min_y = (1000, 1000)
    max_y = (-1, -1)
    labeled, nr_objects = ndimage.label(img)
    objects = ndimage.find_objects(labeled)
    for i in range(nr_objects):
        loc = objects[i]
        (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                    (loc[0].stop + loc[0].start) / 2)

        (xg, yg) = min_y
        (xd, yd) = max_y
        if (yc < yg):
            min_y = (xc, yc)
        if (yc > yd):
            max_y = (xc, yc)
    linesEndpoints.append([max_y, min_y])
    return linesEndpoints


#ucitaj vec postojeci sacuvani model ili napravi novi

print "KNN training start"
knn = cv2.ml.KNearest_create()

if os.path.isfile('knn_data.npz'):
    with np.load('knn_data.npz') as filedata:
        # print data.files
        data = filedata['train']
        labels = filedata['train_labels']
        knn.train(data.astype(np.float32), cv2.ml.ROW_SAMPLE, labels.astype(np.float32))
else:
    from sklearn.datasets import fetch_mldata

    mnist = fetch_mldata("MNIST original")
    data = (mnist.data / 255.0) > 0.5
    labels = mnist.target.astype('int')

    for var in range(len(data)):
        img = data[var]
        img = img.reshape(28, 28)
        img = transform_img(img)
        data[var] = img.flatten()

    np.savez('knn_data.npz', train=data, train_labels=labels)

print "KNN training stop"

############################
linesExpected=1
videoName="lvl2/video-3.avi"
############################

cap = cv2.VideoCapture(videoName)

line = [(100,450), (500, 100)]

cc = -1
def nextId():
    global cc
    cc += 1
    return cc

def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal

# color filter
kernel = np.ones((2,2),np.uint8)
lower = np.array([230, 230, 230])
upper = np.array([255, 255, 255])

#boundaries = [
#    ([230, 230, 230], [255, 255, 255])
#]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output-rezB.avi',fourcc, 20.0, (640,480))

elements = []
t =0
counter = 0
suma = 0
times = []


frmCounter=0



while (1):
    start_time = time.time()
    ret, img = cap.read()

    if ret==True:
        if(frmCounter==0):
            frm=img.copy()
            #linesEndpoints=getLinesEndpoints(proba)
            linesEndpoints=getLinesEndpointsH(frm,linesExpected)

        for linija in linesEndpoints:
            cv2.circle(img, linija[1], 2, (255, 0, 255), 1)
            cv2.circle(img, linija[0], 2, (255, 0, 255), 1)

        # (lower, upper) = boundaries[0]
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(img, lower, upper)
        img0 = 1.0 * mask

        img0 = cv2.dilate(img0, kernel)  # cv2.erode(img0,kernel)
        img0 = cv2.dilate(img0, kernel)

        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled)
        for i in range(nr_objects):
            loc = objects[i]
            (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                        (loc[0].stop + loc[0].start) / 2)
            (dxc, dyc) = ((loc[1].stop - loc[1].start),
                          (loc[0].stop - loc[0].start))



            if (dxc > 11 or dyc > 11):
                cv2.circle(img, (xc, yc), 16, (25, 25, 255), 1)
                elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
                # find in range
                lst = inRange(20, elem, elements)
                nn = len(lst)
                if nn == 0:
                    elem['id'] = nextId()
                    elem['t'] = t
                    #elem['pass'] = False
                    elem['pass'] = [False] if linesExpected == 1 else [False, False]
                    elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                    elem['future'] = []
                    img_slice = img0[yc-14 : yc+14, xc-14: xc+14]
                    img_slice = (img_slice / 255.0) > 0.5
                    img_slice = transform_img(img_slice)
                    img_slice = np.float32(img_slice.reshape(-1, 784))
                    ret, result, neighbours, dist = knn.findNearest(img_slice, k=1)
                    elem['value']=ret
                    elements.append(elem)
                elif nn == 1:
                    lst[0]['center'] = elem['center']
                    lst[0]['t'] = t
                    lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                    lst[0]['future'] = []

        for el in elements:
            tt = t - el['t']
            if (tt < 3):
                for le in range(len(linesEndpoints)):
                    dist, pnt, r = pnt2line(el['center'], linesEndpoints[le][0], linesEndpoints[le][1])
                    if r > 0:
                        # cv2.line(img, pnt, el['center'], cl[le], 1)
                        c = (25, 25, 255)
                        if (dist < 9):
                            c = (0, 255, 160)
                            if el['pass'][le] == False:
                                el['pass'][le] = True
                                if len(linesEndpoints)==1:
                                    suma+=el['value']
                                else:
                                    if le == 0:
                                        suma -= el['value']
                                    else:
                                        suma += el['value']
                                counter += 1

                        cv2.circle(img, el['center'], 16, c, 2)

                id = el['id']
                cv2.putText(img, str(el['id']),
                            (el['center'][0] + 10, el['center'][1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

                cv2.putText(img, str(el['value']),
                            (el['center'][0] -20, el['center'][1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255))
                for hist in el['history']:
                    ttt = t - hist['t']
                    if (ttt < 100):
                        cv2.circle(img, hist['center'], 1, (0, 255, 255), 1)
                for fu in el['future']:
                    ttt = fu[0] - t
                    if (ttt < 100):
                        cv2.circle(img, (fu[1], fu[2]), 1, (255, 255, 0), 1)

        elapsed_time = time.time() - start_time
        times.append(elapsed_time * 1000)
        cv2.putText(img, 'Counter: ' + str(counter), (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)
        cv2.putText(img, 'Summ:  ' + str(suma), (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)

        frmCounter+=1

        # print nr_objects
        t += 1
        #if t % 10 == 0:
        #    print t
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        out.write(img)
    else:
        break


print "za " + str(videoName) + " suma je " + str(suma)
out.release()
cap.release()
cv2.destroyAllWindows()

et = np.array(times)
#print 'mean %.2f ms' % (np.mean(et))
# print np.std(et)


#---------ANN-----------------

#Uzimanje slucajnog uzorka za obuku i testiranje
#train_rank = 5000
#test_rank = 100
#------- MNIST subset --------------------------
#train_subset = np.random.choice(data.shape[0], train_rank)
#test_subset = np.random.choice(data.shape[0], test_rank)

# train dataset
#train_data = data[train_subset]
#train_labels = labels[train_subset]

# test dataset
#test_data = data[test_subset]
#test_labels = labels[test_subset]

# Transformacija labela u oblik koji ce biti na izlazu NM

#def to_categorical(labels, n):
#    retVal = np.zeros((len(labels), n), dtype='int')
#    ll = np.array(list(enumerate(labels)))
#    retVal[ll[:,0],ll[:,1]] = 1
#    return retVal

#test = [3, 5, 9]
#print to_categorical(test, 10)

# train and test to categorical
#train_out = to_categorical(train_labels, 10)
#test_out = to_categorical(test_labels, 10)

# Keras biblioteke za NM
#from keras.models import Sequential
#from keras.layers.core import Activation, Dense
#from keras.optimizers import SGD

# Konfiguracija NM
# prepare model
#model = Sequential()
#model.add(Dense(70, input_dim=784))
#model.add(Activation('relu'))
#model.add(Dense(50))
#model.add(Activation('tanh'))
#model.add(Dense(10))
#model.add(Activation('relu'))

# compile model with optimizer
#sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
#model.compile(loss='mean_squared_error', optimizer=sgd)

# Obucavanje i evaluacija NM
# training
#training = model.fit(train_data, train_out, nb_epoch=500, batch_size=400, verbose=0)
#print training.history['loss'][-1]

# evaluate on test data
#scores = model.evaluate(test_data, test_out, verbose=1)
#print 'test', scores

# evaluate on train data
#scores = model.evaluate(train_data, train_out, verbose=1)
#print 'train', scores

# Predikcija cifara koje su presle liniju
#keys = passIDs.keys()
#result = 0
#for k in keys:
#    imgB = passIDs[k]
#    (h,w) = imgB.shape
#    if (h*w == 784):
#        imgB_test = imgB.reshape(784)
#        imgB_test = imgB_test/255.
#        print 'za element sa id: ' + str(k)
#        tt = model.predict(np.array([imgB_test]), verbose=1)
#        rez_t = tt.argmax(axis=1)
#        result += rez_t[0]
#        print 'procenjena vrednost: ' + str(rez_t[0])
#        print '---------------------------------------'
        
#print 'Ukupan zbir = ' + str(result)