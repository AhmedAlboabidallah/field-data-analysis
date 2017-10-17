# IMPORT
# IMPORT # most libraries installed with pip install file.whl#
#most libraries downloaded from http://www.lfd.uci.edu/~gohlke/pythonlibs/
#shapefile installed with pip# pip install pyshp
teststep=0
#from tkFileDialog import *
import tkinter as tk
from tkinter import *
#from Tkinter import Tk
import math as m
import math
import csv
import numpy as np
import numpy
import matplotlib.pyplot as plt
from matplotlib  import cm
import sys
import shapefile#                                             pip install pyshp
from time import gmtime, strftime
import cv2 #            conda install -c https://conda.binstar.org/menpo opencv
import scipy.linalg
from scipy import signal as sg
from scipy.linalg import inv, eigh, solve
import pylab
from mpl_toolkits.mplot3d import Axes3D
#from tkinter import *
#import Image #http://www.pythonware.com/products/pil/
from PIL import Image
#from __future__ import print_function
import glob
import os
#from easygui import *
import vigra #conda create -n vigra -c ukoethe python=2.7.10.vc11 vigra=1.11.0.vc11
             #activate vigra
from vigra import *
#sys.path.extend(['C:/Program Files/ArcGIS/Desktop10.1/ArcToolbox/Scripts','C:/Program Files/ArcGIS/Desktop10.1/bin','C:/Program Files/ArcGIS/Desktop10.1/arcpy'])
#sys.path.extend(['C:/Program Files (x86)/ArcGIS/Desktop10.3/arcpy/arcpy/geoprocessing','C:/Program Files (x86)/ArcGIS/Desktop10.3/ArcToolbox/Scripts','C:/Program Files (x86)/ArcGIS/Desktop10.3/bin','C:/Program Files (x86)/ArcGIS/Desktop10.3/arcpy'])
#sys.path.extend(['', 'C:\\Windows\\system32\\python27.zip', 'C:\\Python27\\ArcGISx6410.3\\DLLs', 'C:\\Python27\\ArcGISx6410.3\\lib', 'C:\\Python27\\ArcGISx6410.3\\lib\\plat-win', 'C:\\Python27\\ArcGISx6410.3\\lib\\lib-tk', 'C:\\Python27\\ArcGISx6410.3', 'C:\\Python27\\ArcGISx6410.3\\lib\\site-packages', 'C:\\Program Files (x86)\\ArcGIS\\Desktop10.3\\bin64', 'C:\\Program Files (x86)\\ArcGIS\\Desktop10.3\\ArcPy', 'C:\\Program Files (x86)\\ArcGIS\\Desktop10.3\\ArcToolBox\\Scripts'])

import winsound
import pickle
import subprocess
#import gc
#gc.disable()
import pandas as pd
import timeit
import gdal, osr                   #conda install gdal     #conda upgrade numpy #pip install pyqt5
from skimage.morphology import skeletonize
from skimage import draw
import matplotlib.pyplot as plt
import string
import math
from scipy.linalg import inv, eigh, solve
import numpy as np
from numpy.linalg import eig, inv
import itertools
from numpy import mean,cov,double,cumsum,array,rank,dot,linalg,size,flipud
from pylab import *
import shutil, errno
import numpy.linalg as la
from gdalconst import *
import gdal, osr
import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy.lib.stride_tricks import as_strided as ast
import dask.array as da
from joblib import Parallel, delayed, cpu_count
from joblib import *
import joblib
import os
from skimage.feature import greycomatrix, greycoprops
from scipy import ndimage
import glob
from skimage.morphology import skeletonize
from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import requests
from PIL import *
from pandas import *
#from PIL import Image
#import arcpy
#from arcpy import env
#from tkFileDialog import *

#os.environ['ESRI_SOFTWARE_CLASS']='Professional'
'''
# FUNCTIONS
'''
winsound.Beep(900, 50)
#1
def abs_images(imagespath, image,outimagepath,image2):
    arr2 =vigra.impex.readImage(imagespath+'/'+image, dtype='', index=0, order='')
    vigra.impex.writeImage(abs(arr2.astype(numpy.int16)), outimagepath+image2, dtype = '', compression = '', mode = 'w')  
    #abs_images(imagespath, image,outimagepath,image2)
#2    
# a function to add a list to an existed external file 
def Addtofile(path,file1,list1,NoOfColumns=3):
    try:
        os.stat(path)
    except:
        os.makedirs(path)
    F=path+file1
    text1=''
    for i in range(NoOfColumns):
        text1=text1+',x'+str(i+1)
    text1=text1[1:]
    read=open(F,'a')
    if NoOfColumns!=1:
        for line in list1:
            exec(text1+'= [float(value) for value in line]')
            for i in range(NoOfColumns):
                exec("read.write(str(x"+str(i+1)+'))')
                read.write(',')
            read.write('\n')
    else:
        for line in list1:
            exec("read.write(str(line))")
            read.write('\n')
    read=0
winsound.Beep(400, 350)
# a function to apply the minimum elevation point filter along the profiles
#3
def appendinlist(list1,index,appedwith):
    list2=list1[index]
    list2.append(appedwith)
    list1[index]=list2
    return list1
#4
def array2raster_old(imagepath,imagefile1,originX,originY,pixelWidth,pixelHeight,array,type1=gdal.GDT_Byte):
    try:
        os.stat(imagepath)
    except:
        os.makedirs(imagepath) 
    newRasterfn=imagepath+imagefile1
    [rows,cols]=np.shape(array)
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, type1 )
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)  
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
#5
def array2raster(imagepath,imagefile1,originX,originY,pixelWidth,pixelHeight,array,type1=gdal.GDT_Byte):
    try:
        os.stat(imagepath)
    except:
        os.makedirs(imagepath) 
    newRasterfn=imagepath+imagefile1
    [rows,cols]=np.shape(array)
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, type1 )
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)  
    outRaster.SetProjection('PROJCS["WGS 84 / UTM zone 30N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-3],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","32630"]]')
    outband.FlushCache()
#6
def avg_layers(infilepath,layer_from=15,layer_to=30,ext='tif'):# a function to calculate the avaragre of a given range of layers# works only where number of layers is less than 25 because of gdal naming limitation
    #print(1)
    infilepath=infilepath.replace('/', "\\")
    #print(2)
    files=FindFilesInFolder(infilepath,'*'+ext)
    try:
        files.remove('avg.tif')
        print('old avg.tif will be replaced')
    except:
        pass
    files=list(filter(lambda x: int(''.join(c for c in x if c.isdigit()))<layer_to and int(''.join(c for c in x if c.isdigit()))>=layer_from, files))
    #print(3)
    k1=list(string.ascii_uppercase)
    k2=['']
    k2.extend(k1)
    text1=r''
    text2=r'('
    i=-1
    #print(4)
    for file1 in files:
        i+=1
        text1=text1+'-'+k2[math.floor(i/26)]+k1[i-(math.floor(i/26))*26]+' '+infilepath+file1+' '
        text2=text2+k2[math.floor(i/26)]+k1[i-(math.floor(i/26))*26]+'+'
    #print(5)
    text2=text2[:-1]+')'
    #text1=text1.replace('/', "\\")
    print(text1)
    print(text2)
    #infilepath=infilepath.replace('/', "\\") 
    #print(6)
    gdal_it=r'C:\\gdal_calc.py '+ text1+' --outfile='+infilepath+'avg.tif '+' --calc="'+text2+'/'+str(len(files))+'"'
    #print(gdal_it)
    os.system('python '+gdal_it)
#7
def Az_C2p(Xc,Yc,Xp,Yp):#ccw angle from X axis to the P around C
    sin=(Xp-Xc)/(((Yp-Yc)**2+(Xp-Xc)**2)**0.5)
    cos=(Yp-Yc)/(((Yp-Yc)**2+(Xp-Xc)**2)**0.5)
    ang=abs((math.asin(sin)))
    if sin>0:
            if cos>0:
                th=np.pi/2-ang
            else:
                th=3*np.pi/2+ang
    else:
            if cos>0:
                th=np.pi/2+ang
            else:
                th=3*np.pi/2-ang
    return th
#8
#function for binary array production
def binary(array1,thrushold,DGmax=1):
    low_values_indices = array1 < thrushold  # Where values are low
    array1[low_values_indices] = 0      
    high_values_indices = array1 >= thrushold
    array1[high_values_indices] = DGmax    
    return array1  
#9
def BranchLength(SortedBranchPoints):
    Length=0
    for point in BranchPoints[0:-1]:
        D=(Dx**2+Dy**2+Dz**2)**.5
        Length=Length+D
    return Length
#import matplotlib.pyplot as mp
#import matplotlib.colors as colors
#10
def BranchBiomass2(List_Series_pixels_Coordinates,step1,BranchBiomassImage,centres,layerpath):
    # read the image
    #read image
    # for branch
    B,l,ClosestCentres=[],[],[]
    for chain in List_Series_pixels_Coordinates:
        #print(len(chain))
        #print(chain)
        ClosestCentreI=ClosestCentre(chain,centres)
        #print(1)
        ClosestCentres.append(ClosestCentreI)
        #print(2)
        points1,sortedchain=ClosestFarthestPoints(chain,ClosestCentreI)
        #print(3)
        length=(points1[1][-1]-points1[0][-1])*step1
        #print(4)
        l.append(length)
        firstpart=list(filter(lambda a:a[-1]<sortedchain[0][-1]+.30,sortedchain))
        #lastpart =filter(lambda a:a[-1]>sortedchain[0][-1]-.30,sortedchain)
        A=[]
        for point in firstpart:
            A.append([point[0],point[1],float(point[5][6:])])
        B.append(A)
    layers=FindFilesInFolder(layerpath,'layer*')
    layers=list(filter(lambda l:int(l[5:-4])>15,layers))
    #layers=filter(lambda l:int(l[5:-4])<150,layers)
    layers=sorted(layers,key=lambda l:int(l[5:-4]))
    X=[]
    print(5)
    for i in range(shape(B)[0]):
        X.append([])
    for layer in layers:
        #layer#read layer
        layer1=readtolist(layerpath,layer,5)
        print('layerpath,layer',layerpath,layer)
        iii=-1
        for start in B:
            iii+=1
            #print('iii=',iii)
            #start
            try:
                minX,maxX=min(np.array(start)[:,0])-step1/2,max(np.array(start)[:,0])+step1/2
                minY,maxY=min(np.array(start)[:,1])-step1/2,max(np.array(start)[:,1])+step1/2
                minZ,maxZ=min(np.array(start)[:,2])*.1-.5,max(np.array(start)[:,2])*.1+.5
                #filter the layer for min max
                X[iii].append(list(filter(lambda l:l[0]<maxX and l[1]<maxY and l[0]>minX and l[1]>minY and l[3]> minZ and l[3]> maxZ,layer1)))
            except:
                 print('memory error ','layerpath',layerpath,'layer',layer,'iii',iii,'out of', len(B))
        #coor.append(coori)
        #np.array(A)
    coor=[]
    #for ii in range(shape(B)[0]):
    #    exec('coor.append(X'+str(int(ii))+')')
    ii=-1
    rr,vol,W=[],[],[]
    print(6)
    for chain in List_Series_pixels_Coordinates:
        ii+=1
        try:
            coeff,score,latent=princomp(np.array(X[ii])[:,0:3],3)
            r=2*median(abs(score[2]))
        except:
            r=.05
        #print(r)
        rr.append(r)#
        vol.append(r**2*np.pi*l[ii])
        W.append(wightedSeries(List_Series_pixels_Coordinates[ii],ClosestCentres[ii]))
    print(7)
    import pickle
    try:
        pickle.dump(vol, open(layerpath+'/BranchBiomassVol.txt', 'w')) 
        pickle.dump(W, open(layerpath+'/BranchBiomassW.txt', 'w'))
        #vol=pickle.load(open(layerpath+'/BranchBiomassVol.txt')) 
    except:
        pass
    WXV=[]
    i=0
    arr1 = np.zeros((50/step1+1,50/step1+1))
    if centres[0][0]<0:
        Xmin=math.ceil(centres[0][0]/50)*50.0
    else:
        Xmin=math.ceil(centres[0][0]/50)*50.0-50
    if centres[0][1]<0:
        Ymin=math.ceil(centres[0][1]/50)*50.0
    else:
        Ymin=math.ceil(centres[0][1]/50)*50.0-50
    print(8)
    for i in range(len(vol)-1):
        for pixel in W[i]:
            arr1[int(round((pixel[0]-Xmin)/step1)),int(round((pixel[1]-Ymin)/step1))]=arr1[int(round((pixel[0]-Xmin)/step1)),int(round((pixel[1]-Ymin)/step1))]+pixel[2]*vol[i]
            WXV.append([pixel[0],pixel[1],pixel[2]*vol[i]])
    print(9)
    arr2=1000000*np.transpose(arr1)
    vigra.impex.writeImage(arr2.astype(numpy.float32), BranchBiomassImage, dtype = '', compression = '', mode = 'w')  
    pickle.dump(arr2,open(layerpath+'/BranchBiomassArray.txt', 'w'))
    #f = open(layerpath+'/BranchBiomass2.txt', 'r')
    #AAAA= pickle.load(f)
    # find the closest centre
    # find the distances of each pixel 
    #pc first and last 20 cm 
    # take the mode as r 
    # find the volume
    # make the wights matrix
    # find the summation of wights
    # read the image
    # for point add wighted volume
#11
def BranchBiomass3(List_Series_pixels_Coordinates,step1,BranchBiomassImage,centres,layerpath):
    # read the image
    #read image
    # for branch
    B,l,ClosestCentres=[],[],[]
    for chain in List_Series_pixels_Coordinates:
        #print(len(chain))
        #print(chain)
        ClosestCentreI=ClosestCentre(chain,centres)
        #print(1)
        ClosestCentres.append(ClosestCentreI)
        #print(2)
        points1,sortedchain=ClosestFarthestPoints(chain,ClosestCentreI)
        #print(3)
        chain=sorted(chain,key=lambda l:int(l[5][6:]))
        Dz=int(chain[0][5][6:])-int(chain[-1][5][6:])
        length=((points1[1][-1]-points1[0][-1])**2+Dz**2)**.5
        #print(4)
        l.append(length)
        firstpart=list(filter(lambda a:a[-1]<sortedchain[0][-1]+.30,sortedchain))
        #lastpart =filter(lambda a:a[-1]>sortedchain[0][-1]-.30,sortedchain)
        A=[]
        for point in firstpart:
            A.append([point[0],point[1],float(point[5][6:])])
        B.append(A)
    #layers=FindFilesInFolder(layerpath,'layer*')
    #layers=filter(lambda l:int(l[5:-4])>10,layers)
    #layers=sorted(layers,key=lambda l:int(l[5:-4]))
    X=[]
    print(5)
    for i in range(shape(B)[0]):
        X.append([])
    coor=[]
    #for ii in range(shape(B)[0]):
    #    exec('coor.append(X'+str(int(ii))+')')
    ii=-1
    rr,vol,W=[],[],[]
    print(6)
    for chain in List_Series_pixels_Coordinates:
        ii+=1
        try:
            coeff,score,latent=princomp(np.array(B[ii]),3)
        except:
            score[2]=.04
        r=2*median(abs(score[2]))+.04
        print('ok')
        #except:
        #    r=.05
        #print(r)
        rr.append(r)#
        vol.append(r**2*np.pi*l[ii])
        W.append(wightedSeries(List_Series_pixels_Coordinates[ii],ClosestCentres[ii]))
    print(7)
    import pickle
    try:
        f = open(layerpath+'/BranchBiomassVol.txt', 'w')
        pickle.dump(vol, f) 
        f = open(layerpath+'/BranchBiomassW.txt', 'w')
        pickle.dump(W, f)
        
    except:
        pass
    WXV=[]
    i=0
    #try:
    #    arr1 =vigra.impex.readImage(BranchBiomassImage, dtype='', index=0, order='')
        #arr1 = arr1[:,:,0]
    #except:
    arr1 = np.zeros((50/step1,50/step1))
    if centres[0][0]<0:
        Xmin=math.ceil(centres[0][0]/50)*50.0
    else:
        Xmin=math.ceil(centres[0][0]/50)*50.0-50
    if centres[0][1]<0:
        Ymin=math.ceil(centres[0][1]/50)*50.0
    else:
        Ymin=math.ceil(centres[0][1]/50)*50.0-50   
    print(8)
    for i in range(len(vol)-1):
        for pixel in W[i]:
            arr1[int((pixel[0]-Xmin)/step1),int((pixel[1]-Ymin)/step1)]=arr1[(pixel[0]-Xmin)/step1,(pixel[1]-Ymin)/step1]+pixel[2]*vol[i]
            WXV.append([pixel[0],pixel[1],pixel[2]*vol[i]])
    print(9)
    arr2=np.transpose(arr1)
    vigra.impex.writeImage(arr2.astype(numpy.float32), BranchBiomassImage, dtype = '', compression = '', mode = 'w')  
#12
def combinedinlist(list1,index1,index2):
    list2=list1[index1]
    list2.extend(list1[index2])
    list1[index1]=list2
    del  list1[index2]
    return list1
#4
def ClosestCentre(chain,centres):
    Xs,Ys,Zs=[],[],[]
    for i in chain:
        Xs.append(i[0])
        Ys.append(i[1])
        Zs.append(float(int(i[5][6:]))*0.1)
    Xmean,Ymean,Zmean=np.average(np.array(Xs)),np.average(np.array(Ys)),np.average(np.array(Zs))
    i=-1
    #centresD=centres
    centresD=[]
    for centre in centres[0:]:
            centre
            try:
                Xcentre,Ycentre,Zcentre=centre[0],centre[1],centre[2]*.1
            except:
                print('error in centres')
            i+=1
            D=((Xmean-Xcentre)**2+(Ymean-Ycentre)**2+(Zmean-Zcentre)**2)**.5
            #appendinlist(,i,[D])
            centresD.append([Xcentre,Ycentre,Zcentre,D])
    centresD=sorted(centresD,key=lambda l:l[-1])
    return centresD[0]
    #ClosestCentreI=ClosestCentre(chain,centres)
#13
def ClosestFarthestPoints(chain,ClosestCentreI):
        centre1=ClosestCentreI
        #centre2=ClosestCentreI[1]
        Xcentre1,Ycentre1=centre1[0],centre1[1]#
        #Xcentre2,Ycentre2=centre2[1],centre2[2]#
        i=-1
        for point in chain:
            i+=1
            #print(i)
            D1=((point[0]-Xcentre1)**2+(point[1]-Ycentre1)**2)**.5
            #D2=((point[0]-Xcentre2)**2+(point[1]-Ycentre2)**2)**.5
            appendinlist(chain,i,D1)
            #appendinlist(chain,i,D2)
        chain=sorted(chain,key=lambda l:l[-1])
        points1=[chain[0],chain[-1]]
        #chain=sorted(chain,key=lambda l:l[-2])
        #points2=[chain[0],chain[-1]]
        #appendinlist(List_Series_Coordinates,t,[points1,points2])
        return points1,chain
        #ClosestCentreI,points1,chain=ClosestFarthestPoints(chain,ClosestCentreI)
#14
def Cdetector(imagepath,image,x0,y0,step):
    imagefile=imagepath+image
    img = cv2.imread(imagefile,0)
    #img = cv2.medianBlur(img,int(6*step))
    img= cv2.blur(img,(2,2))
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,2,30,
                                param1=300,param2=1,minRadius=int(.01/step),maxRadius=int(1.5/step))
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    #cimg.save(cimg)
    cv2.imwrite('detected circles.png',cimg)
    #me again
    detected=[]
    for c in circles[0,:]:
        cc=step*c+[y0, x0,0]
        cc=[cc[1],cc[0],cc[2]]
        detected.append(list(cc))
    c=circles[0]
    c.astype(float) 
    return [np.array(detected),c]
#15
def center_of_mass(SegmentsFilesPath,SID):#finds the centre of gravity of the segment
    P=Read_Gegment(SegmentsFilesPath,SID)
    XC=np.mean(np.array(P)[:,0])
    YC=np.mean(np.array(P)[:,1])
    return (XC,YC) 
#16
def cleanC(input,THRUSHOLD):# input is [[x,y,sth],[x,y,sth],...]
  output = []
  for x in input:
      test=len(list(filter(lambda a: (a[0]-x[0])**2+(a[1]-x[1])**2<THRUSHOLD**2, output)))
      if test==0:
           output.append(x)
  output.sort()
  return output
#17
def ClosestPixel(path1,file1,x,y,r,step1=0.05):
    #xx,yy=ClosedtPixel('C:/Users/ahalboabidallah/Desktop/ash_farm_new/x-25y-25/images/','15tif.tif',-3.549,-11.706,5,.05)
    try:
        a=subset2np(path1,file1,x-r,y-r,x+r,y+r)
        b=np.transpose(numpy.where(a > 0))
        c=b-np.shape(a)[0]/2
        k=c[:,0]**2+c[:,1]**2
        xx=c[numpy.where(k == min(k))[0][0]][1]*step1+x
        yy=c[numpy.where(k == min(k))[0][0]][0]*step1+y
        xx=xx+step1
        yy=yy+step1
    except:
        xx=x
        yy=y
    return xx,yy
#18
#finding the closest segment to point and the angle it covers
#    inputs are: GLOBAL centre coordinates and find4pts results
def closestSegment(x,y,B):
    D=[]
    SS=[]
    for point in B:
        Xp,Yp,Yimg,Ximg,S=point
        d=((Xp-x)**2+(Yp-y)**2)**.5
        D.append([d,S])
    D= sorted(D,key=lambda l:l[0])
    D= sorted(D,key=lambda l:l[1])
    k=D[0][1]
    SS=[D[1]]
    
    for i in D:
        [d,s]=i
        if s==k:
            pass
        else:
            SS.append(i)
            k=s
    SS= sorted(SS,key=lambda l:l[0])
    Ds=np.array(SS)[:,0]
    Ss=np.array(SS)[:,1]
    #SS is a list of [distances,Segments] with  low to high sequence.
    return(Ss,Ds)  
#19
#finding the closest segment to point and the angle it covers
#    inputs are: GLOBAL centre coordinates and find4pts results
def closestSegment2(x11,y11,B):
    D=[]
    SS=[]
    try:
        BD= DataFrame(B,columns=['Xp','Yp','Yimg','Ximg','S','Layer'])
        BD=BD.drop(['Layer'], axis=1)
    except:
        BD= DataFrame(B,columns=['Xp','Yp','Yimg','Ximg','S'])
    BD['D']=((BD['Xp']-x11)**2+(BD['Yp']-y11)**2)**.5    
    BD=BD.sort_values(['D'], ascending=[True])
    BD=BD.sort_values('D')
    #for point in B:
    #    print(point)
    #    Xp,Yp,Yimg,Ximg,S=point
    #    d=((Xp-x)**2+(Yp-y)**2)**.5
    #    D.append([d,S])
    #D= sorted(D,key=lambda l:l[0])
    #D= sorted(D,key=lambda l:l[1])
    #k=D[0][1]
    #SS=[D[1]]
    BD1=BD.drop_duplicates(['S'], keep='first')
    BD1=BD1.drop(['Xp','Yp','Yimg','Ximg'], axis=1)
    Ss=BD1['S'].values.tolist()
    Ds=BD1['D'].values.tolist()
    #SS is a list of [distances,Segments] with  low to high sequence.
    return Ss,Ds
#20
def CNtoLC_LN(CN,CurrentLayer,NextLayer):#current next parenting list to layer current and layer next #only for formating; no calculations.
    LC=np.column_stack(([CurrentLayer]*np.shape(CN)[0],np.array(CN)[:,0]))
    LN=np.column_stack(([NextLayer]*np.shape(CN)[0],np.array(CN)[:,1]))
    LCLN=np.column_stack((LC,LN))
    return LC,LN,LCLN

#21
def CombineCentreFiles(centrespath):
    # FindFilesInFolder centres
    A1=FindFilesInFolder(centrespath,'*.csv')
    A2=FindFilesInFolder(centrespath,'e*')
    A3=FindFilesInFolder(centrespath,'H*')
    centre_files=[item for item in A1 if item not in set(A2) and item not in set(A3)]
    # read centres
    centresHights=readtolist(outpath+'Centres/','ellipsesWithHights.csv',7) 
    list2=[]
    for centrei in centre_files:
        centrei
        thisCentre=readtolist(outpath+'Centres/',centrei,6)
        #newcentrei=filter(lambda a: a[-5:]==thisCentre[0][-5:] or a[-5:]==thisCentre[1][-5:], centresHights)#
        newcentrei=list(filter(lambda a: int(a[1]*100)==int(thisCentre[0][0]*100) and int(a[2]*100)==int(thisCentre[0][1]*100), centresHights))#
        layer=[]
        C=(thisCentre[0][2]*thisCentre[0][3]*pi/(newcentrei[0][0]-thisCentre[0][5])**2)
        for layer in thisCentre:
            layer.append(newcentrei[0][0])
            C
            layer[5]
            layer.append(C*(newcentrei[0][0]-layer[5])**2)
            layer
            Addtofile(centrespath,'H'+centrei,[layer],8)
#22
#Function to compare the the  points elevations for all points that have the same id 
#inputs: the lists and the buffer around the DTM
# retunes a new non-ground points list with the thier hights above the ground
def comparelists(list1,listg,grass):
    list1,listg=np.array(list1),np.array(listg)
    i=-1
    j=-1
    list3=[]
    for point in listg:
        i+=1
        j+=1
        while float(listg[i,-1]) != float(list1[j,-1]): #and j < np.shape(list1)[0]
            list3.append(list(list1[j]))#when no ground dont miss point
            #list3.append([j,1])#test
            j+=1
        if (float(list1[j,2])-float(listg[i,2]))>grass:#when ground exist compare points
            list3.append(list(list1[j]))
            #list3.append([j,2])#test
        else:
            pass
    j+=1
    while j<float(np.shape(list1)[0]):
        #list3.append([j,3])#test
        list3.append(list(list1[j]))
        j+=1
    return(list3)

#23
def ConvGen(r_pixel): # A function to generate a circular ones
    conv=np.ones([2*r_pixel+1,2*r_pixel+1])
    for i in np.ndindex(conv.shape):
        #print (i)
        if (i[0]-r_pixel)**2+(i[1]-r_pixel)**2>r_pixel**2:
            conv[i]=0
    #convWighted=conv/sum(conv)
    return conv#,convWighted
#24
def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise
#25
def creat_local_wighting_filter(xc,yc,step1,path1,image1,imagef):
    ds = gdal.Open(path1+image1)
    (xmino,res1,tilt1,ymino,tilt2,res2)=ds.GetGeoTransform()
    xmaxo = xmino + (ds.RasterXSize * res1)
    ymaxo = ymino + (ds.RasterYSize * res2)
    Ds=(max((xc-xmaxo)**2+(yc-ymaxo)**2,(xc-xmaxo)**2+(yc-ymino)**2,(xc-xmino)**2+(yc-ymaxo)**2,(xc-xmino)**2+(yc-ymino)**2))**0.5
    L=math.ceil(Ds/step1+1)
    x=np.array([list(range(L))]*math.ceil(L))
    y=np.transpose(np.array([list(range(math.ceil(L)))]*L))
    z1=(L)**2-x**2-y**2
    z2=np.fliplr(z1)#
    z=np.vstack((np.flipud(np.hstack((z2,z1))),(np.hstack((z2,z1)))))
    Xmin,Ymin=xc-Ds,yc-Ds
    array2raster(path1,imagef,Xmin,Ymin,step1,step1,z,type1=gdal.GDT_Float64)
    translate = 'gdal_translate -projwin %s %s %s %s %s %s' %( xmino, ymino,xmaxo, ymaxo, path1+imagef, path1+'s'+imagef)
    os.system(translate)
    #subprocess.call([sys.executable, "C:\\gdal_calc.py", "-A", path1+image1,"-B",path1+image1    , "--outfile="+path1,'w'+image1, "--calc=(A*B)"])
    #subprocess.call([sys.executable, "C:\\gdal_calc.py", "-A", path1+image1,"-B",path1+'s'+imagef, "--outfile="+path1,'w'+image1, "--calc=(A*B)"])
    gdal_it=r'C:\\gdal_calc.py '+ '-A '+path1+image1+' -B '+path1+'s'+imagef+' --outfile='+path1+'w'+image1+" --calc=(A*B)"
    os.system('python '+gdal_it)
    return xmino,ymino,xmaxo,ymaxo
#creat_local_wighting_filter(-19.30651198,-5.393043621 ,0.05, 'C:/Users/ahalboabidallah/Desktop/ash_farm_new/x-25y-25/Branches/7/','13139.tif','del.tif') 
#26
def create_ellipse(r, xc, alpha, n=100, angle_range=(0,2*np.pi)):
    """ Create points on an ellipse with uniform angle step
    
    Parameters
    ----------
    r: tuple
        (rx, ry): major an minor radii of the ellipse. Radii are supposed to
        be given in descending order. No check will be done.
    xc : tuple
        x and y coordinates of the center of the ellipse
    alpha : float
        angle between the x axis and the major axis of the ellipse
    n : int, optional
        The number of points to create
    angle_range : tuple (a0, a1)
        angles between which points are created.
        
    Returns
    -------
        (n * 2) array of points 
    """
    R = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])
    
    a0,a1 = angle_range
    angles = np.linspace(a0,a1,n)
    X = np.vstack([ np.cos(angles) * r[0], np.sin(angles) * r[1]]).T
    return np.dot(X,R.T) + xc
#27
def CreateRaster(xx,yy,std,gt,proj,driverName,outFile):  
    '''
    Exports data to GTiff Raster
    '''
    std = np.squeeze(std)
    std[np.isinf(std)] = -99
    driver = gdal.GetDriverByName(driverName)
    rows,cols = np.shape(std)
    ds = driver.Create( outFile, cols, rows, 1, gdal.GDT_Float32)      
    if proj is not None:  
        ds.SetProjection(proj.ExportToWkt()) 
    ds.SetGeoTransform(gt)
    ss_band = ds.GetRasterBand(1)
    ss_band.WriteArray(std)
    ss_band.SetNoDataValue(-99)
    ss_band.FlushCache()
    ss_band.ComputeStatistics(False)
    del ds

#28
#function to convert ascii files with .csv extension into shapefiles
#inputs: file nemes and paths 
def csv2shp(csvpath,csvFile,shppath,shpFile):
    import shapefile
    # Make a point shapefile
    list1=readtolist(csvpath,csvFile)
    s=shppath+shpFile
    w = shapefile.Writer(shapefile.POINTZ)
    w.field('Z','F',10,8)
    w.field('pt','F',10,8)
    ID=-1
    for point in list1:
        ID+=1
        x,y,z=point
        w.point(x,y,z)
        w.record(z,ID)
        #w.record(z)
    try:
        w.save(s)
    except:
        print('Empty file')
#29
#function to convert ascii files with .csv extension into shapefiles with reduction
#inputs: file nemes, paths and reduction factor
def csv2shpr(csvpath,csvFile,shppath,shpFile,sth=50):
    import shapefile
    # Make a point shapefile
    list1=reducetolist(csvpath,csvFile,sth)
    s=shppath+shpFile
    w = shapefile.Writer(shapefile.POINTZ)
    w.field('Z','F',10,8)
    for point in list1:
        x,y,z=point
        w.point(x,y,z)
        w.record(z)
    try:
        w.save(s)
    except:
        print('Empty file')
#30
def Deltas(x0n,an,rn,Zmin,Zmax,Xpoints):
    #from . import materials
    #materials.rough = materials.diffuse
    position=[x0n[0,0][0,0][0,0],x0n[1,0][0,0][0,0],x0n[2,0][0,0][0,0]]
    Dz=-(Zmax-Zmin)
    R=Dz/cos(an[2])
    Dx=float(-R*an[0])
    Dy=float(-R*an[1])
    return [Dx,Dy,Dz]
#31
#colour conversion functions
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
#32
#cylinder fitting functions#http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A352468&dswid=-9696
#https://github.com/agarmo/MasterThesis/blob/master/matlab/toolbox/lsGeometricElements/lscylinder.m
'''to skip go to 735'''
def drrot3(R1=np.mat([]),R2=np.mat([]),R3=np.mat([])):
    if R1 != np.mat([]):
        dR1=np.mat([ [0,0,0], [0,-R1[2, 1],-R1[1, 1]],[0,R1[1, 1],-R1[2,1]]])
    else:
        dR1=0
    if R2 != np.mat([]):  
        dR2=np.mat([ [-R2[0,2],0,-R2[0, 0]],[0,0,0],[R2[0,0],0,-R2[0, 2]]])
    else:
        dR2=0
    if R3 != np.mat([]):
        dR3 = np.mat([ [-R3[1,0],-R3[0,0],0] , [R3[0,0],-R3[1, 0],0]   , [0,0,0]])
    else:
        dR3=0
    #checked and vetted
    return (dR1,dR2,dR3)
#33
#function to find point distance from (2 apoints line)
def dist(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    dist = abs((y2-y1)*x3-(x2-x1)*y3+x2*y1-y2*x1)/((y2-y1)**2+(x2-x1)**2)**.5
    return dist
#34
def draw_elli(X0,Y0,Z0,aa,bb,t):
    #X=draw_elli(X0,Y0,Z0,aa,bb,t)
    #before rotate
    X=[]
    for p in range(48):
        theta= p*np.pi/24
        [r,x,y]=point_elli(theta,aa,bb)
        #print('r',r)
        X.append([x,y,0])
    
    X=np.array(X)
    #after horizontal rotate with t
    t=np.array([0,0,t])
    [R,R1,R2,R3]=frrot3(t)
    X=X*R
        #after origon moving
    X[:,0]=X[:,0]+X0
    X[:,1]=X[:,1]+Y0
    X[:,2]=X[:,2]+Z0
    return [X]
#35
#Function to draw the profile 
#inputs the profile file storage path, the profile file name, the filtered ground points file and a path and name to save the resulted drawing)
def drawprofile(path,profile,path2,ground,imagepath,image):
    #try:
    #    list1=readtolist(path2,profile)
    #    [x,y,z]=np.array(list1).T
    #except:
    #list1=reducetolist(path2,profile)
    [x,y,z]=np.array(reducetolist(path,profile)).T
    x=x[0]
    try:
        [xg,yg,zg]=np.array(readtolist(path2,ground)).T
    except:
        xg=[]
        yg=[]
        zg=[]
        xmean=str(round(x, 2))
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    xg=[]
    
    ax.set_xlabel("Y",fontsize=12)
    ax.set_ylabel("Z",fontsize=12)
    ax.grid(True,linestyle='-',color='0.75')
    # scatter with colormap mapping to z value
    ax.scatter(y,z,s=25,c=z, marker = '*',edgecolors='none')
    if yg != []:
        xmean=str(round(np.mean(xg),2))
        #ax.scatter(yg,zg,s=20, marker = 'o', color='r')
        ax.plot(yg, zg, marker='D', linestyle='--', color='r', label='dtm')
    ax.set_title(image[0:-4]+",X="+xmean,fontsize=14)
    y=[]
    z=[]
    image2=imagepath+image
    plt.savefig(image2)    
    #plt.show(block=False)
    #import subprocess
    #subprocess.Popen([image, imagepath])
    plt.close('all')
#36
#function for drawing points in a list=x
def drawx(x):
    x=np.array(x)
    fig = plt.figure()
    ax = Axes3D(fig)
    sequence_containing_x_vals = x[:,0]
    sequence_containing_y_vals = x[:,1]
    sequence_containing_z_vals = x[:,2]
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
    plt.xlabel('xlabel', fontsize=18)
    plt.ylabel('ylabel', fontsize=16)
    #pyplot.zlabel('zlabel', fontsize=16)
    plt.show(block=False)
#37
def drawcyl(x1,x2,scatterx=[]):
    ax = plt.axes(projection='3d')
    for i in range(np.size(x1)/3):
        #print(i)
        ax.plot([x1[i,0] ,x2[i,0]],[x1[i,1] ,x2[i,1]],[x1[i,2] ,x2[i,2]],c='g')
        
    ax.plot(np.array(x1[:,0]), np.array(x1[:,1]),np.array(x1[0,2]), '-b',c='r')
    ax.plot([x1[0,0] ,x1[-1,0]],[x1[0,1] ,x1[-1,1]],[x1[0,2] ,x1[-1,2]], '-b',c='r')
    ax.plot(np.array(x2[:,0]), np.array(x2[:,1]),np.array(x2[0,2]), '-b',c='b')###########################################################################
    ax.plot([x2[0,0] ,x2[-1,0]],[x2[0,1] ,x2[-1,1]],[x2[0,2] ,x2[-1,2]], '-b',c='b')
    sequence_containing_x_vals = scatterx[:,0]
    sequence_containing_y_vals = scatterx[:,1]
    sequence_containing_z_vals = scatterx[:,2]
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals, c='B',marker='o')
    plt.show(block=false)
    #################################################################################################################################################
    #plt.show(block=False)
    #ax.plot(x, y, z, '-b')
#38
def drowcylinder(x0n,an,rn,Zmin,Zmax,Xpoints):
    #from . import materials
    #materials.rough = materials.diffuse
    position=[x0n[0,0][0,0][0,0],x0n[1,0][0,0][0,0],x0n[2,0][0,0][0,0]]
    Dz=-(Zmax-Zmin)
    R=Dz/cos(an[2])
    Dx=float(-R*an[0])
    Dy=float(-R*an[1])
    k=cylinder(pos=vector(position),axis=(Dx,Dy,Dz), radius=rn,color=color.orange, opacity=0.5)
    points(pos=Xpoints, size=50, color=color.red)
    arrow(pos=(0,0,0), axis=(5,0,0), color=(1,0,0), shaftwidth=0.1)    
    arrow(pos=(0,0,0), axis=(0,5,0), color=(0,1,0), shaftwidth=0.1)    
    arrow(pos=(0,0,0), axis=(0,0,5), color=(0,0,1), shaftwidth=0.1)    
    grid_xy = make_grid(1, 5)
    grid_xz = make_grid(1, 5)
    grid_xz.rotate(angle=pi/2, axis=(1,0,0), origin=(0,0,0))
    grid_yz = make_grid(1, 5)
    grid_yz.rotate(angle=-pi/2, axis=(0,1,0), origin=(0,0,0))
    k.rotate(angle=pi, axis=(0,0,0), origin=(0,0,0))
    #%xtitle='X', ytitle='Y', 
    return [Dx,Dy,Dz]
#39
def ellipseVertices(X0,Y0,Z0,aa,bb,t,theta):
    #X=draw_elli(X0,Y0,Z0,aa,bb,t,R)
    #this function is to find the vertices of the ellipde and reprojects them to the global coordinate system
    #before rotate
    X=[]
    for p in range(4):
        theta1= p*pi/2
        [r,x,y]=point_elli(theta1,aa,bb)
        #print('r',r)
        X.append([x,y,0])
    
    X=np.array(X)
    #after horizontal rotate with t
    t=np.array([0,0,t])
    [R,R1,R2,R3]=frrot3(t)
    X=X*R
        #after origon moving
    X[:,0]=X[:,0]+X0
    X[:,1]=X[:,1]+Y0
    X[:,2]=X[:,2]+Z0
        #after inverse projection
    [R,R1,R2,R3]=frrot3(-theta)
    X=X*R
    return [X]
winsound.Beep(700, 250)
#40
def ElFitError(centrespath):
        A1=FindFilesInFolder(centrespath,'H*')
        for file1 in A1:
            centresHights=readtolist(outpath+'Centres/',file1,8) 
            for layer in centresHights:
                C=(layer[7]-layer[2]*layer[3]*np.pi)/layer[7]
                C
                Addtofile(centrespath,'test.csv',[[layer[5],C]],2)
#41
def frrot3(theta,U0=numpy.matrix(numpy.identity(3))):
    ct = np.mat([[math.cos(theta[0])],[math.cos(theta[1])],[math.cos(theta[2])]])
    st = np.mat([[math.sin(theta[0])],[math.sin(theta[1])],[math.sin(theta[2])]])
    if max((theta.shape)) > 0:
        R1 = np.mat([[1,0,0],[0,ct[0],-st[0]],[0,st[0],ct[0]]])
        R = R1;
    if max((theta.shape)) > 1:
        R2 = np.mat([[float(ct[1]),0,-st[1]],[0,1,0],[st[1],0,ct[1]]]);
        R = R2*R;
    
    if max((theta.shape)) > 2:
        R3 = np.mat([[float(ct[2]),-st[2],0],[st[2],ct[2],0],[0,0,1]]);
        R = R3*R;
    R = R*U0;
    #checked and vetted
    return (R,R1,R2,R3)
#42
def fgrrot3(theta, R0=numpy.matrix(numpy.identity(3))):
    R, R1, R2, R3 = frrot3(theta, R0);
    dR1, dR2, dR3 = drrot3(R1, R2, R3);
    DR1 = R3*R2*dR1*R0;
    DR2 = R3*dR2*R1*R0;
    DR3 = dR3*R2*R1*R0;
    #checked and vetted
    return (R,DR1,DR2,DR3)
winsound.Beep(800, 250)
#43
def find_in_list_of_lists(list1,find_what):
    return1=[]
    u=0
    for i in list1:
        u+=1
        if find_what in i:
            return1.append(u-1)
    return return1
#44
def _find_max_eigval(S):
    """
    Finds the biggest generalized eigen value of the system  
    S * x = l * C * x  
    where
    ::
        C = | 0  0 2 |
            | 0 -1 0 |
            | 2  0 1 |
    Parameters:
    -----------    
    S : 3x3 matrix
    Returns:
    --------
    the highest eigen value
    """
    a = S[0,0]
    b = S[0,1]
    c = S[0,2]
    d = S[1,1]
    e = S[1,2]
    f = S[2,2]
    # computes the coefficients of the caracteristique polynomial
    # det(S - x * C) = 0
    # Since the matrix is 3x3 we have a 3rd degree polynomial
    # _a * x**3 + _b * x**2 + _c * x + _d
    _a = -4
    _b = 4 * (c - d)
    _c = a * f - 4 * b * e + 4 * c * d - c * c
    _d = a * d * f - b * b * f - a * e * e + 2 * b * c * e - c  * c * d
    # computes the roots of the polynomial
    # there must be 2 negative roots and one
    # positive, i.e. the biggest one.
    x2, x1, x0 = sorted(np.roots([_a, _b, _c, _d] ))
    return x0
winsound.Beep(600, 250)
#45
def _find_max_eigvec(S):
    """
    Computes the positive eigen value and the corresponding
    eigen vector of the system:
        
        S * x = l * C * x
    
        where
        ::
        
            C = | 0  0 2 |
                | 0 -1 0 |
                | 2  0 1 |
                
    Parameters:
    -----------    
    S : 3x3 matrix
    
    Returns:
    --------
        (l, u)
    
    l : float
        the positive eigen value
    
    u : the corresponding eigen vector
    """
     
    l = _find_max_eigval(S)

    a11 = S[0,0]
    a12 = S[0,1]
    a13 = S[0,2]
    a22 = S[1,1]
    a23 = S[1,2]
    
    u = np.array([
        a12 * a23 - (a13  - 2*l) * (a22 + l),
        a12 * (a13  - 2*l) - a23 * a11,
        a11 * (a22 + l) - a12 * a12
    ])

    c = 4 * u[0] * u[2] - u[1] * u[1]
    
    return l, u/np.sqrt(c)
#46
#another ellipse fitting code
#reference:  https://code.google.com/p/fit-ellipse/source/browse/trunk/fit_ellipse.py
""" 2D Ellipse fitting a,b,c,d,e,f and aa,bb,t
"""
def _find_max_eigval(S):
    a,b,c,d,e,f= S[0,0],S[0,1],S[0,2],S[1,1],S[1,2],S[2,2]
    _a = -4
    _b = 4 * (c - d)
    _c = a * f - 4 * b * e + 4 * c * d - c * c
    _d = a * d * f - b * b * f - a * e * e + 2 * b * c * e - c  * c * d
    x2, x1, x0 = sorted(np.roots([_a, _b, _c, _d] ))
    return x0
#47
def _find_max_eigvec(S):
    l = _find_max_eigval(S)
    a11,a12,a13,a22,a23 = S[0,0],S[0,1],S[0,2],S[1,1],S[1,2]
    u = np.array([
        a12 * a23 - (a13  - 2*l) * (a22 + l),
        a12 * (a13  - 2*l) - a23 * a11,
        a11 * (a22 + l) - a12 * a12
    ])

    c = 4 * u[0] * u[2] - u[1] * u[1]
    
    return l, u/np.sqrt(c)
#''' end of ellipse functions'''
# slice to image where Xmin,Xmax,Ymin,Ymax are given
winsound.Beep(900, 250)
#48
def find4pts(SegmentsFilesPath):
    #finding files
    segments=FindFilesInFolder(SegmentsFilesPath,'*csv')
    B=[]
    for sg1 in segments:
        sg1
        sgment=np.array(readtolist(SegmentsFilesPath,sg1))
        sgment1=[[np.max(sgment[:,0]),np.max(sgment[:,1]),np.max(sgment[:,2]),np.max(sgment[:,3]),np.max(sgment[:,4]),np.max(sgment[:,5])],
                 [np.min(sgment[:,0]),np.max(sgment[:,1]),np.min(sgment[:,2]),np.max(sgment[:,3]),np.max(sgment[:,4]),np.max(sgment[:,5])],
                 [np.max(sgment[:,0]),np.min(sgment[:,1]),np.max(sgment[:,2]),np.min(sgment[:,3]),np.max(sgment[:,4]),np.max(sgment[:,5])],
                 [np.min(sgment[:,0]),np.min(sgment[:,1]),np.min(sgment[:,2]),np.min(sgment[:,3]),np.min(sgment[:,4]),np.max(sgment[:,5])]]
        
        #arrange by X
        #sgment= sorted(sgment,key=lambda l:l[0])
        ##append the first and last points
        #B.append(sgment[0])
        #B.append(sgment[-1])
        #arrange by Y
        #sgment= sorted(sgment,key=lambda l:l[1])
        #append the first and last points
        #B.append(sgment[0])
        B.append(sgment1)
    return B
#49
def fgcylinder(a,X):
    m=X.shape[0]
    x0=float(a[0])
    y0=float(a[1])
    alpha=float(a[2])
    beta=float(a[3])
    s=float(a[4])
    R,DR1,DR2,DR3=fgrrot3((np.mat([alpha,beta,0])).T)
    Xt=(X-(np.mat(np.ones(m)).T)*np.mat([x0,y0,0]))*R.T
    Xt=np.array(Xt)
    xt=Xt[:,0]
    yt=Xt[:,1]
    rt=np.sqrt([xt*xt+yt*yt])#matlab>>rt = sqrt(xt.*xt + yt.*yt); 
    Nt=(np.mat(((xt/rt)[0],(yt/rt)[0],(np.zeros(m))))).T
    f=(np.mat(np.diag(np.dot(np.mat(Xt),Nt.T)))).T
    f=f-s
    J=np.mat(np.zeros((m,5)))
    A1=np.mat(np.ones((m,1)))*(R*(np.mat([-1,0,0])).T).T
    J[:,0]=(np.mat(np.diag(np.dot(np.mat(A1),Nt.T)))).T
    A2=np.mat(np.ones((m,1)))*(R*(np.mat([0,-1,0])).T).T
    J[:,1]=(np.mat(np.diag(np.dot(np.mat(A2),Nt.T)))).T
    A3=(X-np.mat(np.ones((m,1)))*np.mat([x0,y0,0]))*DR1.T#A3 = (X - ones(m, 1) * [x0 y0 0]) * DR1'; 
    J[:,2]=(np.mat(np.diag(np.dot(np.mat(A3),Nt.T)))).T
    A4 = (X-np.mat(np.ones((m,1)))*np.mat([x0,y0,0]))*DR2.T
    J[:,3]=(np.mat(np.diag(np.dot(np.mat(A4),Nt.T)))).T
    J[:,4]=-1*np.ones((m,1))
    #checked and vetted
    return (f,J)
#50
#A function to find files of a given extension in a given directory or path
#folder1 is the given directory 
#and extention is the extension of the objected file type

def FindFilesInFolder(folder1,extension):                            
    dr=os.getcwd()
    os.chdir(folder1)
    files=glob.glob(extension)
    os.chdir(dr)
    return files
#51
def fit_ellipse(X):
    """ Fit an ellipse.
    Computes the best least squares parameters of an ellipse  expressed as:
        a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0
    Parameters
    ----------
    X : N x 2 array
        an array of N 2d points.
    Returns:
    --------
    an array containing the parameters:
        [ a , b, c, d, e, f]
    """
    x = X[:,0]
    y = X[:,1]
    # building the design matrix
    D = np.vstack([ x*x, x*y, y*y, x, y, np.ones(X.shape[0])]).T
    S = np.dot(D.T, D)
    S11 = S[:3][:,:3]
    S12 = S[:3][:,3:]
    S22 = S[3:][:,3:]
    S22_inv = inv(S22)
    S22_inv_S21 = np.dot(inv(S22), S12.T)
    Sc =  S11 - np.dot(S12, S22_inv_S21)
    l, a = _find_max_eigvec(Sc)
    b = - np.dot(S22_inv_S21, a)
    return np.hstack([a,b])
#52
# reference> http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
def fitEllipse(X):
    x = X[:,0][:,np.newaxis]
    y = X[:,1][:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a
#[X0,Y0,aa,bb,theta]=gen_elli(a,b,c,d,e,f) 
#53
def gen_elli(a,b,c,d,e,f):
    A = np.array([
        [ a, b/2 ],
        [b/2, c  ]
    ])    
    B = np.array([d,e])    
    w,u = eigh(A)
    Xc = solve(-2*A,B)
    r2 = -0.5 * np.inner(Xc,B) - f
    rr2 = r2 / w
    t = np.arccos(u[0,0])
    if t > np.pi/2:
        t = t - np.pi
        
    t *= np.sign(u[0,1])
    X0,Y0=tuple(Xc)
    aa,bb=tuple(np.sqrt(rr2))
    return [X0,Y0,aa,bb,t]
#54
def gen_elli2(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    X0=(c*d-b*f)/num
    Y0=(a*f-b*d)/num
    theta=0.5*np.arctan(2*b/(a-c))
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    aa=np.sqrt(up/down1)
    bb=np.sqrt(up/down2)
    return [X0,Y0,aa,bb,theta]
#or alternatively
#55
def gncc2(f0,f1,p,g,scale,tolr,scalef):
    inconv=0
    eps=2.2204e-16
    sp = float(max(abs((p[ii]*scale[ii])) for ii in xrange(p.shape[0])))
    sg = float(max(abs(ii) for ii in (g/ scale)))#(g/ scale));
    c=[sp/(scalef*tolr**.7)]
    delf=f0-f1
    c.append(np.fabs(delf)/(tolr*scalef))
    d3=(tolr**.7)*scalef
    d4=scalef*(eps**.7)
    d5=eps**.7*scalef
    c.append(sg/d3)
    c.append(f1/d4)
    c.append(sg/d5)
    if c[0]<1 and c[1]<0 and c[2]<1:
        inconv=1
    elif c[3]<1:
        inconv=1
    elif c[4]<1:
        inconv=1
    C=(np.mat(c))
    #checked and vetted
    return (C, inconv,sp,sg)
#56
def graph(formula, x_range):  
    x = np.array(x_range)  
    y = eval(formula)
    plt.plot(x, y)  
    plt.show(block=False)
#57
def gr(x,y=0):
    #x,y is 
    #U,c,s=gr(x,y)
    if y==0:
        c=1
        s=0
    elif math.fabs(y)>=math.fabs(x):
        t=x/y
        s=1/(1+t**2)**.5
        c=t*s
    else:
        t=y/x
        c=1/(1+t**2)**.5
        s=t*c
    #checked and vetted
    return (np.mat([[c,s],[-s,c]]),c,s)
#58
def ImagesThrushold(InPath,OutPath,thrushold):
    v=FindFilesInFolder(InPath,'*png')
    try:
        os.stat(OutPath)
    except:
        os.makedirs(OutPath)
    for image in v:
        #read image
        img = cv2.imread(InPath+'/'+image,0)
        #thrushold image
        #image_mat2=binary(img,thrushold,255)
        ret,image_mat2 = cv2.threshold(img,thrushold,255,cv2.THRESH_BINARY)
        #write image
        img = Image.fromarray(np.uint8(image_mat2),'L')
        img.save(OutPath+image)
#59
def img2np(path1,file1):
    f=path1+'/'+file1
    ds = gdal.Open(f.replace("//", "/"))
    return np.array(ds.GetRasterBand(1).ReadAsArray())
#60
def img2pandas(path1,file1):
    #open file
    src_dataset = gdal.Open(path1+'/'+file1)
    z = src_dataset.ReadAsArray()
    #read georeferencing
    (xmin,res1,tilt1,ymin,tilt2,res2)=src_dataset.GetGeoTransform()
    ys,xs=np.shape(z)
    x = np.array([list(np.linspace(xmin, xmin+(xs-1)*res1, xs))]*(ys))
    y = np.transpose(np.array([list(np.linspace(ymin, ymin+(ys-1)*res1, ys))]*(xs)))
    #z1=list(z.ravel())
    #y1=list(y.ravel())
    #1=list(x.ravel())
    data=np.array([list(x.ravel()),list(y.ravel()),list(z.ravel())])
    #'C:/Users/ahalboabidallah/Desktop/ash_farm_new/profiles/profiles/results/AllGround.tiff'
    return pd.DataFrame(data,index=['X','Y','Z']).transpose()
#61
def IntersectLayers2(Pathin,image1,image2,Pathout,imageout):
    try:
        os.stat(Pathout)
    except:
        os.makedirs(Pathout) 
    init1=r''+Pathin+image1
    init1=init1.replace('/', "\\")
    init2=r''+Pathin+image2
    init2=init2.replace('/', "\\")
    outit=r''+Pathout+imageout
    outit=outit.replace('/', "\\")
    gdal_it=r'C:\\gdal_calc.py '+ '-A '+init1+' -B '+init2+' --outfile='+outit+' --calc="255*(logical_and((A>0),(B>0)))"'
    os.system('python '+gdal_it)
#IntersectLayers2('C:/Users/ahalboabidallah/Desktop/ash_farm_new/x-25y0/images/','0tif.tif','10tif.tif','C:/Users/ahalboabidallah/Desktop/ash_farm_new/x-25y0/images/','del.tif')
#62
def IntersectLayers(Pathin,image1,image2,Pathout,imageout,inverted=1):
    from PIL import Image
    try:
        os.stat(Pathout)
    except:
        os.makedirs(Pathout)
    imagefile=Pathin+image1
    img1 = cv2.imread(imagefile,0)
    imagefile=Pathin+image2
    img2 = cv2.imread(imagefile,0)#
    if inverted==1:
        array1=((img1-255)*(img2-255))**.5
        array1=((array1-255)*-1)
    else:
        array1=((img1)*(img2))**.5
    array1=array1.astype(np.uint8)
    array1=binary(array1,254,255)
    cv2.imwrite(Pathout+'/'+imageout,array1)
    #IntersectLayers(Pathin,image1,image2,Pathout,imageout,inverted=1)
#63
def Is_angle(X,Y,SegmentsFilesPath,SID,minAngleDegree):
    #find4pts(SegmentsFilesPath):B #closestSegment(x,y,B)>SS, #ptsByB(B,SID):(pts), 
    P=Read_Gegment(SegmentsFilesPath,SID)
    P=np.array(P)[:,0:2]
    if point_in_poly(X,Y,P)!=True:# when the point is outside the segment then the segment is toward the direction of its centre of gravity
        angles=[]
        XG,YG=center_of_mass(SegmentsFilesPath,SID)
        AG=Az_C2p(X,Y,XG,YG)#angle to the CG of the segment
        for point in P:
            XP,YP=point
            if XP==X and YP==Y: # rear case but could happen# The coordinates could be shifted with a tiny distance
                XP=XP+0.01
                YP=YP+0.01
            AP=Az_C2p(X,Y,XP,YP)-AG
            if AP < 0.0:
                AP=AP+2*np.pi
            angles.append([AP])
        angles=sorted(angles,key=lambda l:l[0])
        #find the maximum gap
        for i in range(len(angles)-1):
            if abs(angles[i][0]-angles[i+1][0])>np.pi-.001:
                A1=angles[i][0]
                Aa=angles[i+1][0]
        try:        
            AA=2*np.pi-abs(Aa-A1)
            print (AA*180/np.pi)
        except:
            print('->180')
            AA=np.pi
        if AA>minAngleDegree*np.pi/180:
            theta=True
        else:
            theta=False
    else:
        theta=True
        print ('>180')
    return theta
#64
#Function to find if the list is existed or creat an empty one/ a fancy function :)
#inputs list name
winsound.Beep(400, 250)

def isexist(mylist):
    try:
        mylist
        exec('itis='+mylist)
    except NameError:
        itis=[]
    return itis
winsound.Beep(300, 250)
#65
#function to find if a segment pixel lies on the line between the detected center and current pixel it also retrns the distance between the center and the tested pixel
def IsIntersect(xCtr,yCtr,iPixeltarget,jPixeltarget,iPixeltested,jPixeltested,XminImage,YminImage,step):
    #inverse the center point to the image coordinates
    XCtr=(xCtr-XminImage)/step
    YCtr=(yCtr-YminImage)/step
    dist1=dist(XCtr,YCtr, iPixeltarget,jPixeltarget,iPixeltested,jPixeltested)
    PtCtrDistance=((XCtr-iPixeltested)**2+(YCtr-jPixeltested)**2)**0.5
    if dist1<step/2:
        return [1,PtCtrDistance]
    else:
        return [0,PtCtrDistance]
#66
def Intersect_polyline_layer(pointlistXYZ,layerNo):
    pointlistXYZ=sorted(pointlistXYZ,key=lambda l:l[-1])
    n=len(list(filter(lambda a: a[-1] <= layerNo, pointlistXYZ)))-1
    if layerNo>=min(np.array(pointlistXYZ)[:,-1]) and layerNo<max(np.array(pointlistXYZ)[:,-1]):
        #xc=X+DX*dz/DZ
        Xc=float(pointlistXYZ[n][0])+float(pointlistXYZ[n+1][0]-pointlistXYZ[n][0])*(float(layerNo-pointlistXYZ[n][-1])/float(pointlistXYZ[n+1][-1]-pointlistXYZ[n][-1]))
        Yc=float(pointlistXYZ[n][1])+float(pointlistXYZ[n+1][1]-pointlistXYZ[n][1])*(float(layerNo-pointlistXYZ[n][-1])/float(pointlistXYZ[n+1][-1]-pointlistXYZ[n][-1]))
        D1=True
    else:
        Xc,Yc=1000000,1000000
        D1=False
    return D1,Xc,Yc
#67
def list2image(serpts,imagepath, imagefile, step1=0.05):
    # creatarray to be image
    serpts1=np.array(serpts)
    try:
        os.stat(imagepath)
    except:
        os.makedirs(imagepath) 
    Xmin1,Xmax1,Ymin1,Ymax1=min(serpts1[:,0]),max(serpts1[:,0]),min(serpts1[:,1]),max(serpts1[:,1])
    imin,imax,jmin,jmax=min(serpts1[:,2]),max(serpts1[:,2]),min(serpts1[:,3]),max(serpts1[:,3])
    image_mat=np.zeros([jmax-jmin+3,imax-imin+3]).astype(int)#+3 :i index and j index  starting at 0 not 1, two extra rows and 2 extra columns
    for row1 in  serpts:
        image_mat[row1[3]-jmin+1,row1[2]-imin+1]=255
    array2raster(imagepath,imagefile+'.tif',Xmin1,Ymin1,step1,step1,np.transpose(image_mat))
#68
 def list2segmentpoints(PathtoSegmentationFiles, layerPath):
    layers=FindFilesInFolder(layerPath,'layer*')#[1:3]
    for layer in layers:
        print(layer)
        try:
            segments=FindFilesInFolder(PathtoSegmentationFiles+layer[:-4],'seg*')
            layerpoints=readtolist(layerPath,layer,5) 
            for segment1 in segments[1:]:
                segfile=[]
                winsound.Beep(900, 10)
                print(segment1)
                pixels=readtolist(PathtoSegmentationFiles+layer[:-4]+'/',segment1,5) 
                for pixel in pixels: 
                    pixelpoints=list(filter(lambda a:abs((a[0]-pixel[0]-step))<=step and abs((a[1]-pixel[1]-step))<=step,layerpoints))#
                    segfile.extend(pixelpoints)
                writetofile(PathtoSegmentationFiles+'/'+layer[:-4]+'/','point'+segment1,segfile,5)
        except:
            print('layer ',layer,' not segmented')
            winsound.Beep(100, 500)
        #save segment file
winsound.Beep(1100, 250)   
#69
#another aproach for slicing according to slices parallel to ground
def listztin(path,shp):#list with distance from the tin
    shp=path+shp
    sf = shapefile.Reader(shp)
    shapes = sf.shapes()
    records=sf.records()
    list1=[]
    for i in range(len(shapes)):
        x=shapes[i].points[0][0]
        y=shapes[i].points[0][1]
        ztin=shapes[i].z[0]
        zlidar=float(records[i][0])
        ID=int(float(records[i][1]))
        Dz=zlidar-ztin
        list4add=[x,y,zlidar,Dz,ID]
        list1.append(list4add)
    return(list1)
#70
def lscylinder(X,x0,a0,r0,tolp, tolg,w=1):
    #x0n, an, rn, d, sigmah, conv, Vx0n, Van, urn, GNlog,a, R0, R = lscylinder(X, x0, a0, r0, tolp, tolg, w)
    m = X.shape[0]#m = size(X, 1);
    R0=rot3z(a0)#
    x1=R0*x0#
    xb=[np.mean(X[:,0]),np.mean(X[:,1]),np.mean(X[:,2])]#xb = mean(X)';
    xb=(np.mat(xb)).T
    xb1=R0*xb#
    t=x1+((xb1[2]-x1[2])*(np.mat([[0,0,1]]))).T#t = x1 + (xb1(3) - x1(3)) * [0 0 1]'
    X2=(X*R0.T) - (np.mat(np.ones(X.shape[0]))).T*t.T#X2 = (X * R0') - ones(m ,1) * t'; 
    x2=x1-t#
    xb2=xb1-t#
    ai=(np.mat([0,0,0,0,r0])).T
    tol=np.mat([tolp,tolg])
    #ok
    #nlss11
    a,d,R,GNlog,conv=nlss11(ai, tol, X2, w)
    rn = a[4]; 
    aaa=np.array([float(a[2]),float(a[3]),0])
    R3, DR1, DR2, DR3= fgrrot3(aaa); 
    an = R0.T * R3.T * (np.mat([0,0,1])).T 
    p = R3 * (xb2 - np.mat([a[0],a[1],0]).T); 
    pz = (np.mat([0,0,p[2]])).T; 
    x0n = R0.T * (t + (np.mat([a[0],a[1],0])).T + R3.T * pz); 
    #nGN = GNlog.shape; 
    #nGN=nGN[0]
    #conv = GNlog(nGN, 1); 
    if conv == 0 :
        print( '*** Gauss-Newton algorithm has not converged ***')
    dof = m - 5;
    sigmah = numpy.linalg.norm(d)/dof**.5; 
    ez =(np.mat([0,0,1])).T
    G = np.zeros((7, 5)) 
    dp1 = R3 * (np.mat([-1,0,0])).T
    dp2 = R3 * (np.mat([0,-1,0])).T
    dp3 = DR1 * (xb2 - (np.mat([a[1],a[2],0])).T )
    dp4 = DR2 * (xb2 - (np.mat([a[1],a[2],0])).T )
    G[0:3, 0] = (R0.T * (np.mat([1,0,0])).T + R3.T * (np.mat([0,0,dp1.T*ez])).T).T
    G[0:3, 1] = (R0.T * (np.mat([0,1,0])).T + R3.T * (np.mat([0,0,dp2.T*ez])).T).T
    G[0:3, 2] = (R0.T * (DR1.T * (np.mat([0,0,p.T*ez])).T) + R3.T * (np.mat([0,0,dp3.T*ez])).T ).T
    G[0:3, 3] = (R0.T * (DR2.T * (np.mat([0,0,p.T*ez])).T) + R3.T * (np.mat([0,0,dp4.T*ez])).T ).T
    G[3:6, 2] = (R0.T * DR1.T * (np.mat([0,0,1])).T).T 
    G[3:6, 3] = (R0.T* DR2.T * (np.mat([0,0,1])).T).T
    G[6, 4]= 1 
    #Gt = R'\(sigmah * G')==x = A\l >> Ax=l>>X=inv(A'A)A'l>>Gt=(R'.T*R').I*R''*(sigmah * G')
    Gt=(np.mat(R*R.T)).I*R*(sigmah * G.T)
    
    Va = Gt.T * Gt 
    Vx0n = Va[0:3, 0:3]
    Van = Va[4:7, 4:7]
    urn = Va[6, 6]**.5
    print('x0n(Estimate of the point on the axis) is',x0n)
    print('an (Estimate of the axis direction)  is',an)
    print('rn (Estimate of the cylinder radius) is',rn)
    return (x0n, an, rn, d, sigmah, conv, Vx0n, Van, urn, GNlog,a, R0, R)
#71
def loadDataFromCSV(csvfile, dRange, uRange):
    """
        Read data from csvfile. Data is between the columns dRange and uRange
    """
    csv = __import__('csv')
    dataReader = csv.reader(open(csvfile), delimiter=',')
    data = []
    indexes = range(dRange,uRange)
    for index in indexes:
        data.append([])
    for row in dataReader:
        for i, index in enumerate(indexes):
            data[i].append(float(row[index]))
    return np.array(data)
#def local_wighting_branch(xc,yc,path1,image1,step1=0.05):
#    XYZ=img2pandas(path1,image1)
#    D=max(((XYZ[0]-xc)**2+(XYZ[1]-yc)**2)**0.5)
#    XYZ['W']=(D-(((XYZ[0]-xc)**2+(XYZ[1]-yc)**2)**0.5))*XYZ[2]
#    ds = gdal.Open(path1+image1)
#    (xmino,res1,tilt1,ymino,tilt2,res2)=ds.GetGeoTransform()
#    IMG=img2np(path1,image1)
#49
#function: cylinder drawing and drawing requiremwnt function set 
def make_grid(unit, n):
        nunit = unit * n
        f = frame()
        for i in xrange(n+1):
            if i%5==0: 
                color = (1,1,1)
            else:
                color = (0.5, 0.5, 0.5)
            curve(pos=[(0,i*unit,0), (nunit, i*unit, 0)],color=color,frame=f)
            curve(pos=[(i*unit,0,0), (i*unit, nunit, 0)],color=color,frame=f)
        return f
#72
def MakeSegmentsFiles(SegmentedImagePath,SegmentedImage,SegmentsFilesPath,Xmin,Ymin,step1):   
    imagefile=SegmentedImagePath+SegmentedImage
    outlist=[]
    im1=numpy.array(vigra.impex.readImage(imagefile, dtype='', index=0, order=''))[...,0]
    im1=im1.astype(int)# segmented to int seg numbers
    #make folder
    try:
        os.stat(SegmentsFilesPath)
    except:
        os.makedirs(SegmentsFilesPath)
    lenim=np.shape(im1)[0]*np.shape(im1)[1]
    im2=np.reshape(im1, (lenim,1)).tolist()
    im2 = [val for sublist in im2 for val in sublist]
    im2=list(set(im2))
    im2.remove(0)
    for seg in im2:
        #x,y,i,j,seg,SegmentedImage
        #find where it is in im1
        ij=pd.DataFrame(np.transpose(np.where(im1==seg)))
        ij['x']=step1*(ij[0])+Xmin
        ij['y']=step1*(ij[1])+Ymin
        ij['seg']=[int(seg)]*np.shape(np.transpose(np.where(im1==seg)))[0]
        try:
            ij['layer']=[int(SegmentedImage[:-7])]*np.shape(np.transpose(np.where(im1==seg)))[0]
        except:
            ij['layer']=15
        #arrange
        ij = ij[['x','y',0,1,'seg','layer']]
        #write to csv
        ij.to_csv(SegmentsFilesPath+'seg'+str(int(seg))+'.csv',index=False,header=False)    
#73
def MakeSegmentsFiles1(SegmentedImagePath,SegmentedImage,SegmentsFilesPath,Xmin,Ymin,step):   #old version
    imagefile=SegmentedImagePath+SegmentedImage
    outlist=[]
    im1=vigra.impex.readImage(imagefile, dtype='', index=0, order='')
    im1=numpy.array(im1[...,0])
    #im1 = cv2.imread(imagefile,0)
    #make folder
    try:
        os.stat(SegmentsFilesPath)
    except:
        os.makedirs(SegmentsFilesPath)
    j=0
    for line in im1:
        j+=1
        i=0
        for pixel in line:
            i+=1
            if pixel>0:
                X=step*(i-1)+Xmin
                Y=step*(j-1)+Ymin
                outlist.append([X,Y,i-1,j-1,pixel])
    outlist=sorted(outlist,key=lambda l:l[4])
    SEGi=[]
    seg=0
    for pixel in outlist:
        if pixel[4]==seg:
            SEGi.append(pixel)
        else:
            writetofile(SegmentsFilesPath,'seg'+str(seg)+'.csv',SEGi,5)
            seg+=1
            SEGi=[pixel]
    try:
        os.remove(SegmentsFilesPath+"seg0.csv")
    except:
        pass
# for each segment find points that have minX, maaxX,minY,Maxy
#74
def norm(ar):
    return 255.*np.absolute(ar)/np.max(ar)
#75
def np_from_img(fname):
    return np.asarray(Image.open(fname), dtype=np.float32)
#76
def nlss11(ai,tol,p1,p2):
    #a, f, R, GNlog,conv= nlss11(ai, tol, p1, p2)
    """
    a
    f
    R
    GNlog
    conv
    """
    a0 = ai
    n = max(a0.shape)
    if n == 0:
        print('Empty vector of parameter estimates:')
    
    mxiter=100+math.ceil(n**.5)
    conv=0.0
    niter=0.0
    eta=.01
    GNlog=[]
    while niter<mxiter and conv ==0:
        #print('niter=',niter)
        f0,J=fgcylinder(a0,p1)
        if niter==0:
            (mJ,nJ)=J.shape
            scale=np.zeros((nJ,1))
            for j in xrange(nJ):
                scale[j]=numpy.linalg.norm(J[:,j])
        m=max(f0.shape)
        if niter==0 and m<n:
            'number of obsurvation is not enough nlssll'
        F0=numpy.linalg.norm(f0)
        qr1=((J.T).tolist())
        qr2=((f0.T).tolist())
        qr1.append(qr2[0])
        qr3=list(map(list, zip(*qr1)))
        fqr,Ra=scipy.linalg.qr(np.mat(qr3))
        ff=fqr.tolist()
        ff2=Ra.tolist()
        for ii in ff2:
            ff.append(ii)
        
        #Ra=np.triu(fq)#Ra = triu(qr([J, f0]));
        R=Ra[0:nJ,0:nJ]# R = Ra(1:nJ,1:nJ);
        q=(np.mat((Ra[0:nJ,nJ]))).T#q = Ra(1:nJ,nJ+1); 
        #p=(np.mat((np.dot(R.T, R)))).I*np.dot((np.mat(R)).T, q)#p = -R\q;
        p=-scipy.linalg.solve(R,q)
        g=2*R.T*q
        G0=float(g.T*p)
        a1=a0+p
        niter=niter+1
        f1,J2=fgcylinder(a1,p1)
        F1=np.linalg.norm(f1)
        if type(tol) != list:
            tol=tol.tolist()
        #print('F0,F1,p,g,scale,float(tol[0]),float(tol[1])',F0,F1,p,g,scale,float(tol[0]),float(tol[1]))
        c,conv,sp,sg=gncc2(F0,F1,p,g,scale,tol[0][0],tol[0][1])
        #print(conv)
        if conv != 1:
            rho=(F1-F0)*(F1+F0)/G0
            if float(rho) < eta:
                tmin= max([0.001,1/(2*(1-rho))])
                a0=a0+float(tmin)*p
            else:
                a0=a0+p   
        GNlog.append([niter,F0,sp,sg])
    a=a0+p
    f=f1
    GNlog.append([conv,F1,0,0])
    return (a,f,R,GNlog,conv)
#77
#function open file with windows
def openfile():
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    myfile = askopenfilename(title='original Points File')
    return myfile
#78
def point_elli(theta,aa,bb):
    #[r,x,y]=point_elli(theta,aa,bb)
    r=aa*bb/((bb*math.cos(theta))**2+(aa*math.sin(theta))**2)**0.5
    x=r*math.cos(theta)
    y=r*math.sin(theta)
    return [r,x,y] 
#79
#referance: http://geospatialpython.com/2011/01/point-in-polygon.html
def point_in_poly(x,y,poly):
    n = len(poly)
    inside = False
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y
    return inside
#80
def points2segments(Xmin,Ymin,step,pointfilepath,pointfile,segmentsfilepath,SegmentedImagePath,SegmentedImage):
    pts=readtolist(pointfilepath,pointfile,NoOfColumns=3)
    pts=sorted(pts,key=lambda l:l[0])
    pts=sorted(pts,key=lambda l:l[1])
    img = cv2.imread(SegmentedImagePath+SegmentedImage,0)
    myfile= open(pointfilepath+pointfile,'r')
    for row in myfile:
        row=row.split(",")
        try:
            x,y,zlidar,ztin=float(row[0]),float(row[1]),float(row[2]),float(row[3])
        except:
            pass
        i=int(math.floor(((x-Xmin)/step)))
        j=int(math.floor(((y-Ymin)/step)))
        segment=img(i,j)
        C=open(pointfilepath+str(segment)+'.csv','a')
        C.write(str(x))
        C.write(',')
        C.write(str(y))
        C.write(',')
        C.write(str(zlidar))
        C.write(',')
        C.write(str(ztin))
        C.write('\n')
#81
# project points into isometric plane to fit an ellipse
def projection(an,X):
    X1=np.mat(X)
    #(Xprojected,theta)=projection(an,X)
    theta=np.array([math.acos(an[0])+np.pi/2,math.acos(an[1])+np.pi/2,math.acos(an[2])])
    R,R1,R2,R3=frrot3(theta)
    X2=np.array((X1*R))
    return(X2,theta)
#82
def princomp(A,numpc=1):
    """PCA using linalg.eig, from http://glowingpython.blogspot.it/2011/07/pca-and-image-compression-with-numpy.html"""
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
    [latent,coeff] = linalg.eig(cov(M))#latent, a vector containing the eigenvalues of the covariance matrix of A
    p = size(coeff,axis=1)
    idx = argsort(latent) # sorting the eigenvalues
    idx = idx[::-1]       # in ascending order
    # sorting eigenvectors according to the sorted eigenvalues
    coeff = coeff[:,idx]
    latent = latent[idx] # sorting eigenvalues
    if numpc < p or numpc >= 0:
        coeff = coeff[:,range(numpc)] # cutting some PCs
    score = dot(coeff.T,M) # projection of the data in the new space
    #coeff is the rotating matrix
    return coeff,score,latent
    #coeff,score,latent=princomp(A,3)
#83
def profile_filter2(inputPath,profilefile,interval,OutputPath):
    list1=readtolist(inputPath,profilefile)
    try:
        ymin=min(np.array(list1)[:,1])
        ymax=max(np.array(list1)[:,1])
        listg=[]
        for i in range(int((ymax-ymin)/interval)):
            list2=list(filter(lambda x: x[1]<=ymin+i*interval and x[1]>ymin+(i-1)*interval, list1))
            list2=sorted(list2,key=lambda l:l[2])
            try:
                listg.append(list2[0][:3])
            except:
                pass
        Addtofile(OutputPath,'g'+profilefile,listg,3)
    except:#
        print('error in: '+profilefile)    
winsound.Beep(500, 250)
#84
def profiles(pointlistXYZ,OutputPath,xmin,xmax,spacer=2,width=0.02):
    #xmin=min(np.array(pointlistXYZ)[:,0])
    #xmax=max(np.array(pointlistXYZ)[:,0])
    k=math.ceil((xmin--1000)/spacer)# the absolute origin is -1000
    for i in range(int((xmax-xmin)/spacer)):
        k+=1
        list2=list(filter(lambda x: x[0]<k*spacer-1000+width/2 and x[0]>k*spacer-1000-width/2, pointlistXYZ))
        Addtofile(OutputPath,'Profile_No'+str(k)+'.csv',list2,3)
winsound.Beep(700, 150)
#85
def parinting(path, parentlayer,childlayer):
    parent=np.array(vigra.impex.readImage(path+parentlayer, dtype='INT16', index=0, order='')[:,:, 0])
    child=np.array(vigra.impex.readImage(path+childlayer, dtype='INT16', index=0, order='')[:,:, 0])
    child=np.column_stack((np.zeros([np.shape(child)[0],1]),child,(np.zeros([np.shape(child)[0],1]))))
    child=   np.row_stack((np.zeros([1,np.shape(child)[1]]),child,(np.zeros([1,np.shape(child)[1]]))))
    list1=[]
    lenim=np.shape(parent)[0]*np.shape(parent)[1]
    parent=np.reshape(parent, (lenim,1)).astype(int)
    #child1=np.reshape(np.array(child), 1)
    shifts=[[0,0],[0,1],[0,-1],[1,0],[1,1],[1,-1],[-1,0],[-1,1],[-1,-1]]
    list1=[]
    for shift in shifts:
        #[1+shift[0],np.shape(child)[0]-1+shift[0],1+shift[1],np.shape(child)[1]-1+shift[1]]
        shiftedchild=np.reshape(child[1+shift[0]:np.shape(child)[0]-1+shift[0],1+shift[1]:np.shape(child)[1]-1+shift[1]], (lenim,1)).astype(int)
        list1.extend((np.column_stack((parent,shiftedchild))).tolist())
    list1=list(filter(lambda a: a[0] != 0 and a[1] != 0, list1))
    list1=sorted(list1,key=lambda l:l[0])
    list1=sorted(list1,key=lambda l:l[1])
    #list1=list(set(list1))
    list1.sort()
    list1=list(list1 for list1,_ in itertools.groupby(list1))
    #list2=[]
    #for point in list1:
        #
        #open file to add child
        #Addtofile(path+'/'+parentlayer[0:-4]+'/','C'+str(int(point[0]))+'.csv',[str(int(point[1]))],1)
        #open file to add parent
        #Addtofile(path+'/'+childlayer[0:-4]+'/','P'+str(int(point[1]))+'.csv',[str(int(point[0]))],1)
    #writetofile(path,file1,list1,NoOfColumns=3)
    writetofile(path+'/'+parentlayer[0:-4]+'/','Parinting_C'+childlayer[0:-4]+'_P'+parentlayer[0:-4]+'.csv',list1)
#86
def PointsInSegment(layerPath,LayerFile,SegmentFilePath,segmentfile,Step):
    list1=readtolist(layerPath,LayerFile,5)
    list2=readtolist(SegmentFilePath,segmentfile,5)
    list2=sorted(list2,key=lambda l:l[0])
    xmin=list2[0][0]
    xmax=list2[-1][0]
    list2=sorted(list2,key=lambda l:l[1])
    ymin=list2[0][1]
    ymax=list2[-1][1]
    list1=list(filter(lambda a: a[1] <xmax, list1))
    list1=list(filter(lambda a: a[1] >xmin, list1))
    list1=list(filter(lambda a: a[2] <ymax, list1))
    list1=list(filter(lambda a: a[2] >ymin, list1))
    list3=[]
    for point in list1:
        D=[]
        for pixel in list2:
            D.append(((point[0]-pixel[0])**2+(point[1]-pixel[1])**2)**.5)
        if np.min(D)<=2**.5*Step:
            list3.append(point)
    return list3
#87
def PointsInSegment2(list1,SegmentFilePath,segmentfile,Step1):
    #FitPtslisti=PointsInSegment2(list1,'E:/castle/final/LR/noground/7/segmented/30/','seg'+str(int(segment1[0]))+'.csv',step)
    #list1=readtolist(layerPath,LayerFile,5)
    seg=readtolist(SegmentFilePath,segmentfile,5)
    addlist=[]
    for point in seg:
        point
        addlist.extend(list(filter(lambda a: a[1] <point[1]+2*step1 and a[1] >point[1]-2*step1 and a[0] <point[0]+2*step1 and a[0] >point[0]-2*step1 , list1)))#
    return addlist
#88
# a function to seperate profile points from a given point list and save them 
#to an ascii file in the spesified path 
# inputs are:- pointlistXYZ:the point coordinates list, 
# OutputPath= the spesified output profiles' file path, xmin,xmax= the minimum 
#and maximum X coordinates of the plot,
# spacer= the desired distance between each succesive profiles, 
#and width=the width of the profile slice.
def ProfilesPoints(pointlistXYZ,limits,xmin=0):
    pointlistXYZ=list(filter(lambda x: x[0]-xmin>float(round(x[0]-xmin))-limits and x[0]-xmin<float(round(x[0]-xmin))+limits, pointlistXYZ))
    return pointlistXYZ
#89
def ptsByB(B,SID):
    #B is the output of find4pts and SID is the segment ID
    pts=[]
    for i in B:
        if int(i[-1])==int(SID):
            pts.append(i)
    return(pts)
#90
def TreeExist(list1,centre,d):
    l,x,y,a,b,t,l2=centre
    list1=list(filter(lambda a: a[0] <x+d and a[0] >x-d, list1))
    list1=list(filter(lambda a: a[1] <y+d and a[1] >y-d, list1))
    return len(list1)
#91
def Read_Gegment(SegmentsFilesPath,SID):
    P=readtolist(SegmentsFilesPath,'seg'+str(int(SID))+'.csv',5)
    return(P)
#92
def read_raster(in_raster,band=1):
    in_raster=in_raster
    ds = gdal.Open(in_raster)
    data = ds.GetRasterBand(band).ReadAsArray()
    #data[data<=0] = np.nan
    gt = ds.GetGeoTransform()
    xres = gt[1]
    yres = gt[5]
    # get the edge coordinates and add half the resolution 
    # to go to center coordinates
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5
    del ds
    # create a grid of xy coordinates in the original projection
    xx, yy = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
    return data, xx, yy, gt
#93
#2- A function to import a file into a list
#inputs the file storage path and the file name
def readtolist(path,file1,NoOfColumns=3,alradyhasheader=0):
    df=readtopandas(path,file1,alradyhasheader=0)
    list1=df.values.tolist()
    return list1
winsound.Beep(200, 250)
#94
def readtopandas(path1,file1,alradyhasheader=0):#
    #F='C:/Users/ahalboabidallah/Desktop/test.csv'
    F=path1+file1
    #add header to the file if there is no header #
    if alradyhasheader==0:
        #generate a header 
        df = pd.read_csv(F,header=None)
    else:
        df = pd.read_csv(F)#needs a csv with a header line
    return df   
#95
# A function to resize np array
def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1) 
#96
#Function to import a huge file into a list to visulize the file
''' to visulize the file, not for analyses'''
#inputs the file storage path, the file name, and reduction factor(take one point each sth points)
def reducetolist(path,file1,sthpoints=50):
    F=path+file1
    read=open(F,'r')
    list1=[]
    c=0.0
    for line in read:
        c=c+1
        #print(c)
        if c/sthpoints==int(c/sthpoints):
            exec('x,y,z=' +line)
            list1.append([x,y,z])
    return list1
winsound.Beep(500, 250)
#97
def reversed_array(imagepath,imagefile1,originX,originY,pixelWidth,pixelHeight,array):
    reversed_arr = array[::-1] # reverse array so the tif looks like the array
    array2raster(imagepath+imagefile1,originX,originY,pixelWidth,pixelHeight,reversed_arr) # convert array to raster
#98
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
#99
def rot3z(a):
    #U=rot3z(a)
    W,c1,s1=gr(float(a[1]),float(a[2]))
    z=float(c1*a[1]+s1*a[2])
    V=np.mat([[1,0,0],[0,s1,-c1],[0,c1,s1]])
    W,c2,s2=gr(float(a[0]),float(z))
    W,c2,s2
    if float((c2*a[0]+s2*z))<0:
        c2=-c2
        s2=-s2
    
    W=np.mat([[s2,0,-c2],[0,1,0],[c2,0,s2]])
    U=W*V
    #checked and vetted
    return U
#100
def save_as_img(ar, fname):
    Image.fromarray(ar.round().astype(np.uint8)).save(fname)
#101
#a function to segment layers  
def segmentIm(inpath, image1, originX, originY, outpath, reverse=1,step1=0.05):  
    out_dot_tif_image='seg'+image1[0:-4]+'.tif'
    #print(out_dot_tif_image)
    arr1 =img2np(inpath,image1)
    segim=segment(arr1,reverse)
    #need changing
    #vigra.impex.writeImage(segim.astype(numpy.int16), outpath+out_dot_tif_image, dtype = '', compression = '', mode = 'w')
    array2raster(outpath,out_dot_tif_image,originX,originY,step1,step1,segim,type1=gdal.GDT_Float32)
    #segim=segmentIm('C:/Users/ahalboabidallah/Desktop/ash_farm_new/x0y0/images/', '0tif.tif', 0, 0, 'C:/Users/ahalboabidallah/Desktop/ash_farm_new/x0y0/images/', reverse=0,step1=0.05)
    #return segim
#102
def SegmentChains(PathtoSegmentationFiles,PreviousLayer,CurrentLayer,NextLayer,List_Series=[]):
    #(PathtoSegmentationFiles,PreviousLayer,CurrentLayer,NextLayer,List_Series)=(outpath+'SegImages_no_Centres/',previouslayer,currentlayer,nextlayer,List_Series1)
    #List_Series=[[[layer,segment],...,[layer,segment]],...,[[layer,segment],...,[layer,segment]]]#
    #readtolist('Parinting_C'+childlayer[0:-4]+'_P'+parentlayer[0:-4]+'.csv')
    CN=readtolist(PathtoSegmentationFiles+CurrentLayer+'/','Parinting_C'+NextLayer+'_P'+CurrentLayer+'.csv')#current layer as parent and Next layer as child
    LC,LN,LCLN=CNtoLC_LN(CN,CurrentLayer,NextLayer)
    for segment in CN:
        segment
        Parents=list(filter(lambda a:a[-1][0]==CurrentLayer and int(float(a[-1][1]))==int(segment[0]),List_Series))
        #LCLN[:,1]
        Children=(np.array(list(filter(lambda a:int(float(a[1]))==int(segment[0]),LCLN)))[:,2:]).tolist()
        #find parent series locations in List_Series
        new_series=[[CurrentLayer,str(segment[0])]]
        for parent in Parents:
            new_series.extend(parent)#put them in a new list #combine them
            List_Series.remove(parent)#delete them from List_Series
        new_series.sort()
        new_series=list(new_series for new_series,_ in itertools.groupby(new_series))
        new_series.extend(Children)#Add children to the new series
        List_Series.append(new_series)#Add the new series to the old List_Series
    return List_Series#List_
#103
def SegmentChains1(PathtoSegmentationFiles,CurrentLayer,nextlayer,previouslayer,List_Series=[]):#this is an old version# slow
    #List_Series=[[[layer,segment],...,[layer,segment]],...,[[layer,segment],...,[layer,segment]]]#
    #readtolist()
    CurrentLayerSegs=FindFilesInFolder(PathtoSegmentationFiles+'/'+CurrentLayer+'/','seg*')
    #ChildLayerSegs=FindFilesInFolder(PathtoSegmentationFiles+'/'+CurrentLayer,'C*')
    #print(2)#for each segment in current layer#
    for seg in CurrentLayerSegs:
        #print('seg',end='')
        #children=readtolist(PathtoSegmentationFiles+'/'+CurrentLayer[0:6]+str(int(CurrentLayer[6:])+1),'C'+seg1[3:-4]+'.csv',1)
        if os.path.isfile(PathtoSegmentationFiles+'/'+CurrentLayer+'/P'+seg[3:-4]+'.csv'):
            #print(3)
            parents=readtolist(PathtoSegmentationFiles+'/'+CurrentLayer,'/P'+seg[3:-4]+'.csv',1)
        else:
            #print(4)
            parents=[]
        No_Parents=len(parents)
        #Children=find children by using ChildLayer
        if No_Parents == 0:
            #print(5)
            #make new series 
            List_Series.append([[CurrentLayer,seg[0:-4]]])
        elif No_Parents == 1:
            '1'
            #print(6)
            #add current segment to the parent series
            #print('parents[0]=',parents[0])
            #
            index=find_in_list_of_lists(List_Series,[previouslayer,'seg'+str(int(parents[0][0]))])
            #print('List_Series=',List_Series)
            #print('index=',index)
            #print('[CurrentLayer,seg]=',[CurrentLayer,seg])
            if index != []:
                appendinlist(List_Series,index[0],[CurrentLayer,seg[0:-4]])
                #print(7)
            else:
                List_Series.append([[CurrentLayer,seg[0:-4]]])
                #print(8)
                #print('can not find>>',[CurrentLayer[0:6]+str(int(CurrentLayer[6:])-1),seg+str(int(parents[0][0]))])
        else:
            #print(9)#combine Parents chains
            for p in parents:
                #print('p=',p)
                index=find_in_list_of_lists(List_Series,[previouslayer,'seg'+str(int(p[0]))])
            if index!=[]:
                #print(10)
                index=sorted(index,key=lambda l:l)
                while len(index)>1:
                     combinedinlist(List_Series,index[0],index[-1])
                     del index[-1]
                #print(11)
                appendinlist(List_Series,index[0],[CurrentLayer,seg[0:-4]])     
            else:
                #print(12)
                List_Series.append([[CurrentLayer,seg[0:-4]]])
                print('can not find>>',[CurrentLayer,previouslayer,seg])
            #print(13)#add current segment to the parent series
            #appendinlist(List_Series,index[0],[CurrentLayer,seg[0:-4]])
    return List_Series#List_Series=SegmentChains(PathtoSegmentationFiles,CurrentLayer),old_List_Series)
#104
def SegmentsIntersectionPoints(List_Series):
    IntersectionPoints=[]
    for series in List_Series:
        series=sorted(series,key=lambda l:l[1])
        series=sorted(series,key=lambda l:l[0])
        series=list(filter(lambda b: series[series.index(b)][0:2] ==series[series.index(b)+1][0:2] or series[series.index(b)][0:2] ==series[series.index(b)-1][0:2],series[0:-1]))
        IntersectionPoints.append(series)
    return IntersectionPoints
#105
def segmentIntersectionPoints(List_Series):
    IntersectionPoints=SegmentsIntersectionPoints(List_Series)
    for series in IntersectionPoints:
        series=sorted(series,key=lambda l:l[-1])#sort by layer
        series=sorted(series,key=lambda l:l[1])
        series=sorted(series,key=lambda l:l[0])
        segmenti=series[0]
        BranchPoints=[]
        Branchs=[]
        for point in series[1:]:
            if ((point[0]-segmenti[-1][0])**2+(point[1]-segmenti[-1][1])**2)<2:
                #extend segment
                segmenti.append(point)
            else: 
                #calculate the mean of old segment
                mean=[np.mean(np.array(segmenti)[:,0]),np.mean(np.array(segmenti)[:,1]),np.mean(np.array(segmenti)[:,-1])]#[x,y,z]
                #append the means list
                BranchPoints.append(mean)
                #make anew segment with the current segment
                segmenti=series[point]
        Branchs.append(BranchPoints)
    return Branchs
#106
def segment(arr1,reverse=1):#=1 laer points dark and backgrond wight else=0
    #inetialize groups out go1
    arr1=binary((arr1),1,1)
    if reverse==1:
        arr1=arr1*-1+1
    rs,cs=np.shape(arr1)
    go1=np.zeros((rs+2,cs+2))
    arr2=np.zeros((rs+2,cs+2))
    arr2[1:-1,1:-1]=np.array(arr1)
    rs,cs=np.shape(arr2)
    g=0
    similar=[]
    for rr in range(rs-1):#exclude 1st row
        for cc in range(cs-2):#exclude 1st and last columns
            r,c=rr+1,cc+1
            if arr2[r,c]==1:
                neighbours=[arr2[r-1,c-1],arr2[r-1,c],arr2[r-1,c+1],arr2[r,c-1]]
                neighbours2=[go1[r-1,c-1],go1[r-1,c],go1[r-1,c+1],go1[r,c-1]]
                if sum(neighbours)==0:
                    g=g+1
                    go1[r,c]=g
                elif sum(neighbours)==1:
                    go1[r,c]=max(neighbours2)
                elif neighbours[1]==0:
                    go1[r,c]=max(neighbours2)
                    while max(neighbours2)==go1[r,c]:
                        neighbours2.remove(max(neighbours2))
                    similar.append([go1[r,c],max(neighbours2)])
                else:
                    go1[r,c]=max(neighbours2)
        else:
            pass
    similar2=[]
    for t in similar:
        if min(t)!=0 and t[1]!=t[0]:
            similar2.append(t)
    similar=similar2
    #print('similar=',similar)
    i=-1
    #return go1,similar
    for t2 in similar:
        i+=1
        go1=np.where(go1==similar[i][1],similar[i][0],go1)
        arraysimilar=np.array(similar)
        similar=np.array(np.where(arraysimilar==similar[i][1],similar[i][0],arraysimilar)).tolist()
    return go1[1:-1,1:-1]
#107
def series_clasification(path1,image1,centres):
    #(path1,image1,centres)=(outpath+'Series/','15322.tif',CentresH)
    ds = gdal.Open(path1+image1)
    (xmino,res1,tilt1,ymino,tilt2,res2)=ds.GetGeoTransform()
    xmaxo = xmino + (ds.RasterXSize * res1)
    ymaxo = ymino + (ds.RasterYSize * res2)
    centres1=sorted(centres,key=lambda l:(l[0]-xmino)**2+(l[1]-ymino)**2)[0]
    centres2=sorted(centres,key=lambda l:(l[0]-xmaxo)**2+(l[1]-ymino)**2)[0]
    centres3=sorted(centres,key=lambda l:(l[0]-xmino)**2+(l[1]-ymaxo)**2)[0]
    centres4=sorted(centres,key=lambda l:(l[0]-xmaxo)**2+(l[1]-ymaxo)**2)[0]
    return sorted([centres1,centres2,centres3,centres4],key=lambda l:(l[0]-xmino)**2+(l[0]-ymino)**2)[0]
#108
def Series_pixels_Coordinates(PathtoSegmentationFiles,List_Series):
    List_Series_Coordinates=[]
    i=0
    for chain in List_Series:
        i+=1
        winsound.Beep(90, 5)
        #chain
        currentChain=[]
        for segment in chain:
            segment
            #seg=read segment file
            seg=readtolist(PathtoSegmentationFiles+segment[0]+'/',segment[1]+'.csv',5)
            seg = [x + [segment[0],segment[1]] for x in seg]
            currentChain.extend(seg)
        List_Series_Coordinates.append(currentChain)
        writetofile(PathtoSegmentationFiles,'output'+str(int(i))+'.txt',currentChain,7)
    return List_Series_Coordinates# List_Series_Coordinates=Series_Coordinates(PathtoSegmentationFiles,List_Series)
#109
def Series_Coordinates(PathtoSegmentationFiles,List_Series,layerPath,step=0.07):
    List_Series_Coordinates=[]
    for chain in List_Series:
        winsound.Beep(900, 250)
        chain
        currentChain=[]
        for segment in chain:
            #segment
            #seg=read segment file
            #seg=readtolist(PathtoSegmentationFiles+segment[0]+'/',segment[1]+'.csv',5)
            seg=PointsInSegment(layerPath,segment[0]+'.csv',PathtoSegmentationFiles+segment[0]+'/',segment[1]+'.csv',step)
            #seg = [x + [segment[0],segment[1]] for x in seg]
            currentChain.append(seg)
        List_Series_Coordinates.append(currentChain)
    return List_Series_Coordinates# List_Series_Coordinates=Series_Coordinates(PathtoSegmentationFiles,List_Series)
#110
def Series_Coordinates2(PathtoSegmentationFiles,List_Series2,layerPath,step=0.07):
    List_Series_Coordinates=[]
    i=0
    for chain in List_Series2:
        i+=1
        winsound.Beep(900, 250)
        #chain
        currentChain=[]
        for segment in chain:
            #read the file
            #seg=read segment file
            try:
                seg=readtolist(PathtoSegmentationFiles+segment[0]+'/','point'+segment[1]+'.csv',5)
            except:
                print('error reading segment> ',segment )
            #seg = [x + [segment[0],segment[1]] for x in seg]
            currentChain.extend(seg)
        #List_Series_Coordinates.append(currentChain)
        #save chain
        #f = open(PathtoSegmentationFiles+'/output'+str(int(i))+'.txt', 'w')
        #pickle.dump(seg, f)
        writetofile(PathtoSegmentationFiles+'/output',str(int(i))+'.txt',seg,5)
    return List_Series_Coordinates# Series_Coordinates2(PathtoSegmentationFiles,List_Series2,layerPath,0.07)
#111
#Function to import a shape file into a list
''' '''
#inputs the file storage path and the file name
def shp2list(path,shp):
    shp=path+shp
    sf = shapefile.Reader(shp)
    shapes = sf.shapes()
    records=sf.records()
    list1=[]
    for i in range(len(shapes)):
        list4add=[]
        list4add=[shapes[i].points[0][0],shapes[i].points[0][1],shapes[i].z[0]]
        list4add.extend(records[i])
        list1.append(list4add)
    return(list1)   
#112
def skeletonizeIT(path1,image1,outpath,outimage):
    try:
        os.stat(path1)
    except:
        os.makedirs(path1) 
    #read it
    image=(img2np(path1,image1))
    image = Image.fromarray(image)
    image.convert('L')
    image =binary(np.array(image.filter(ImageFilter.BLUR)),1,1)
    #blurred_image = original_image.filter(ImageFilter.BLUR)
    ds = gdal.Open(path1+image1)
    (xmino,res1,tilt1,ymino,tilt2,res2)=ds.GetGeoTransform()
    #blur it
    skeleton = skeletonize(image)*255
    #save the skeleton
    array2raster(outpath,outimage,xmino,ymino,res1,res2,skeleton,type1=gdal.GDT_Byte)
#113
def skeletonizeIT(path1,image1,outpath,outimage):
    try:
        os.stat(path1)
    except:
        os.makedirs(path1) 
    #read it
    image=(img2np(path1,image1))
    image = Image.fromarray(image)
    image.convert('L')
    image =binary(np.array(image.filter(ImageFilter.BLUR)),1,1)
    #blurred_image = original_image.filter(ImageFilter.BLUR)
    ds = gdal.Open(path1+image1)
    (xmino,res1,tilt1,ymino,tilt2,res2)=ds.GetGeoTransform()
    #blur it
    skeleton = skeletonize(image)*255
    #save the skeleton
    array2raster(outpath,outimage,xmino,ymino,res1,res2,skeleton,type1=gdal.GDT_Byte)
#114
def signal_noise_ratio(in_raster,band=1):
    img, xx, yy, gt=read_raster(in_raster,band=band)#read it
    img=np.where(img <-100000, 0, img)[1000:-100]
    img=np.transpose(img)[500:-500]
    #img[np.isnan(img)] = 0
    SNR= scipy.stats.signaltonoise(img, axis=None)
    return SNR
#115
def slice2image3(slicepath,slicefile,imagepath, imagefile,Xmin,Xmax,Ymin,Ymax, step=0.12):
    # creatarray to be image
    try:
        os.stat(imagepath)
    except:
        os.makedirs(imagepath) 
    Dx=int((math.floor(Xmax-Xmin)/step))-1
    Dy=int((math.floor(Ymax-Ymin)/step))-1
    M=Dx+2
    N=Dy+2
    image_mat=np.zeros((N,M))
    myfile=readtopandas(slicepath,slicefile,alradyhasheader=0)
    myfile=myfile.drop(myfile.columns[[0]],1)
    myfile=myfile.rename(columns={1: 'X', 2: 'Y', 3: 'Z',4:'ZDEM',5:'Dz'})
    #myfile= open(slicepath+slicefile,'r')    
    #for row in myfile:
    #    row=row.split(",")
    #    try:
    #        x,y,zlidar,ztin=float(row[0]),float(row[1]),float(row[2]),float(row[3])
    #    except:
    #        pass
    #    i=int(math.floor(((x-Xmin)/step)))  
    #    j=int(math.floor(((y-Ymin)/step)))
        #pix=pixels[i,j]
        #pixels[i,j]=(pix[1]-1,pix[1]-1,pix[1]-1)
    #myfile['i']=int(math.floor(((myfile['X']-Xmin)/step)))
    myfile['j']=myfile['X']-Xmin
    myfile['j']=myfile['j'].apply(lambda x: math.floor(x/step1))
    myfile['i']=myfile['Y']-Ymin
    myfile['i']=myfile['i'].apply(lambda y: math.floor(y/step1))
    myfile=myfile.transpose()
    for row in myfile:
        #print(myfile[row].tolist())
        try:
            image_mat[int(myfile[row].tolist()[-2]),int(myfile[row].tolist()[-1])]=image_mat[int(myfile[row].tolist()[-2]),int(myfile[row].tolist()[-1])]+1
        except:
            print('Error in'+str(row))    
    #image_mat2=np.ones((M,N))*255-image_mat#/(np.max(image_mat)/255)#
    #img = PIL.Image.frombytes('L', (3, 3), image_mat2)
    #img2 = Image.fromarray(image_mat,'L')
    #vigra.impex.writeImage(abs(image_mat.astype(numpy.int16)), imagepath+imagefile, dtype = '', compression = '', mode = 'w')  
    array2raster(imagepath,imagefile+'.tif',Xmin,Ymin,step,step,np.transpose(image_mat))

#116
# a function to convert slices to images
def slice2image(slicepath,slicefile,imagepath, imagefile, step=0.12):
    myfile= open(slicepath+slicefile,'r')
    x_points,y_points,z_points,pts=[],[],[],[]
    for row in myfile:
        row=row.split(",")
        x,y,zlidar,ztin=float(row[0]),float(row[1]),float(row[2]),float(row[3])
        x_points.append(x)
        y_points.append(y)
        z_points.append(zlidar)
        pts.append([x,y,zlidar,ztin])#z2_points.append(dz)
        #rgb.append([r,g,b])
        
    #image dimensions#
    Dx=int(math.floor((np.max(x_points)-np.min(x_points))/step))-1
    Dy=int(math.floor((np.max(y_points)-np.min(y_points))/step))-1
    M=Dx+2
    N=Dy+2
    'M=',M,'N=',N
    image_mat=np.zeros((M,N))
    #img = Image.new('RGB', (M,N), "white") # create a new black image
    #img = Image.new('L', (M,N))
    #pixels = img.load() # create the pixel map
    #for i in range(img.size[0]):    # for every pixel:
    #    for j in range(img.size[1]):
    #        pixels[i,j] = (i, j, 100) # set the colour accordingly
    for pt in pts:
        x=float(pt[0])
        y=float(pt[1])
        i=int(math.floor(((x-np.min(x_points))/step)))
        j=int(math.floor(((y-np.min(y_points))/step)))
        #pix=pixels[i,j]
        #pixels[i,j]=(pix[1]-1,pix[1]-1,pix[1]-1)
        image_mat[i,j]=image_mat[i,j]+1
    image_mat2=np.ones((M,N))*255-image_mat#/(np.max(image_mat)/255)#
    #img = PIL.Image.frombytes('L', (3, 3), image_mat2)
    #img2 = Image.fromarray(image_mat,'L')
    img = Image.fromarray(np.uint8(image_mat2),'L')
    img.save(imagepath+imagefile)
    #img2.save('image2.jpg')
    #img3.save('image3.jpg')
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show(block=False)
#117
#arrange point list
def SortBranchPoints(Branchs,centreX,centreY):
    Branchs2=[]
    for BranchPoints in Branchs:
         BranchPoints=sorted(BranchPoints,key=lambda l:((centreX-l[0])**2+(centreY-l[1])**2))
         Branchs2.append(BranchPoints)
    return Branchs2
#118
def sortAndUniq(input):
  output = []
  for x in input:
    if x not in output:
      output.append(x)
  output.sort()
  return output
#119
def split_df_layers_pandas(xyzlidarDzID,layer_path,layer_depth):
    try:
        os.stat(layer_path)
    except:
        os.makedirs(layer_path)
    ##read file to pandas
    xyzlidarDzID['Dz']=xyzlidarDzID['Z']-xyzlidarDzID['ZDEM']+1
    zmin=max([min(xyzlidarDzID['Dz']),0])
    zmax=max(xyzlidarDzID['Dz'])
    layers=np.array(range(math.ceil(zmax/layer_depth)-math.floor(zmin/layer_depth)))+math.floor(zmin/layer_depth)
    for layer in layers:
        try:
            df=xyzlidarDzID[(xyzlidarDzID['Dz']>=layer*layer_depth) & (xyzlidarDzID['Dz']<(layer+1)*layer_depth)]
            df.to_csv(layer_path+str(layer)+'.csv',mode = 'a',header=None)
        except:
            print('error in layer:',layer)
#120
def split_layers_pandas(inpath,infile,layer_path,layer_depth):
    try:
        os.stat(layer_path)
    except:
        os.makedirs(layer_path)
    ##read file to pandas
    xyzlidarDzID=readtopandas(inpath+'/',infile,alradyhasheader=1)
    xyzlidarDzID = xyzlidarDzID.drop(xyzlidarDzID.columns[[0]],1)
    xyzlidarDzID['Dz']=xyzlidarDzID['Z']-xyzlidarDzID['ZDEM']+1
    zmin=max([min(xyzlidarDzID['Dz']),0])
    zmax=max(xyzlidarDzID['Dz'])
    layers=np.array(range(math.ceil(zmax/layer_depth)-math.floor(zmin/layer_depth)))+math.floor(zmin/layer_depth)
    for layer in layers:
        try:
            df=xyzlidarDzID[(xyzlidarDzID['Dz']>=layer*layer_depth) & (xyzlidarDzID['Dz']<(layer+1)*layer_depth)]
            df.to_csv(layer_path+str(layer)+'.csv',mode = 'a',header=None)
        except:
            print('error in layer:',layer)
#121
def split_to_layers(xyzlidarDzID,OutputPath,layerdepth=0.1):
    try:
        os.stat(OutputPath)
    except:
        os.makedirs(OutputPath)
    xyzlidarDzID=sorted(xyzlidarDzID,key=lambda l:l[3])
    zi=0
    for i in range(int(xyzlidarDzID[-1][3]/layerdepth)):
        zi+=layerdepth
        list2=list(filter(lambda x: x[3]<zi and x[3]>zi-layerdepth, xyzlidarDzID))
        Addtofile(OutputPath,'layer'+str(int(zi/layerdepth))+'.csv',list2,5)
#122
def split_to_layers_Pandas(inpath,infile,OutputPath,layerdepth=0.1):
    try:
        os.stat(OutputPath)
    except:
        os.makedirs(OutputPath)
    #read file to pandas
    xyzlidarDzID=readtopandas(inpath+'/',infile,alradyhasheader=1)
    xyzlidarDzID = xyzlidarDzID.drop(0,1)
    xyzlidarDzID=xyzlidarDzID.rename(columns={1: 'X', 2: 'Y', 3: 'Z',4:'Zdem'})
    xyzlidarDzID.Dz=xyzlidarDzID.Z-xyzlidarDzID.Zdem
    
    zi=0
    for i in range(int(xyzlidarDzID[-1][3]/layerdepth)):
        zi+=layerdepth
        list2=list(filter(lambda x: x[3]<zi and x[3]>zi-layerdepth, xyzlidarDzID))
        Addtofile(OutputPath,'layer'+str(int(zi/layerdepth))+'.csv',list2,5)
#123
#split small files each 50
def splitgrid(inputPath, file1, Dm):
    #read file to layer
    list1=readtolist(inputPath,file1,3)
    #find minx, maxx, miny and maxy
    ymin=min(np.array(list1)[:,1])
    xmin=min(np.array(list1)[:,0])
    ymax=max(np.array(list1)[:,1])
    xmax=max(np.array(list1)[:,0])    
    #for between minx and maxx
    for ii in range(int(xmax/Dm)-int(xmin/Dm)+2):
        ii
        xrangemin=(int(xmin/Dm)+ii)*Dm-Dm
        xrangemax=xrangemin+Dm
        for j in range(int(ymax/Dm)-int(ymin/Dm)+2):
            j
            yrangemin=(int(xmin/Dm)+j)*Dm-Dm
            yrangemax=yrangemin+Dm
            #        filter list for this Dm by Dm
            
            list2=list(filter(lambda x: x[0]>=xrangemin and x[0]<xrangemax and x[1]>=yrangemin and x[1]<yrangemax, list1))
            # creat folder and save new files
            if list2!=[]:
                print(file1,xrangemin,xrangemax,'y',yrangemin,yrangemax)
                try:# make a new folder to save the new segments--------------
                    os.stat(inputPath+'x'+str(xrangemin)+'y'+str(yrangemin)+'/')#                         --
                except:#                                                                --
                    os.mkdir(inputPath+'x'+str(xrangemin)+'y'+str(yrangemin)+'/')#
                Addtofile(inputPath+'x'+str(xrangemin)+'y'+str(yrangemin)+'/',file1,list2,3)
            else:
                pass
#124
# a function to split the whole scan into plots and extract profiles for each small file 
def splitgridprofile(inputPath, file1, Dm,OutputPath,spacer=2,width=0.02):#profiles(pointlistXYZ,OutputPath,xmin,xmax,spacer=2,width=0.02)
    #read file to layer
    list1=readtolist(inputPath,file1,3)
    #find minx, maxx, miny and maxy
    ymin=min(np.array(list1)[:,1])
    xmin=min(np.array(list1)[:,0])
    ymax=max(np.array(list1)[:,1])
    xmax=max(np.array(list1)[:,0])
    profiles(list1,OutputPath,xmin,xmax,spacer,width)    
    #for between minx and maxx
    for ii in range(int(xmax/Dm)-int(xmin/Dm)+2):
        ii
        xrangemin=(int(xmin/Dm)+ii)*Dm-Dm
        xrangemax=xrangemin+Dm
        for j in range(int(ymax/Dm)-int(ymin/Dm)+2):
            j
            yrangemin=(int(ymin/Dm)+j)*Dm-Dm
            yrangemax=yrangemin+Dm
            #        filter list for this Dm by Dm
            
            list2=list(filter(lambda x: x[0]>=xrangemin and x[0]<xrangemax and x[1]>=yrangemin and x[1]<yrangemax, list1))
            # creat folder and save new files
            if list2!=[]:
                print(file1,xrangemin,xrangemax,'y',yrangemin,yrangemax)
                try:# make a new folder to save the new segments--------------
                    os.stat(inputPath+'x'+str(xrangemin)+'y'+str(yrangemin)+'/')#--
                except:#                                                         --
                    os.mkdir(inputPath+'x'+str(xrangemin)+'y'+str(yrangemin)+'/')#
                Addtofile(inputPath+'x'+str(xrangemin)+'y'+str(yrangemin)+'/',file1,list2,3)
            else:
                pass
#125
def subset2np(path1,file1,xmin,ymin,xmax,ymax):
    f=path1+'/'+file1
    ds = gdal.Open(f.replace("//", "/"))
    (xmino,res1,tilt1,ymino,tilt2,res2)=ds.GetGeoTransform()
    mincol=max(0,math.floor((xmin-xmino)/res1))
    minrow=max(0,math.ceil((ymin-ymino)/res2))
    maxcol=min(ds.RasterXSize,math.floor((xmax-xmino)/res1))
    maxrow=min(ds.RasterYSize,math.ceil((ymax-ymino)/res2))
    return np.array(ds.GetRasterBand(1).ReadAsArray(xoff=mincol, yoff=minrow, win_xsize=maxcol-mincol, win_ysize=maxrow-minrow))
#a=subset2np(path1,file1,xmin,ymin,xmax,ymax)
#126
def thrushold_gdal(infilepath,infile,outfilepath,outfile, thrushold_value):
    try:
        os.stat(outfilepath)
    except:
        os.makedirs(outfilepath)
    #subprocess.call([sys.executable, 'C:\\Program Files\\Anaconda3\\Scripts\\gdal_calc.py', '-A', infilepath+infile, '--outfile='+outfilepath+outfile, '--calc="255*(A>'+str(thrushold_value)+')"'])
    infilepath,infile,outfilepath,outfile=infilepath.replace("/", "\\"),infile.replace("/", "\\"),outfilepath.replace("/", "\\"),outfile.replace("/", "\\")
    print('"C:\\gdal_calc.py",', '"-A",', infilepath+infile, ',"--outfile='+outfilepath+outfile, '","--calc=255*(A>'+str(thrushold_value)+')"')
    subprocess.call([sys.executable, "C:\\gdal_calc.py", "-A", infilepath+infile, "--outfile="+outfilepath+outfile, "--calc=255*(A>"+str(thrushold_value)+")"])
#127
def trim_images_trunks(imagespath, centrespath,outimagepath,step3):
    #find images 
    layers=FindFilesInFolder(imagespath,'layer*.png')#step
    #read centres
    centres=readtolist(centrespath,'all_layers_all_centres.csv',6)
    #for layer(known by image)
    import numpy as np
    for layer in layers[:-1]:
        print(layer)
        s=int(layer[5:-4])
        #print(s)
        #read image
        arr1 =vigra.impex.readImage(imagespath+'/'+layer, dtype='', index=0, order='')[:,:,0]
        #arr1=arr1*0
        #filter centres
        cent=list(filter(lambda a:a[2]==float(s),centres))#
        #for centre
        #Xmin,Ymin=math.ceil(cent[0][0]/50)-1,math.ceil(cent[0][1]/50)-1
        for centre in cent:
             #find it in image
             #Ximage,Yimage=centre
             Xi,Yi,S,A,Ximage,Yimage=centre
             #Ximage,Yimage=Ximage-int(1/step),len(arr1)-Yimage
             
             #Yimage1,Ximage1=int((Xi-Xmin)/step+1),int((Yi-Ymin)/step+1)
             #print('Ximage,Yimage',Ximage,Yimage,'Yimage1,Ximage1',Yimage1,Ximage1)
             kkk=np.pi
             Rimage=int(A/(kkk*step3**2)+2)
             #make zero pixels
             #print(arr1[max([Ximage-Rimage,0]):min([Ximage+Rimage,int(50/step)]),max([Yimage-Rimage,0]):min([Yimage+Rimage,int(50/step)])])
             arr1[max([Ximage-Rimage+1,0]):min([Ximage+Rimage+1,int(50/step3)]),max([Yimage-Rimage+1,0]):min([Yimage+Rimage+1,int(50/step3)])]=0*arr1[max([Ximage-Rimage+1,0]):min([Ximage+Rimage+1,int(50/step3)]),max([Yimage-Rimage+1,0]):min([Yimage+Rimage+1,int(50/step3)])]+255
        # rewrite the out image
        #arr1 =np.flipud(arr1)
        vigra.impex.writeImage(arr1.astype(numpy.int16), outimagepath+'/t'+layer, dtype = '', compression = '', mode = 'w')  
#trim_images_trunks(imagespath, centrespath,outimagepath,step)
#128
def trunkbiomass(centrespath,biomassimage,layerpath,step=0.12):
    #read image
    try:
        arr1 =vigra.impex.readImage(biomassimage, dtype='', index=0, order='')
        arr1 = arr1[:,:,0]
    except:
        arr1 = np.zeros((50/step,50/step))
    #read centres and add them to an array
    AA1=FindFilesInFolder(centrespath,'H*')
    #find the areas of ellipses in the 15 and 30 layers  
    for centrei in AA1:
        #centrei=AA1[0]
        thisCentre=sortAndUniq(readtolist(centrespath,centrei,8))
        if thisCentre[0][0]<0:
            Xmin=math.ceil(thisCentre[0][0]/50)*50.0
        else:
            Xmin=math.ceil(thisCentre[0][0]/50)*50.0-50
        if thisCentre[0][1]<0:
            Ymin=math.ceil(thisCentre[0][1]/50)*50.0
        else:
            Ymin=math.ceil(thisCentre[0][1]/50)*50.0-50
        #Ymin=math.ceil(thisCentre[0][0]/50)*50.0-50
        #Xmin=math.ceil(thisCentre[0][1]/50)*50.0-50
        hight=int(thisCentre[0][6])
        XiYis=[]
        for i in range(int(hight)):
            s=int(hight-i)
            s
            next2layers=(sorted(thisCentre,key=lambda l:abs(l[5]-s)))[0:2]
            try:
                D1,D2,A1,A2,Y1,X1,Y2,X2=hight-next2layers[0][5],hight-next2layers[1][5],next2layers[0][7],next2layers[1][7],next2layers[0][0],next2layers[0][1],next2layers[1][0],next2layers[1][1]
            except:
                D1,D2,A1,A2,Y1,X1,Y2,X2=hight-next2layers[0][5],hight-next2layers[0][5]+1,next2layers[0][7],next2layers[0][7],next2layers[0][0],next2layers[0][1],next2layers[0][0],next2layers[0][1]
            if max([A1,A2])>9:
                print(centrei)
            D3=hight-s
            #(Di2-D12)/A1-A2
            try:
                A3=A2*(hight-s)**2/D2**2
            except:
                A3=A2*(hight-s)**2/(D2**2+.001)
            try:
                Xi=X1+(X2-X1)*(D1-D3)/(D1-D2)
                #Yi=(Y2-Y1)*(s-D1)/(D2-D1)+Y1
                Yi=Y1+(Y2-Y1)*(D1-D3)/(D1-D2)
            except:
                print('duplicated centre in layer',s,centrei,  'Xi,Yi,D1,D2=',Xi,Yi,D1,D2)
            #Addtofile(centrespath,'all_layers_all_centres.csv',[Xi,Yi,s])
            Rimage=int(math.ceil((A3/np.pi)**.5/step))
            Ximage,Yimage=abs((Xi-Xmin)/step),abs((Yi-Ymin)/step)
            XiYis.append([Yi,Xi,s,A3,Ximage,Yimage])
            vol=A3*0.1*1000000#the hight change 10cm for each i
            stamp = np.zeros((Rimage*2+1,Rimage*2+1))
            for ii in range(Rimage*2+1):
                for jj in range(Rimage*2+1):
                    if ((ii-Rimage)**2+(jj-Rimage)**2)<=Rimage**2:
                        stamp[ii,jj]=1
                    else:
                        pass
            pixelvol=abs(vol/np.sum(stamp))
            for ii in range(Rimage*2):
                for jj in range(Rimage*2):
                    try:
                        arr1[int(math.ceil(Ximage+ii-Rimage)),int(math.ceil(Yimage+jj-Rimage))]=arr1[int(math.ceil(Ximage+ii-Rimage)),int(math.ceil(Yimage+jj-Rimage))]+abs(pixelvol*stamp[ii,jj])
                    except:
                        print(centrei,'out of bounds',Yi,Xi,s,A3,Ximage,Yimage )
        Addtofile(centrespath,'all_layers_all_centres.csv',XiYis,6)
        #print(int(math.ceil(Ximage+ii-Rimage)),int(math.ceil(Yimage+jj-Rimage)))
    vigra.impex.writeImage(arr1.astype(numpy.float32), biomassimage, dtype = '', compression = '', mode = 'w')  
    pickle.dump(arr1, open(layerpath+'/TrunkBiomassArray.txt', 'w'))
    #return arr1
    #arr1 = trunkbiomass(CentresPath,CentresPath+'Tbiomas1.tif',step)
#129
def wightedSeries(chain,centre):
    wight=[]
    for point in chain:
        Xc,Xp,Yc,Yp=centre[0],point[0],centre[1],point[1]
        print(Xp,Yp)
        distance2=(Xc-Xp)**2+(Yc-Yp)**2
        wight.append([Xp,Yp,1/distance2])
    wight=np.array(wight)
    wight[:,-1]=wight[:,-1]/sum(wight[:,-1])
    return wight
#130
#Function to export a list to a file
#inputs the list, the file storage path, the file name and the number of columns in the list

def writetofile2(path,file1,list1,NoOfColumns=3):
    try:
        os.stat(path)
    except:
        os.makedirs(path)
    F=path+file1
    text1=''
    for i in range(NoOfColumns):
        text1=text1+',x'+str(i+1)
    text1=text1[1:]
    read=open(F,'w')
    if NoOfColumns!=1:
        for line in list1:
            try:
                exec(text1+'= [float(value) for value in line]')
            except:
                exec(text1+'= [value for value in line]')
            for i in range(NoOfColumns):
                exec("read.write(str(x"+str(i+1)+'))')
                read.write(',')
            read.write('\n')
    else:
        for line in list1:
            exec("read.write(str(line))")
            read.write('\n')
    read=0
winsound.Beep(300, 300)
#131
def writetofile(path,file1,list1,NoOfColumns=3):
    try:
        os.stat(path)
    except:
        os.makedirs(path)
    #convert to pandas
    df=DataFrame(list1)
    #write to csv
    df.to_csv(path+file1,index=False,header=False)
winsound.Beep(300, 300)








