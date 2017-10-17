# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 18:37:06 2016

@author: ahalboabidallah
"""

# -*- coding: utf-8 -*-
"""
final2
Created on Tue Oct 27 11:39:09 2015

@author: ahalboabidallah
"""
# this file is to combine all codes done before 
#1. making small files from the original file and TAKING PROFILES
#2. FITERING GROUND POINTS
#3. MAKING A SURFACE
#4. MAKING SLICES
#5. MAKING POINT INTENSITY RASTER FROM SLICES
#6. FINDING THE CENTRES IN LAYER 1.5m
#7. FINDING THE CLOSEST SEGMENT/SEGMENTS FOR EACH CENTRE
#8. FINDING POINTS RELATED TO EACH CENTER
#9. FITTING CYLINDER
#10. MOVING TO THE 1.3 AND 1.4 LAYERS
#11. COMPARING RESULTS 
#12. MAKING LOWER LAYERS
#13. MOVING UP WITHT TRUNK 
#14. WORKING WITH BRANCHS IN EACH LAYER


#arcpy.CheckOutExtension("3D")
import subprocess# 
import winsound
import time 
winsound.Beep(440, 300) # frequency, duration
winsound.Beep(700, 250)
winsound.Beep(900, 250)
import os
import gdal 
from gdal import *
import ogr
#from subprocess import call
#askopenfilename(title=_("Select Attachment"))
# show me the orional points file
#file1='F:/beliver/all/all.txt'
#file1='F:/drakes park/final.txt'
#path='F:/beliver/all/'
#path='F:/drakes park/'
#file1='F:/Data/res ahmed/all/all.txt'
#path='F:/Data/res ahmed/all/'
file1='C:/Users/ahalboabidallah/Desktop/ash_farm_new/all.txt'
path='C:/Users/ahalboabidallah/Desktop/ash_farm_new/'


teststep=1
#file1=askopenfilename(title=("Select the Origional Txt File"), filetypes=[('txt', '*.txt')])
# show me the working directory
#path=askdirectory(title='Chosing the working directory')

#path=path.replace("\", '/')
path=path+'/'
path=path.replace('//', '/')
#making the folders tree
ProfilesPath=path+'profiles/'
ProfilesResultsPath=ProfilesPath+'profiles/results/'
ProfilesResultsShpPath=ProfilesResultsPath+'shapefiles/all/'
TINpath=path+'profiles/profiles/results/shapefiles/ground/'#where the tin is
SmallFilePath=path#where  the 500 000 are

step=0.05#maight changed
step1=step


original=open(str(file1))
#you have to install GunWin or coreutils-5.3.0 from http://sourceforge.net/projects/gnuwin32/files/coreutils/5.3.0/
##os.makedirs('F:/Data/res ahmed/all/profiles/profiles/results/shapefiles/ground/')
inkscape_dir=file1[0:len(file1) - 1 - file1[::-1].index('/')]
assert os.path.isdir(inkscape_dir)
os.chdir(inkscape_dir)
oldfiles=FindFilesInFolder(file1[0:len(file1) - 1 - file1[::-1].index('/')],'*')
#subprocess.call('split -a 3 -l 1000000 '+file1[len(file1)  - file1[::-1].index('/'):], shell=True)
smallfiles=FindFilesInFolder(file1[0:len(file1) - 1 - file1[::-1].index('/')],'*.txt')
subprocessfiles=FindFilesInFolder(path,'x*')
gg=0
smallfiles=[]
for file2 in subprocessfiles:
    gg+=1
    file4=str(gg)+'.txt'
    print ('the million:',gg)
    os.rename(file2, file4)

    #############################################################################

smallfiles=FindFilesInFolder(SmallFilePath,'*txt')
smallfiles.remove('all.txt')

for i in smallfiles:
    i
    #splitgrid(SmallFilePath, i, 50)
    splitgridprofile(SmallFilePath, i, 25,ProfilesPath,10,width=0.02)# helps to save time by doing both on one list reading
    
teststep=2
interval=5

profiles=FindFilesInFolder(ProfilesPath,'*csv')
for profile in profiles:
    profile_filter2(ProfilesPath,profile,interval,ProfilesResultsPath)
    
gprofiles=FindFilesInFolder(ProfilesResultsPath,'*csv')
allgroundlist=[]
#2
for k in gprofiles:
    list1=readtolist(ProfilesResultsPath,k,3)
    allgroundlist.extend(list1)
writetofile(ProfilesResultsPath,'AllGround.csv',allgroundlist)
#we have to creat DEM 
#csv2shp(ProfilesResultsPath,'AllGround.csv',TINpath,'shp'+'AllGround.csv')    
#
#creat a list of all ground points to build a TIN from it
#creat TIN surface
#arcpy.CheckOutExtension("3D")
#env.workspace = TINpath#arcpy.CreateTin_3d("NewTIN", "Coordinate Systems/Projected Coordinate Systems/State Plane/NAD 1983 (Feet)/NAD 1983 StatePlane California II FIPS 0402 (Feet).prj", "points.shp Shape.Z masspoints", "constrained_delaunay")
#arcpy.CreateTin_3d("tin", "", "AllGround.shp Shape.Z masspoints", "constrained_delaunay")

teststep=3
######################################################################################################
#creat a dem with 2m resolution 




# your directory with all your csv files in it
dir_with_csvs = "C:/Users/ahalboabidallah/Desktop/ash_farm_new/profiles/profiles/results/"

# make it the active directory
os.chdir(dir_with_csvs)

# get the filenames
csvfiles = ['AllGround.csv']

# loop through each CSV file=
# for each CSV file, make an associated VRT file to be used with gdal_grid command
# and then run the gdal_grid util in a subprocess instance
for fn in csvfiles:
    vrt_fn = fn.replace(".csv", ".vrt")
    lyr_name = fn.replace('.csv', '')
    out_tif = fn.replace('.csv', '.tiff')
    temp_tif='temp.tiff'#just to find corners
    with open(vrt_fn, 'w') as fn_vrt:
        fn_vrt.write('<OGRVRTDataSource>\n')
        fn_vrt.write('\t<OGRVRTLayer name="%s">\n' % lyr_name)
        fn_vrt.write('\t\t<SrcDataSource>%s</SrcDataSource>\n' % fn)
        fn_vrt.write('\t\t<GeometryType>wkbPoint</GeometryType>\n')
        fn_vrt.write('\t\t<GeometryField encoding="PointFromColumns" x="X" y="Y" z="Z"/>\n')
        fn_vrt.write('\t</OGRVRTLayer>\n')
        fn_vrt.write('</OGRVRTDataSource>\n')

    gdal_cmd = 'gdal_grid -a invdist:power=2.0:smoothing=5.0 -zfield "Z" -outsize 1 1 -of GTiff -ot Float64 -l %s %s %s' % (lyr_name, vrt_fn, temp_tif)
    subprocess.call(gdal_cmd, shell=True)
    datafile = gdal.Open(temp_tif)
    #Xsize = datafile.RasterXSize
    #Ysize = datafile.RasterYSize
    geoinformation = datafile.GetGeoTransform()
    LowerLeftX = geoinformation[0]
    LowerLeftY = geoinformation[3]
    ox,oy=int(geoinformation[1]/2)+1,int(geoinformation[5]/2)+1
    xmin,xmax=int(LowerLeftX/2)*2-6,int((LowerLeftX+ox*2)/2)*2+6
    ymin,ymax=int((LowerLeftY+oy*2)/2)*2+6,int(LowerLeftY/2)*2-6
    gdal_cmd = 'gdal_grid -a invdist:power=2.0:smoothing=5.0 -zfield "Z" -txe '+ str(xmin)+' '+str(xmax) +' -tye '+str(ymin)+' '+str(ymax)+' -outsize '+str((xmax-xmin)/2)+' '+str((ymin-ymax)/2)+' -of GTiff -ot Float64 -l %s %s %s' % (lyr_name, vrt_fn, out_tif)
    subprocess.call(gdal_cmd, shell=True)
    datafile=None
    print(gdal_cmd)
#to convert the dem to a csv file
import rasterio
import numpy as np
import csv
filename = 'C:/Users/ahalboabidallah/Desktop/ash_farm_new/profiles/profiles/results/AllGround.tiff'
with rasterio.open(filename) as src:
    #read image
    image= src.read()
    # transform image
    bands,rows,cols = np.shape(image)
    image1 = image.reshape (rows*cols,bands)
    print(np.shape(image1))
    # bounding box of image
    l,b,r,t = src.bounds
    #resolution of image
    res = src.res
    res = src.res
    # meshgrid of X and Y
    x = np.arange(l,r, res[0])
    y = np.arange(t,b, -res[0])
    X,Y = np.meshgrid(x,y)
    print (np.shape(X))
    # flatten X and Y
    newX = np.array(X.flatten(1))
    newY = np.array(Y.flatten(1))
    print (np.shape(newX))
    # join XY and Z information
    export = np.column_stack((newX, newY, image1))
    fname='C:/Users/ahalboabidallah/Desktop/ash_farm_new/profiles/profiles/results/AnXYZ.csv'
    with open(fname, 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(export)
        fp.close() # close file
print('OK')
'''
cmd = ["gdal_translate", "-of", "XYZ", "AllGround.tiff", "toXYZ.xyz"]
subprocess.call(cmd)
'''


###################################################################################################






#find plots
plotfolders=FindFilesInFolder(SmallFilePath,'x*y*')
#split each plot to layers 
try:# make a new folder to save the new segments--------------
    os.stat(path+'profiles/profiles/results/shapefiles/ground/')#                         --
except:#                                                                --
    os.makedirs(path+'profiles/profiles/results/shapefiles/ground/')
#import arcpy
#arcpy.CheckExtension('3D') 
#arcpy.CheckOutExtension("3D")

for plot1 in plotfolders:
    plot1
    #'''
    #------------------------------
    #-                            -
    #------------------------------
    #'''
    smallfiles=FindFilesInFolder(SmallFilePath+plot1+'/','*txt')
    #to work with ARCGIS
    #try:
    #     copyanything(path+'/profiles/profiles/results/shapefiles/ground/tin', SmallFilePath+plot1+'/tin/')
    #except:
    #     print ('tin error')
    #env.workspace = SmallFilePath+plot1+'/'
    #for i in smallfiles:
    #    try:
    #        csv2shp(SmallFilePath+plot1+'/',i,SmallFilePath+plot1+'/','shp'+i[0:-4])
    #        arcpy.InterpolateShape_3d("tin", 'shp'+i[0:-4]+".shp", "Ground"+i[0:-4])
    #    except:
    #        print('Empty file'+i)
    #GroundFiles=FindFilesInFolder(path+plot1+'/','G*shp')
    #for shpg in GroundFiles:
    #    xyzlidarDzID=listztin(SmallFilePath+plot1+'/',shpg)# each point[x,y,zlidar,Dz,ID]
    #    split_to_layers(xyzlidarDzID,SmallFilePath+plot1+'/noground/',0.05)
    #    strftime("%Y-%m-%d %H:%M:%S", gmtime())
    #    print('file ',shpg)
    
teststep=4
#this is not needed with gdal was used with arcgis at the begining
'''for plot1 in plotfolders:
    GroundFiles=FindFilesInFolder(path+plot1+'/','G*shp')
    for shpg in GroundFiles:
        xyzlidarDzID=listztin(SmallFilePath+plot1+'/',shpg)# each point[x,y,zlidar,Dz,ID]
        split_to_layers(xyzlidarDzID,SmallFilePath+plot1+'/noground/',0.1)
        strftime("%Y-%m-%d %H:%M:%S", gmtime())
        print('file ',shpg)'''
# for each plot
#plotfolders=['x0y0']
teststep=5
for plot1 in plotfolders:
    plot1
    # set paths
    slicesnImagePath=path+plot1+'/noground/'+str(int((step*100)))+"/"
    ThrusholdImagePath=slicesnImagePath+'Thrushold/'
    nogroundpath=path+plot1+'/noground/'
    temppath=slicesnImagePath+'temp/'
    layerPath=path+plot1+"/noground/"
    outpath=slicesnImagePath+'out/'
    temppath=slicesnImagePath+'temp/'
    CentresPath=outpath+'Centres/'
    out=slicesnImagePath+'out/'
    path15=out+'/15/'
    path30=out+'/30/'
    PathtoSegmentationFiles=outpath+'segmentfiles2/'
    IntersectPath=slicesnImagePath+'Intersect/'# 
    try:
        os.stat(out)
    except:
        os.makedirs(out)
    IntersectionPath=outpath+'intersected/'
    IntersectPath=slicesnImagePath+'Intersect/'#
    layers=FindFilesInFolder(path+plot1+'/noground/','layer*')# F:\beliver\all\x0y0\noground
    dr=path
    
    # find layers
    from PIL import Image
    teststep=6
    for layer in layers:
        layer
        # convert layers to images
        yin=(plot1.find('y'))
        Xmin,Ymin=int(plot1[1:yin]),int(plot1[yin+1::])
        slice2image2(nogroundpath,layer,slicesnImagePath, layer[0:-4]+'.png',Xmin,Xmin+50,Ymin,Ymin+50, step)
        strftime("%Y-%m-%d %H:%M:%S", gmtime())
        print(plot1,layer,'converted to image and thrushed')
    ImagesThrushold(slicesnImagePath,ThrusholdImagePath,253)
    Images=FindFilesInFolder(ThrusholdImagePath,'*png')
    #####1. for layer 1.5m=2slice15.csv    #########################
    #intersectioning layers
    #            intersect layer 13 with layer 15
    teststep=7
    IntersectLayers(slicesnImagePath,'layer13.png', 'layer15.png',IntersectionPath,'13X15.png',1)
    segmentIm(IntersectionPath,'13X15.png', IntersectionPath, 'seg13X15.tif')
    #find the closest segments list for each centre
    ###############################################################################
    MakeSegmentsFiles(IntersectionPath,'seg13X15.tif',path15,Xmin,Ymin,step)
    B=find4pts(path15)
    SegmentsFilesPath=path15
    #finding rough centres without Cdetector
    c=[]
    C=[]
    for i in range(len(B)/4):
        try:
            b=filter(lambda x: x[4]==i+1, B)
            if  len(sortAndUniq(b))!=1:
                b=np.array(b)
                C.append([np.mean(b[:,0]),np.mean(b[:,1]),1])
                c.append([np.mean(b[:,3]),np.mean(b[:,2]),1])
        except:
            print(i)
    
    c=cleanC(c,.5/step)
    C=cleanC(C,.5)
    '''#            find the centres
    [C,c]=Cdetector(IntersectionPath,'13X15.png',Xmin,Ymin,step)'''
    
    writetofile(temppath+'Centres/','scanner13X15.csv',C,3)
    writetofile(temppath+'Centres/','Image13X15.csv',c,3)
    ############################################################################
    i=0
    image='13X15.png'
    layer=image
    im = plt.imread(IntersectionPath+image)
    #creat folder
    try:
        os.stat(temppath+image[0:-4]+'/')
        print('get an old'+image)
    except:
        os.mkdir(temppath+"/"+ image[0:-4]+'/')
    try:
        os.stat(path+plot1+'/classification/')
    except:
        os.mkdir(path+plot1+'/classification/')
    #            check the centres 
    teststep=8
    f=readtolist(temppath+'Centres/','Image13X15.csv',3)
    CorrectedCentersXY=[]
    i=0
    for point in f:
        i+=1
        i
        exec('fig'+str(i) +'= plt.figure()')
        test=50
        xtest=test
        ytest=test
        [xc,yc,r]=point
        xmil=xc-xtest
        if xmil<0:
            xtest=test+xmil
            xmil=0
        xmal=xc+test
        if xmal>np.shape(im)[1]:
            xmal=np.shape(im)[1]
        ymil=yc-test
        if ymil<0:
            ytest=test+ymil
            ymil=0
        ymal=yc+test
        if ymal>np.shape(im)[0]:
            ymal=np.shape(im)[0]
        ffff=im[ymil:ymal,xmil:xmal]
        implot = plt.imshow(250*ffff)
        plt.scatter(x=xtest, y=ytest, c='y', s=30)
        plt.annotate((xc, yc),xy = (xtest,ytest), xytext = (-5, -5))
        #######
        plt.savefig(temppath+'temp.png')
        subim = Image.open(temppath+'temp.png')
        subim.convert('RGB')
        subim.save(path+'test.gif')
        #Chatterbox Chatter Box
        choices=['yes','no','not sure']
        quiry=buttonbox(choices=choices,image=path+'test.gif')#plt.show(block=False)
        if quiry=='yes':
            cv2.imwrite(path+'/classification/Yes'+layer[0:-4]+str(i)+'.png',ffff*255)
            xcorrected=step*yc+Xmin
            ycorrected=step*xc+Ymin
            CorrectedCentersXY.append([xcorrected,ycorrected])
            #write file
        elif quiry=='not sure':
            cv2.imwrite(path+'/classification/NS'+layer[0:-4]+str(i)+'.png',ffff*255)
            #add corrections
            #write file
            xcorrected=step*yc+Xmin
            ycorrected=step*xc+Ymin
            CorrectedCentersXY.append([xcorrected,ycorrected])
        else:
            cv2.imwrite(path+'/classification/No'+layer[0:-4]+str(i)+'.png',ffff*255)
        plt.close()
        try:
            exec ('del fig'+str(i))
        except:
            pass
        try:
            exec ('plt.close(fig'+str(i)+')')
        except:
            pass
    teststep=9
    writetofile(temppath+'Centres/','Corrected13X15.csv',CorrectedCentersXY,2)
    #segment '13X15.png'
    #read centres to list 'centres'
    centres=readtolist(temppath+'Centres/','Corrected13X15.csv',2)#?
    #creat folder to save results 
    #with a file for each centre includes subjected centres for it
    #closestSegments    
    gg=0
    teststep=10
    for centre in centres:
        gg+=1
        gg
        #read X,Y coordinates of this centre from 'Corrected13X15.csv'
        [X,Y]=centre
        # find the closest segments
        SID,D=closestSegment(X,Y,B)
        # find angle
        try:
            theta=Is_angle(X,Y,SegmentsFilesPath,SID[0],100)
        except: 
            theta=Is_angle(X,Y,SegmentsFilesPath,SID[0]-1,100)
            print ('error segment name', SID[0])
        list1=[SID[0]]
        p=0
        while theta!=True:
            # find the next closest centre
            p+=1
            print('p=',p)
            Si=SID[p]
            Di=D[p]
            #if the distance less than 1.5m add its point to the four points and make new 4pts
            if Di<1:
                list1.append(Si)
                theta=Is_angle(X,Y,SegmentsFilesPath,Si,100)
                try:
                    theta=Is_angle(X,Y,SegmentsFilesPath,Si,100)
                except: 
                    theta=Is_angle(X,Y,SegmentsFilesPath,Si-1,100)
                    print ('error segment name', SID[0])
            # calculate the angle again
            #list1.append(segment)
            else:
                theta=True# to break the loop
                print('week tree scan C='+str(X)+','+str(Y))
            #angle=101
        #save list1 to file
        writetofile(temppath+'Centres/13X15/',str(gg)+'.csv',list1,1)
        #save the centre's id and its coordinate
        Addtofile(temppath+'Centres/13X15/','CentreID.csv',[[gg,X,Y]],3)
    #            fit ellipsed for each centre and save them
    # for each centre in centre id read the required segments
    teststep=11
    centres=readtolist(temppath+'Centres/13X15/','CentreID.csv',3)#
    list1=readtolist(layerPath,'layer15.csv',5)
    for centre in centres:
        ID,X,Y=centre
        ID=int(ID)
        ID
        segments=readtolist(temppath+'Centres/13X15/',str(ID)+'.csv',1)#
        layerPath=path+plot1+"/noground/"
        FitPtslist=[]
        for segment1 in segments:
            #find points in this segment and add them to the fitting list
            try:
                FitPtslisti=PointsInSegment2(list1,path15,'seg'+str(int(segment1[0]))+'.csv',step)
            except:
                FitPtslisti=PointsInSegment2(list1,path15,'seg'+str(int(segment1[0]-1))+'.csv',step)  
            FitPtslist.extend(FitPtslisti)
        try:
            t=np.array(FitPtslist)[:,0:2]
            mnx=min(t[:,0])
            mny=min(t[:,1])
            t[:,0]=t[:,0]-min(t[:,0])
            t[:,1]=t[:,1]-min(t[:,1])
            a=fit_ellipse(t)
            [X0,Y0,aa,bb,theta]=gen_elli2(a)
            X0,Y0=X0+mnx,Y0+mny
        except:
            'no enough points in '+str(ID)
        #            save the new centres
        Addtofile(outpath+'Centres/15/','ellipses2.csv',[[X0,Y0,aa,bb,theta]],5)
    teststep=12
    #####2. for layer 3m                  ##########################
    #            intersect layer 28 with layer 30
    #            find the centres
    #            check the centres 
    #            fit ellipsed for each centre and save them
    #            save the new centres
    #intersectioning layers
    #            intersect layer 13 with layer 15
    IntersectLayers(slicesnImagePath,'layer28.png', 'layer30.png',IntersectionPath,'28X30.png',1)
    #            find the centres
    #[C,c]=Cdetector(IntersectionPath,'28X30.png',Xmin,Ymin,step)#
    segmentIm(IntersectionPath,'28X30.png', IntersectionPath, 'seg28X30.tif')
    #find the closest segments list for each centre
    ###############################################################################
    MakeSegmentsFiles(IntersectionPath,'seg28X30.tif',path30,Xmin,Ymin,step)
    B=find4pts(path30)#                    #
    SegmentsFilesPath=path30
    c=[]
    C=[]
    for i in range(len(B)/4):
        try:
            b=filter(lambda x: x[4]==i+1, B)
            if  len(sortAndUniq(b))!=1:
                b=np.array(b)
                C.append([np.mean(b[:,0]),np.mean(b[:,1]),1])
                c.append([np.mean(b[:,3]),np.mean(b[:,2]),1])
        except:
            print(i)
    
    c=cleanC(c,.5/step)
    C=cleanC(C,.5)
    writetofile(temppath+'Centres/','scanner28X30.csv',C,3)
    writetofile(temppath+'Centres/','Image28X30.csv',c,3)
    ############################################################################
    i=0
    dr=path
    image='28X30.png'
    layer=image
    im = plt.imread(IntersectionPath+image)
    #creat folder
    try:
            os.stat(temppath+image[0:-4]+'/')
            print('get an old'+image)
    except:
            os.mkdir(temppath+"/"+ image[0:-4]+'/')
    try:
            os.stat(path+'/classification/')
    except:
            os.mkdir(path+'/classification/')
    #            check the centres 
    f=readtolist(temppath+'Centres/','Image28X30.csv',3)
    CorrectedCentersXY=[]
    i=0
    teststep=13
    for point in f:
        i+=1
        i
        exec('fig'+str(i) +'= plt.figure()')
        test=50
        xtest=test
        ytest=test
        [xc,yc,r]=point
        xmil=xc-xtest
        if xmil<0:
            xtest=test+xmil
            xmil=0
        xmal=xc+test
        if xmal>np.shape(im)[1]:
            xmal=np.shape(im)[1]
        ymil=yc-test
        if ymil<0:
            ytest=test+ymil
            ymil=0
        ymal=yc+test
        if ymal>np.shape(im)[0]:
            ymal=np.shape(im)[0]
        ffff=im[ymil:ymal,xmil:xmal]
        implot = plt.imshow(250*ffff)
        plt.scatter(x=xtest, y=ytest, c='y', s=30)
        plt.annotate((xc, yc),xy = (xtest,ytest), xytext = (-5, -5))
        #######
        plt.savefig(temppath+'temp.png')
        subim = Image.open(temppath+'temp.png')
        subim.convert('RGB')
        subim.save(dr+'\\test.gif')
        #Chatterbox Chatter Box
        choices=['yes','no','not sure']
        quiry=buttonbox(choices=choices,image=dr+'\\test.gif')#plt.show(block=False)
        if quiry=='yes':
            cv2.imwrite(path+'/classification/Yes'+layer[0:-4]+str(i)+'.png',ffff*255)
            xcorrected=step*yc+Xmin
            ycorrected=step*xc+Ymin
            CorrectedCentersXY.append([xcorrected,ycorrected])
            #write file
        elif quiry=='not sure':
            cv2.imwrite(dr+'/classification/NS'+layer[0:-4]+str(i)+'.png',ffff*255)
            #add corrections
            #write file
            xcorrected=step*yc+Xmin
            ycorrected=step*xc+Ymin
            CorrectedCentersXY.append([xcorrected,ycorrected])
        else:
            cv2.imwrite(dr+'/classification/No'+layer[0:-4]+str(i)+'.png',ffff*255)
        plt.close()
        try:
            exec ('del fig'+str(i))
        except:
            pass
        try:
            exec ('plt.close(fig'+str(i)+')')
        except:
            pass
    teststep=14
    writetofile(temppath+'Centres/','Corrected28X30.csv',CorrectedCentersXY,2)
    #segment '13X15.png'
    
    try:
        os.stat(SegmentsFilesPath)
    except:
        os.mkdir(SegmentsFilesPath) 
    #read centres to list 'centres'
    centres=readtolist(temppath+'Centres/','Corrected28X30.csv',2)#?
    #creat folder to save results 
    #with a file for each centre includes subjected centres for it
    #closestSegments   
    gg=0
    teststep=15
    for centre in centres:
        gg+=1
        gg
        #read X,Y coordinates of this centre from 'Corrected13X15.csv'
        
        [X,Y]=centre
        # find the closest segments
        SID,D=closestSegment(X,Y,B)
        # find angle
        #theta=Is_angle(X,Y,SegmentsFilesPath,SID[0],100)
        try:
            theta=Is_angle(X,Y,SegmentsFilesPath,SID[0],100)
        except: 
            theta=Is_angle(X,Y,SegmentsFilesPath,SID[0]-1,100)
            print ('error segment name', SID[0])
        list1=[SID[0]]
        p=0
        while theta!=True:
            # find the next closest centre
            p+=1
            print('p=',p)
            try:
                Si=SID[p]
                Di=D[p]
                #if the distance less than 1.5m add its point to the four points and make new 4pts
                if Di<1:
                    list1.append(Si)
                    try:
                        theta=Is_angle(X,Y,SegmentsFilesPath,Si,100)
                    except: 
                        theta=Is_angle(X,Y,SegmentsFilesPath,Si-1,100)
                        print ('error segment name', SID[0])
                # calculate the angle again
                #list1.append(segment)
                else:
                    theta=True# to break the loop
                    print('week tree scan C='+str(X)+','+str(Y))
            except :
                theta=True# to break the loop
                print('week tree scan C='+str(X)+','+str(Y))
               #angle=101
        #save list1 to file
        writetofile(temppath+'Centres/28X30/',str(gg)+'.csv',list1,1)
        #save the centre's id and its coordinate
        Addtofile(temppath+'Centres/28X30/','CentreID.csv',[[gg,X,Y]],3)
    #            fit ellipsed for each centre and save them
    # for each centre in centre id read the required segments
    centres=readtolist(temppath+'Centres/28X30/','CentreID.csv',3)#
    list1=readtolist(layerPath,'layer30.csv',5)
    teststep=16
    for centre in centres:
        ID,X,Y=centre
        ID=int(ID)
        ID
        segments=readtolist(temppath+'Centres/28X30/',str(ID)+'.csv',1)#
        layerPath=path+plot1+"/noground/"
        FitPtslist=[]
        for segment1 in segments:
            #find points in this segment and add them to the fitting list
            try:
                FitPtslisti=PointsInSegment2(list1,path30,'seg'+str(int(segment1[0]))+'.csv',step)
            except:
                FitPtslisti=PointsInSegment2(list1,path30,'seg'+str(int(segment1[0])-1)+'.csv',step)                
            FitPtslist.extend(FitPtslisti)
        try:
            t=np.array(FitPtslist)[:,0:2]
            mnx=min(t[:,0])
            mny=min(t[:,1])
            t[:,0]=t[:,0]-min(t[:,0])
            t[:,1]=t[:,1]-min(t[:,1])
            a=fit_ellipse(t)
            [X0,Y0,aa,bb,theta]=gen_elli2(a)
            X0,Y0=X0+mnx,Y0+mny
        except:
            'no enough points in '+str(ID)
        #            save the new centres
        Addtofile(outpath+'Centres/30/','ellipses.csv',[[X0,Y0,aa,bb,theta]],5)
    ##### 3. unifi centres of 1 and 2 and make a file of 2 points for each centre region of 1m #############
    teststep=17
    list1=readtolist(outpath+'Centres/30/','ellipses.csv',5)
    list2=readtolist(outpath+'Centres/15/','ellipses2.csv',5)
    list1=sorted(list1,key=lambda l:l[1])
    list1=sorted(list1,key=lambda l:l[0])
    list2=sorted(list2,key=lambda l:l[1])
    list2=sorted(list2,key=lambda l:l[0])
    # for each layer in layers 3m and 1.5 m for each centre if there is a centre very close -> the same tree delete one of them
    s=0
    list3=[]
    for point in list1[0:-1]:
        s+=1
        if (list1[s][0]-list1[s-1][0])**2+(list1[s][1]-list1[s-1][1])**2>1:
            point.append(30)
            list3.append(point)
        else:
            pass
    for p1 in list2:
        D=[]#
        for p2 in list1:
            D.append(((p1[1]-p2[1])**2+(p1[0]-p2[0])**2)**.5)
        if min(D)>1:
            p1.append(15)
            list3.append(p1)
            #keep
        else:
            #delete
            pass
    # make a file for each centre in layer 3m 
    teststep=18
    i=0
    writetofile(outpath+'Centres/','ellipse30.csv',list3,6)
    for ellipse in list3:
        i+=1
        writetofile(outpath+'Centres/',str(i)+'.csv',[ellipse],6)
    # for each file if one point only -> duplicate it
    #pass
    ##### 4. for layers 3.5, 4, 4.5, 5, 5.5 ... end #################
    #segment intersections of all layers-------------------------------------- 
    IntersectPath=slicesnImagePath+'Intersect/'#                             -
    try:#                                                                    -
        os.stat(IntersectPath)#                                              -
    except:#                                                                 -
        os.mkdir(IntersectPath)#                                             -
    mm=FindFilesInFolder(ThrusholdImagePath,'*png')#                         -
    mm2=[]#                                                                  -
    for file1 in mm:#                                                        -
        mm2.append([file1,int(file1[5:-4])])#                                -
    mm=0#                                                                    -
    mm2=sorted(mm2,key=lambda l:l[1])#                                       -
    mm2=filter(lambda a: a[1] >30, mm2)#                                     -
    i=0#                                                                     -
    teststep=19
    for i in range(len(mm2)/2):#                                             -
        IntersectLayers(ThrusholdImagePath,mm2[2*i][0], mm2[2*i-1][0],IntersectPath,'Intersect'+str(int(2*i-1))+'x'+str(int(2*i))+'.png',1)
        segmentIm(IntersectPath,'Intersect'+str(int(2*i-1))+'x'+str(int(2*i))+'.png', IntersectPath, 'Intersect'+str(int(2*i-1))+'x'+str(int(2*i))+'.tif')
        MakeSegmentsFiles(IntersectPath,'Intersect'+str(int(2*i-1))+'x'+str(int(2*i))+'.tif',IntersectPath+str(int(2*i-1))+'x'+str(int(2*i))+'/',Xmin,Ymin,step)
    mm=0#---------------------------------------------------------------------
      # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ##           find the tree hight for each centre in 1 and 2 above##          
    mm=FindFilesInFolder(layerPath,'*csv')                                    #
    mm2=[]                                                                    #
    for file1 in mm:                                                          #
        mm2.append([file1,int(file1[5:-4])])                                  #
    #                                                                         #
    mm2=sorted(mm2,key=lambda l:l[1])                                         #
    #i=mm2[-1][1]     
    i=len(mm2)                                                                #
    test=0                                                                    #
    list3=readtolist(outpath+'Centres/','ellipse30.csv',6)                    #
    list3=np.array(list3)                                                     #
    list3= np.append([[0 for _ in range(0,len(list3))]], list3.T,0).T         #
    teststep=20
    while test==0:                                                               
        i-=1                                                                  #
        print('i= ',i)                                                           
        list1=readtolist(layerPath,mm2[i][0],5)                               #
        p=0                                                                      
        for centre in list3:                                                  #
            p+=1                                                                 
            if list3[p-1,0]==0:                                               #
                if TreeExist(list1,centre,.5)>10:                                
                    list3[p-1,0]=i                                            #
        if min(list3[:,0])>0 or i==35:                                        #
            test=1                                                            #
    teststep=21
    writetofile(outpath+'Centres/','ellipsesWithHights.csv',list3,7)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       for each layer fit ellipses for the centres from the previous layer
    mm=FindFilesInFolder(outpath+'Centres/','*csv')
    centres=[]
    i=0
    for i in mm:
        try:
            centres.append([i,int(i[0:-4])])
        except:
            pass
    teststep=22
    segmentedlayers=FindFilesInFolder(IntersectPath,'*tif')
    segmentedlayers=filter(lambda a: float(a[a.find('x')+1:-4]) >30, segmentedlayers)
    mm2=[]
    teststep=23
    for file1 in segmentedlayers:    #
        mm2.append([file1,int(float(file1[file1.find('x')+1:-4]))])               # 
    mm2=sorted(mm2,key=lambda l:l[1]) 
    teststep=24
    #
    for layer in mm2:
        layer[1]
        B=find4pts(IntersectPath+'/'+str(layer[0][layer[0].find('ect')+3:-4])+'/')
        laserpoints=readtolist(layerPath,'layer'+str(int(layer[1]))+'.csv',5)
        for centre in centres:
            gg=centre[-1]
            #read X,Y coordinates of this centre from 'Corrected13X15.csv'
            [X,Y,a,b,t,l]=readtolist(outpath+'Centres/',centre[0],6)[-1]
            # find the closest segments
            try:
                SID,D=closestSegment(X,Y,B)
            except:
                pass    
            # find angle
            #theta=Is_angle(X,Y,IntersectPath+str(layer[0][layer[0].find('ect')+3:-4])+'/',SID[0],100)
            try:
                theta=Is_angle(X,Y,IntersectPath+str(layer[0][layer[0].find('ect')+3:-4])+'/',SID[0],100)
            except: 
                try:
                    theta=Is_angle(X,Y,IntersectPath+str(layer[0][layer[0].find('ect')+3:-4])+'/',SID[0]-1,100)
                    print ('error segment name', SID[0])
                except:
                    print ('empty error')
            list1=[SID[0]]
            p=0
            while theta!=True:
                # find the next closest centre
                p+=1
                print('p=',p)
                try:
                    Si=SID[p]
                    Di=D[p]
                    #if the distance less than 1.5m add its point to the four points and make new 4pts
                    if Di<1:
                        list1.append(Si)
                        try:
                            theta=Is_angle(X,Y,IntersectPath+str(layer[0][layer[0].find('ect')+3:-4])+'/',Si,100)
                        except:
                            theta=Is_angle(X,Y,IntersectPath+str(layer[0][layer[0].find('ect')+3:-4])+'/',Si-1,100)
                            print ('error segment name', SID[0])
                     # calculate the angle again
                     #list1.append(segment)
                    else:
                        theta=True# to break the loop
                        print('week tree scan C='+str(X)+','+str(Y))
                except:
                    theta=True
                    print ('only one segment in layer', layer)
            #angle=101
            #save list1 to file
            writetofile(temppath+'Centres/'+str(layer[0][layer[0].find('ect')+3:-4])+'/',str(gg)+'.csv',list1,1)
            #save the centre's id and its coordinate
            Addtofile(temppath+'Centres/'+str(layer[0][layer[0].find('ect')+3:-4])+'/','CentreID.csv',[[gg,X,Y]],3)
            #       fit ellipsed for each centre and save them if the ellipse is a good fit
            ID=int(gg)
            segments=readtolist(temppath+'Centres/'+str(layer[0][layer[0].find('ect')+3:-4])+'/',str(ID)+'.csv',1)#
            layerPath=path+plot1+"/noground/"
            FitPtslist=[]
            for segment1 in segments:
                #find points in this segment and add them to the fitting list
                try:
                     FitPtslisti=PointsInSegment2(laserpoints,IntersectPath+str(layer[0][layer[0].find('ect')+3:-4])+'/','seg'+str(int(segment1[0]))+'.csv',step)
                except:
                     try:
                         FitPtslisti=PointsInSegment2(laserpoints,IntersectPath+str(layer[0][layer[0].find('ect')+3:-4])+'/','seg'+str(int(segment1[0]-1))+'.csv',step)    
                     except:
                         FitPtslisti=[]
                         print('***')
                FitPtslist.extend(FitPtslisti)
            try:
                t=np.array(FitPtslist)[:,0:2]
                mnx=min(t[:,0])
                mny=min(t[:,1])
                t[:,0]=t[:,0]-min(t[:,0])
                t[:,1]=t[:,1]-min(t[:,1])
                a=fit_ellipse(t)
                [X0,Y0,aa,bb,theta]=gen_elli2(a)
                X0,Y0=X0+mnx,Y0+mny
                # save the new centres
                if ((X0-X)**2+(Y0-Y)**2)**.5<(layer[1]-l)*.1*.25:
                    Addtofile(outpath+'Centres/',str(int(gg))+'.csv',[[ X0,Y0,aa,bb,theta,int(layer[1]) ]],6)
            except:
                'no enough points in '+str(ID)
    
    ##### 5. for each segmented layer delete the pixels that far with distance of 20cm+Rtree+step from the constructed lines of detected centres to keep branches segments only
    #read layers and centres # # # # # # # # # # # # # # # # # # # # # # # # # # #
    teststep=25
    ThrusholdImagePath=slicesnImagePath+'Thrushold/'                              #
    mm=FindFilesInFolder(ThrusholdImagePath,'*png')#                                     #
    layers1=[]#                                                                   #
    for file1 in mm:#  
        try:                                                                      #
            layers1.append([file1,int(file1[5:-4])])#                             #
        except:
            pass
    mm=0#                                                                         #
    layers1=sorted(layers1,key=lambda l:l[1])#                                    #
    layers1=filter(lambda a: a[1]>10, layers1)#                                   #
    mm=FindFilesInFolder(outpath+'Centres/','*csv')#                                     #
    centres=[]#                                                                   #
    i=0#                                                                          #
    teststep=26                                                                   #
    for i in mm:#                                                                 #
        try:#                                                                     #
            centres.append([i,int(i[0:-4])])#                                     #
        except:#                                                                  #
            pass#                                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    try:# make a new folder to save the new images----------------------------
            os.stat(slicesnImagePath+'Thrushold2/')#                         -
    except:#                                                                 -
            os.mkdir(slicesnImagePath+'Thrushold2/')#                        -
            #-----------------------------------------------------------------
    centresH=readtolist(outpath+'Centres/','ellipsesWithHights.csv',7)#read centres with trees hights will need it for R at each layer
    # work on each layer and then on each centre
    teststep=27
    for layer in layers1:
        #read the image
        im1=vigra.impex.readImage(slicesnImagePath+'Thrushold/'+layer[0], dtype='', index=0, order='')[:,:,0]
        ccc=[]
        for centre in centres:
            pointlistXYZ=readtolist(outpath+'Centres/',centre[0],6)
            centresHi=filter(lambda a: a[1]==pointlistXYZ[0][0], centresH)[0]
            centre,centresHi[1:],centresHi[0]
            R=(centresHi[0]-layer[1])*max(centresHi[3:5])/(centresHi[0]-centresHi[-1])
            try:
                 D1,Xc,Yc=Intersect_polyline_layer(pointlistXYZ,layer[1])
            except:
                print('error in centre ',centre,'layer',layer)
            ccc.append([centre[1],D1,Xc,Yc])
            if D1==True and R>0:
                # Xc,Yc to image coordinate Xi,Yi
                #X=step*(i-1)+Xmin>>>i=(X-Xmin)/step
                #Y=step*(j-1)+Ymin>>>j=(Y-Ymin)/step
                J,I=(Xc-Xmin)/step,(Yc-Ymin)/step
                R=R+step+.2#distance of 20cm+Rtree+step
                for i in range(int(R/step)):
                    for j in range(5+int(R/step)):
                        s=5+int((i-((int(R/step))/2))+I)
                        t=5+int((j-((int(R/step))/2))+J)
                        if (((s-I)**2+(t-J)**2)**.5) < int(1/step)/2:
                            try:
                                im1[int(s),int(t)]=255
                            except:
                                print('Edge centre')
        #centre[1],int(s),int(t)
        #save the new image
        vigra.impex.writeImage(im1, slicesnImagePath+'Thrushold2/'+layer[0], dtype = '', compression = '', mode = 'w')
        #outpath+'segmentfiles2'
    teststep=28
    try:# make a new folder to save the new segments--------------
            os.stat(outpath+'segmentfiles2/')#                         --
    except:#                                                                --
            os.mkdir(outpath+'segmentfiles2/')#                        --
            #-----------------------------------------------------
    layers2=layers1
    #
    for layer in layers2:#
        #images=FindFilesInFolder(slicesnImagePath+'Thrushold2/','*png')#
        segmentIm(slicesnImagePath+'Thrushold2/' ,layer[0][0:-4]+'.png', outpath+'segmentfiles2/', layer[0][0:-4]+'.tif')
        #MakeSegmentsFiles(SegmentedImagePath,SegmentedImage,SegmentsFilesPath,Xmin,Ymin,step)
        MakeSegmentsFiles(outpath+'segmentfiles2/',layer[0][0:-4]+'.tif',outpath+'segmentfiles2/'+layer[0][0:-4]+'/',Xmin,Ymin,step)
    ##### 6. parnts and childs
    PathtoSegmentationFiles=outpath+'segmentfiles2/'
    try:
        os.stat(PathtoSegmentationFiles)#                                   --
    except:#                                                                --
        os.mkdir(PathtoSegmentationFiles)#
    '''
    for layer in range(len(layers1[:-1])):
        #layer
        currentlayer=layers1[layer]#
        nextlayer=layers1[layer+1]
        currentlayer,nextlayer
        #try:
        list1=parinting(PathtoSegmentationFiles, currentlayer[0][0:-4]+'.tif',nextlayer[0][0:-4]+'.tif')
        #except:#                                                                --
        #    print('missing layer',layer[0][0:-4]+'.tif','layer'+str(layer[1]+1)+'.tif')
        #    pass
        #parinting(path, parentlayer,childlayer,parentstep=1,childstep=1)
    '''
    teststep=29
    layeraAll=FindFilesInFolder(PathtoSegmentationFiles,'*')#
    files=FindFilesInFolder(PathtoSegmentationFiles,'*.*')
    layeraAll=[item for item in layeraAll if item not in set(files)]
    layerAll=[]
    
    for file1 in layeraAll:#                                                       -
        layerAll.append([file1,int(file1[5:])])#
    layerAll=sorted(layerAll,key=lambda l:l[1])
    List_Series1=[]
    '''#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    List_Series1=SegmentChains(PathtoSegmentationFiles,layerAll[0][0],layerAll[1][0],[],List_Series1)
    for ii in range(len(layerAll[:-2])):
        CurrentLayer=layerAll[ii+1][0]
        nextLayer=layerAll[ii+2][0]
        previouslayer=layerAll[ii][0]
        previouslayer,CurrentLayer,nextLayer
        #lllllll
        #SegmentChains(PathtoSegmentationFiles,CurrentLayer,List_Series=[]):
        #try:
        List_Series1=SegmentChains(PathtoSegmentationFiles,CurrentLayer,nextLayer,previouslayer,List_Series1)
        #List_Series=[[[layer,segment],...,[layer,segment]],...,[[layer,segment],...,[layer,segment]]]#
    
    List_Series2=filter(lambda a: len(a)>1, List_Series1)#filter the list for one pixel segments
    import pickle
    f = open(PathtoSegmentationFiles+'/output.txt', 'w')
    
    pickle.dump(List_Series2, f)
    f=0#
    '''
    #f2 = open(PathtoSegmentationFiles+'/output10.txt', 'r')
    #List_Series2 = pickle.load(f2)
    #Series_Coordinates2(PathtoSegmentationFiles,List_Series2,layerPath,0.07)
    centrespath=outpath+'Centres/'
    CombineCentreFiles(centrespath)
    layerpath=centrespath
    trunkbiomass(centrespath,centrespath+'Tbiomas.tif',layerpath,step)# find trunks biomass for the lot
    arr1=0# free the space as it is saved as 'Tbiomas.tif'
    outimagepath=ThrusholdImagePath
    trim_images_trunks(ThrusholdImagePath, centrespath,outimagepath,step)# use centres found in trunkbiomass to delet trunks from the images
    #resegment trimmed images
    #try:
    for layer in layers2:#
        #images=FindFilesInFolder(slicesnImagePath+'Thrushold2/','*png')#
        try:
            segmentIm(outimagepath,'t'+layer[0][0:-4]+'.png', outimagepath, 't'+layer[0][0:-4]+'.tif')
            #MakeSegmentsFiles(SegmentedImagePath,SegmentedImage,SegmentsFilesPath,Xmin,Ymin,step)
            MakeSegmentsFiles(outimagepath,'t'+layer[0][0:-4]+'.tif',outimagepath+'t'+layer[0][0:-4]+'/',Xmin,Ymin,step)
        except:
            layers1.remove(layer)
    #parinting the new segmentation process results
    for layer in range(len(layers1[:-1])):
        currentlayer=layers1[layer]#
        nextlayer=layers1[layer+1]
        currentlayer,nextlayer
        try:
            list1=parinting(outimagepath, 't'+currentlayer[0][0:-4]+'.tif','t'+nextlayer[0][0:-4]+'.tif')
        except:
            pass
    List_Series1=[]
    for ii in range(len(layerAll[:-2])):
        try:
            CurrentLayer='t'+layerAll[ii+1][0]    
            nextLayer='t'+layerAll[ii+2][0]
            previouslayer='t'+layerAll[ii][0]
            List_Series1=SegmentChains(outimagepath,CurrentLayer,nextLayer,previouslayer,List_Series1)
        except:
            print('missing layer', ii,ii+1,ii-1)
        
            #lllllll
            #SegmentChains(PathtoSegmentationFiles,CurrentLayer,List_Series=[]):
            #try:
            
        #List_Series=[[[layer,segment],...,[layer,segment]],...,[[layer,segment],...,[layer,segment]]]#
    #except:
    #print('error')
    List_Series2=filter(lambda a: len(a)>3, List_Series1)#filter the list for one pixel segments
    import pickle
    f = open(outimagepath+'/output.txt', 'w')
    pickle.dump(List_Series2, f)           
    List_Series_pixels_Coordinates=Series_pixels_Coordinates(outimagepath,List_Series2)
    #Series_Coordinates(PathtoSegmentationFiles,List_Series2[0:2],layerPath,step)
    f = open(outimagepath+'/List_Series_pixels_Coordinates.txt', 'w')
    pickle.dump(List_Series_pixels_Coordinates, f)
    f=0# 
    #List_Series_Coordinates=Series_Coordinates(PathtoSegmentationFiles,List_Series2[0:2],layerPath,step)
    '''
    List_Series2=pickle.load(open(outimagepath+'/output.txt'))
    List_Series_pixels_Coordinates=pickle.load(open(outimagepath+'/List_Series_pixels_Coordinates.txt'))
    
    '''
    #
        #except:
        #    print('missing layer',CurrentLayer)
        ##### 7. for each series find the closest centre
    #writetofile(path,file1,list1,NoOfColumns=3)
    #f = open(outimagepath+'/List_Series_pixels_Coordinates.txt', 'r')
    #List_Series_pixels_Coordinates= pickle.load(f)
    teststep=30
    #centrespath=outpath+'Centres/'
    #CombineCentreFiles(centrespath)
    #for series in all_series:
    #ClosestCentre(List_Series_Coordinates,centres)
    ##### 8. for each series find the farthest and the closest points to the centre and the shortest distance between them depending on the segments centres of gravity
    #ClosestFarthestPoints(List_Series_Coordinates_ClosestCentre)
    ##### 9. fit cylinder to the closest 3pixels of the series to the centre
    #PointsInSegment2(list1,SegmentFilePath,segmentfile,Step)
    ##### 10.calculate the biomass for each m2 
    centres=readtolist(outpath+'Centres/','all_layers_all_centres.csv',6)
    layerpath=path+plot1+'/noground/'
    #BranchBiomass(SortedBranchPoints,radius,Xmin,Ymin,Xmax,Ymax,step)
    '''
    list3=readtolist(outpath+'Centres/','ellipse30.csv',6)
    i=0
    for ellipse in list3:
         i+=1
         writetofile(outpath+'Centres/',str(i)+'.csv',[ellipse],6)
    '''
    # each series
    # find the closest point
    # find the farthest point
    # find all pixels in 30cm to 30+ 3 x spread
    # find points of these pixels
    # PCA points
    # cylinder fitting >> R
    # while distance less than farthest point distance, distance+=30, calculate Rs, if PCA score<4R, compute biomass volume, distribute biomass with pixels, else add node, segment pixels biond, work on each segment one by one 
    BranchBiomass2(List_Series_pixels_Coordinates,step1,centrespath+'Bbiomas.tif',centres,layerpath)
    
