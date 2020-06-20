from __future__ import print_function
import ee
import folium
#~ %tensorflow_version 2.x
import tensorflow as tf

try:
    ee.Initialize()
except Exception as e:
	ee.Authenticate()
	ee.Initialize() 

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import layers
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import scale

import datetime, time  
import math


import pickle
import os.path
import io,sys
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload

# Installs geemap package
import subprocess

import geemap

import math

#### Options###########

homedir = '/home/saswms/ShahryarWork/BROSS/'

for i in range(-1,0):
  #~ i = -1

  edate = datetime.datetime.now() + datetime.timedelta(days=i)    ##datetime.datetime(2019,10,21) #
  sdate = edate + datetime.timedelta(days=-20)
  print('Checking for new imagery from: {} to {}'.format(sdate.strftime('%Y-%m-%d'),edate.strftime('%Y-%m-%d')))
  startDate = ee.Date(sdate)  
  endDate = ee.Date(edate)
    
  if edate.day < 16:
    writedate = datetime.datetime(edate.year,edate.month, 1)
  else:
    writedate = datetime.datetime(edate.year,edate.month, 16)
  write_day = writedate.strftime('%Y-%m-%d')

    
  # Set the range for the SSC color bar
  scale_min=0
  scale_max=2700

  # load mean and standard deviation
  pickle_in = open(homedir + "scale_matrix.pkl","rb")
  scale_matrix = pickle.load(pickle_in)

  L8_reg='0.161*exp(8.871*(red/green)-1.15)'
  L5L7_reg='3.56*exp(14.59*(red)+2.15)'

  L8_ann = tf.keras.models.load_model('ann_L8S2.h5')
  L5L7_ann = tf.keras.models.load_model('ann_L5L7.h5')

  cloud_thresh = ee.Number(0.3); # min percent of non-cloudy
  ref_thresh_L5L7= ee.Number(0.4); # for red+blue band
  ref_thresh_L8= ee.Number(0.4); # for red+blue band
  red_thresh= ee.Number(0.4); # for red band


  #****Round Time***************************************************************************
  def Roundtime(img):
    I = ee.Image(img)
    time = ee.Number(I.get('system:time_start')).round()
    return I.set('system:time_start',time)

  #****Convert Scale of Surface Reflectance Estimates**************************************
  def Convert_scale(img):
    I = ee.Image(img)
    correct_scale = I.select(['blue','green','red','nir','swir1','swir2']).multiply(0.0001)
    return I.addBands(correct_scale,['blue','green','red','nir','swir1','swir2'],True)


  #****Calculate Area**********************************************************************
  def calcArea(img):
    I = ee.Image(img)
    count = I.reduceRegion(
      reducer= ee.Reducer.count(),
      geometry= ROI.geometry().getInfo(),
      scale= 90,
      maxPixels= 6098838800)
    area = ee.Number(count.get('green')).multiply(8100)
    return I.set('ROI_area',area)

  #****Define Cloud Masking*************************************************************
  # function to mask out cloudy pixels.
  def getCloudsL8(img):
    I = ee.Image(img)
    qa = I.select('pixel_qa').int64()
    mask=qa.eq(324).Or(qa.eq(322))\
    .And(I.select('red').lt(red_thresh)) \
    .And(I.select('green').lt(red_thresh)) \
    .And(I.select('blue').lt(red_thresh)) \
    .And(I.select('nir').lt(red_thresh)) \
    .And(I.select('blue').add(I.select('red')).lt(ref_thresh_L8)) 

    mask=mask.rename('clear')
    sum = mask.reduceRegion(
      reducer= ee.Reducer.sum(),
      geometry= ROI.geometry().getInfo(),
      scale= 90,
      maxPixels= 6098838800)
    I = I.set("clear_pixels",sum.get('clear'))
    I = I.addBands(mask.rename('clear_mask'))
    return I


  #****Define Cloud Masking*************************************************************
  def getCloudsL5L7(img):
    I = ee.Image(img)
    qa = I.select('pixel_qa')
    mask =qa.eq(68).Or(qa.eq(66)) \
    .And(I.select('red').lt(red_thresh)) \
    .And(I.select('red').lt(red_thresh)) \
    .And(I.select('green').lt(red_thresh)) \
    .And(I.select('blue').lt(red_thresh)) \
    .And(I.select('nir').lt(red_thresh)) \
    .And(I.select('blue').add(I.select('red')).lt(ref_thresh_L5L7))

    mask=mask.rename('clear')
    sum = mask.reduceRegion(
      reducer= ee.Reducer.sum(),
      geometry= ROI.geometry().getInfo(),
      scale= 90,
      maxPixels= 6098838800)
    I = I.set("clear_pixels",sum.get('clear'))
    I = I.addBands(mask.rename('clear_mask'))
    return I

  def maskClouds(img):
    I = ee.Image(img)
    return I.updateMask(I.select('clear_mask'))
    
  #****Define Index Calculation*********************************************************
  def calcIndex(img):
    I = ee.Image(img)

    MNDWI = I.normalizedDifference(['green','swir1'])
    MNDWI = MNDWI.rename('MNDWI')

    MBSRV = I.select('green').add(I.select('blue'))
    MBSRV = MBSRV.rename('MBSRV')

    MBSRN = I.select('swir1').add(I.select('nir'))
    MBSRN = MBSRN.rename('MBSRN')

    AWEsh = I.select('blue') \
      .add(I.select('green').multiply(2.5)) \
      .add(MBSRN).multiply(-1.5) \
      .add(I.select('swir2').multiply(-0.25))
    AWEsh = AWEsh.rename('AWEsh')

    I = I.addBands(MNDWI)
    I = I.addBands(MBSRV)
    I = I.addBands(MBSRN)
    I = I.addBands(AWEsh)
    return I

  #****Define DSWE water classification ************************************************
  def WaterTests(img):
    I = ee.Image(img)
    MNDWI = I.select('MNDWI')
    MBSRN = I.select('MBSRN')
    MBSRV = I.select('MBSRV')
    AWEsh = I.select('AWEsh')

    #MNDWI > 0.0123
    Test1 = MNDWI.gt(0.0123)

    #mbsrv > mbsrn
    Test2 = MBSRV.gt(MBSRN)

    #awesh > 0.0
    Test3 = AWEsh.gt(0)

    #mndwi > -0.5 && SWIR1 < 1000 && NIR < 1500
    subTest1 = MNDWI.gt(-0.5)
    subTest2 = I.select('swir1').lt(0.1)
    subTest3 = I.select('nir').lt(0.15)
    Test4 = (subTest1.add(subTest2).add(subTest3)).eq(3)

    #mndwi > -0.5 && SWIR2 < 1000 && NIR < 2000
    subTest4 = MNDWI.gt(-0.5)
    subTest5 = I.select('swir2').lt(0.1)
    subTest6 = I.select('nir').lt(0.2)
    Test5 = (subTest4.add(subTest5).add(subTest6)).eq(3)

    TestSum = Test1.add(Test2).add(Test3).add(Test4).add(Test5)
    Class1 = TestSum.gte(4)
    Class2_1 = TestSum.eq(3)

    Class1 = Class1.rename('Water')
    sum = Class1.reduceRegion(
      reducer= ee.Reducer.sum(),
      geometry= ROI.geometry().getInfo(),
      scale= 90,
      maxPixels= 6098838800)
    I = I.set('water_pixels',sum.get('Water'))
    return I.addBands(Class1)


  #****Define Land Masking***************************************************************
  def maskLand(img):
    I = ee.Image(img)
    mask = I.select('Water').gt(0)
    return I.updateMask(mask)

  def calcCloudAreaRatio(img):
    I = ee.Image(img)
    cloudArea = ee.Number(I.get('clear_pixels')).multiply(8100)
    return I.set('CloudAreaRatio',cloudArea.divide(I.get('ROI_area')))

  #*****Calc SSC from regression****************************************************
  def reg_SSCL8(img):
    I=ee.Image(img)
    ssc=I.expression(L8_reg,
    {'blue':I.select('blue'), 'green': I.select('green'), 'red': I.select('red'), 'nir': I.select('nir')})
    ssc=ssc.rename('reg_ssc')
    I=I.addBands(ssc)
    return I

  def reg_SSCL5L7(img):
    I=ee.Image(img)
    ssc=I.expression(L5L7_reg,
    {'blue':I.select('blue'), 'green': I.select('green'), 'red': I.select('red'), 'nir': I.select('nir')})
    ssc=ssc.rename('reg_ssc')
    I=I.addBands(ssc)
    return I

  def reg_avgSSC(img):
    I = ee.Image(img)
    temp=I.select('ssc')
    avgSSC = temp.reduceRegion(
      reducer= ee.Reducer.mean(),
      geometry= ROI.getInfo(),
      scale= 90,
      maxPixels= 6098838800)
    return I.set('reg_avgSSC',avgSSC.get('reg_ssc'))

  #*****Calc SSC from ANN***************************************************
    
  def ann_SSCL8(img,ic):
    Iorg = ee.Image(img)
    refProjection = Iorg.projection();
    I=Iorg.reproject(refProjection, None, 200) #from 30m to 100m
    # I=Iorg
    band_arrs = I.sampleRectangle(region=ROI.geometry(),defaultValue=0)
    # print(band_arrs.getInfo())
    band_arr_r = band_arrs.get('red')
    r = np.array(band_arr_r.getInfo())

    band_arr_g = band_arrs.get('green')
    g = np.array(band_arr_g.getInfo())

    band_arr_b = band_arrs.get('blue')
    b = np.array(band_arr_b.getInfo())

    band_arr_n = band_arrs.get('nir')
    n = np.array(band_arr_n.getInfo())

    print('region size',n.shape)

    utcd = I.get('system:time_start').getInfo()/1000
    dt= datetime.datetime.fromtimestamp(utcd)

    #~ print('mnth',dt.month)
    ssc = np.zeros(r.shape)
    for i,j in np.ndindex(r.shape):
      ## order of inputs : ['LID', 'blue', 'green', 'nir', 'red', 'total', 'rb', 'rg','rn', 'sp1', 'sp2', 'month']
        
      if b[i,j]!=0 and r[i,j]!=0 and  r[i,j]/b[i,j] <=1e4 and r[i,j]/g[i,j] <=1e4 and r[i,j]/n[i,j]<=1e4:
        input_not_scaled = np.array([[8, b[i,j], g[i,j], n[i,j], r[i,j], b[i,j]+g[i,j]+r[i,j], \
                         r[i,j]/b[i,j], r[i,j]/g[i,j], r[i,j]/n[i,j], \
                         (r[i,j]+g[i,j])/2, (r[i,j]/g[i,j])+n[i,j], dt.month]])

        # Scale the inputs to ANN using the scale function
        input_scaled = (input_not_scaled-scale_matrix['L8S2_mean'])/scale_matrix['L8S2_std']


        ssc[i,j]=L8_ann.predict([input_scaled])
       
        if ssc[i,j]>0 or ssc[i,j]< 9999: 
          ssc[i,j] = ssc[i,j]

          #~ print(i,input_scaled,float(ssc[i,j]))

        else:    #check for nan 
          ssc[i,j] = -9999
      else: 
        ssc[i,j] = -9999
        

    ssc_fix = ssc #np.flip(ssc, axis=0)

    # Write SSC array to ASCII
    nrows = r.shape[0]
    ncols = r.shape[1]   
    
    outf = homedir+'Processed/Raster/ANN/SSC_ann_' + write_day+ '_chunk'+str(ic+1)+'.asc'
    f = open(outf, 'w')
    f.write("ncols " + str(ncols) + "\n")
    f.write("nrows " + str(nrows) + "\n")
    f.write("xllcorner " + str(xllcorner) + "\n")
    f.write("yllcorner " + str(yllcorner) + "\n")
    f.write("cellsize " + str(cellsize) + "\n")
    f.write("NODATA_value " + str(nodata) + "\n")
    f.close()

    f_handle = open(outf, 'a')
    np.savetxt(f_handle, ssc_fix, fmt="%.1f")
    f_handle.close()

    return ssc_fix


  #********* Harmonize Landsat*****************
  coefficients = {
  'etm2oli_ols_not_nir': {
    'itcps': ee.Image.constant([0.0003, 0.0088, 0.0061, 0, 0.0254, 0.0172]),
    'slopes': ee.Image.constant([0.8474, 0.8483, 0.9047, 1, 0.8937, 0.9071])
    },
    'etm2oli_ols': {
    'itcps': ee.Image.constant([0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172]),
    'slopes': ee.Image.constant([0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071])
    },
  'oli2etm_ols': {
    'itcps': ee.Image.constant([0.0183, 0.0123, 0.0123, 0.0448, 0.0306, 0.0116]),
    'slopes': ee.Image.constant([0.885, 0.9317, 0.9372, 0.8339, 0.8639, 0.9165])
    }}

  # Define function to apply harmonization transformation.
  def etm2oli(img):
    I = ee.Image(img)
    convert = I.select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']) \
      .multiply(coefficients['etm2oli_ols_not_nir']['slopes']) \
      .add(coefficients['etm2oli_ols_not_nir']['itcps'])
    return I.addBands(convert,['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],True)
    
    
  ### 
  # chunks
  # chunks = ee.FeatureCollection("users/climateClass/chunks_bross");
  chunks = ee.FeatureCollection("users/saswe/strips_bross_allin");

  def getMinMaxCoord(feature):
    listCoords = ee.Array.cat(feature.geometry().coordinates(), 1); 
    xCoords = listCoords.slice(1, 0, 1);
    yCoords = listCoords.slice(1, 1, 2); 
    return feature.set({'xMin':xCoords.reduce('min', [0]).get([0,0]),\
                        'yMin':yCoords.reduce('min', [0]).get([0,0])})
  #Map the area getting function over the FeatureCollection.
  chunks = chunks.map(getMinMaxCoord);


  ## for loop over each chunk
  chunklist = chunks.toList(chunks.size())


  start = time.time()
  for ic in range(chunks.size().getInfo()):
    ROI = ee.Feature(chunklist.get(ic))
    print('\nProcessing SSC_ann_' + write_day+ '_chunk'+str(ic+1)+'.asc')

    #Print the first feature from the collection with the added property.
    xMin = ROI.get('xMin').getInfo();
    #~ #print('xmin:',xMin)
    yMin = ROI.get('yMin').getInfo();
    #~ #print('ymin:',yMin)

    # write to ascii

    xllcorner = xMin
    yllcorner = yMin
    cellsize =  0.00179  #for 200m     (0.00091 for 100m,for 30m use 0.000277778)
    nodata = -9999.0
    # print(ROI.geometry().getInfo())
    
    #~ #print('dates',startDate.getInfo(),endDate.getInfo())
    L8imgs = (ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterDate(startDate, endDate).filterBounds(ROI.geometry()))

   
    L8imgs = L8imgs.map(Roundtime)
    L8imgs=L8imgs.select(['B2', 'B3','B4','B5','B6','B7','pixel_qa'],
      ['blue', 'green', 'red', 'nir', 'swir1','swir2','pixel_qa'])
    L8imgs = L8imgs.map(Convert_scale)
    L8imgs = L8imgs.map(calcArea)
    L8imgs = L8imgs.map(getCloudsL8)
    L8imgs=L8imgs.select(['blue', 'green', 'red', 'nir', 'swir1','swir2', 'clear_mask', 'pixel_qa'])
    L8natural = L8imgs.select(['nir','red','green','blue', 'pixel_qa'])
    L8imgs = L8imgs.map(maskClouds)
    L8imgs = L8imgs.map(calcIndex)
    L8imgs = L8imgs.map(WaterTests)
    L8imgs = L8imgs.map(maskLand)
    L8imgs = L8imgs.map(calcCloudAreaRatio)
    # L8imgs = L8imgs.map(reg_SSCL8)

    print('number of L8 scenes',L8imgs.size().getInfo())

    if L8imgs.size().getInfo()>0:
      L8img = ee.Image(L8imgs.mosaic()).set('system:time_start',L8imgs.first().get('system:time_start'))
      ssc = ann_SSCL8(L8img,ic)
      
  end = time.time()
  print('Total time elapsed {} sec'.format(end - start))



  #Mosaic
  import subprocess,os,glob
  import matplotlib.pyplot as plt
  from rasterio.merge import merge
  from rasterio.plot import show
  from rasterio.transform import from_origin


  import rasterio

  # File and folder paths

  out_fp = os.path.join(homedir, "Processed/Raster/SSCMap_ann_"+write_day+".tif")

  # Make a search criteria to select the DEM files
  search_criteria = "Processed/Raster/ANN/SSC_ann_"+write_day+"_chunk*"
  q = os.path.join(homedir, search_criteria)
  print(q)

  # glob function can be used to list files from a directory with specific criteria
  dem_fps = glob.glob(q)

  # Files that were found:
  dem_fps

  # List for the source files
  src_files_to_mosaic = []

  # Iterate over raster files and add them to source -list in 'read mode'
  for fp in dem_fps:
      src = rasterio.open(fp)
      src_files_to_mosaic.append(src)

  # Merge function returns a single mosaic array and the transformation info
  bnd = [87.813774, 22.432564, 91.157858, 26.209197]
  mosaic, ot = merge(src_files_to_mosaic, bounds=bnd)


  # Copy the metadata
  out_meta = src.meta.copy()

  # Update the metadata
  out_meta.update({"driver": "GTiff" ,
                   "height": mosaic.shape[1],
                   "width": mosaic.shape[2] ,
                   "transform": from_origin(bnd[0], bnd[3] ,0.00179,0.00179),     #north and west coord
                   "crs": "EPSG:4326"
                   }
                  )
  #~ print(mosaic.shape,out_meta)
  print("Mosaicing rasters... ")
  # Write the mosaic raster to disk
  with rasterio.open(out_fp, "w", **out_meta) as dest:
      dest.write(mosaic)
