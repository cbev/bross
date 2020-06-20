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
import datetime, time


import pickle
import os.path
import io,sys
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
	
# Installs geemap package
import subprocess

#~ try:
import geemap
#~ except ImportError:
    #~ print('geemap package not installed. Installing ...')
    #~ subprocess.check_call(["python", '-m', 'pip', 'install', 'geemap'])

# Checks whether this notebook is running on Google Colab
try:
    import google.colab
    import geemap.eefolium as emap
except:
    import geemap as emap
import math

#### Options###########
startDate = ee.Date('2016-01-01') # ee.Date('1987-01-01')
endDate =ee.Date(datetime.datetime.now()) # ee.Date(Date.now())

# Set the range for the SSC color bar
scale_min=0
scale_max=2700

L8_reg='0.161*exp(8.871*(red/green)-1.15)'
L5L7_reg='3.56*exp(14.59*(red)+2.15)'

L8_ann = tf.keras.models.load_model('ann_L8S2.h5')
L5L7_ann = tf.keras.models.load_model('ann_L5L7.h5')

cloud_thresh = ee.Number(0.3); # min percent of non-cloudy
ref_thresh_L5L7= ee.Number(0.4); # for red+blue band
ref_thresh_L8= ee.Number(0.4); # for red+blue band
red_thresh= ee.Number(0.4); # for red band

ROI=ee.Geometry.Polygon([[[90.16413884053382, 26.209196974738333],
   [89.74678031423282, 26.100738391475154],
   [89.48320857537313, 25.606591400492505],
   [89.41731426909439, 25.170041435112307],
   [89.53809493254754, 24.512275110256923],
   [89.63564765635006, 24.182927231760612],
   [89.59022706709804, 23.887530552888226],
   [89.21201736919494, 23.989445721096523],
   [89.13088877840296, 24.148482465914736],
   [88.87373323885913, 24.247211199001374],
   [88.1894932068987, 24.76704665462088],
   [88.08837930537571, 24.906480264669362],
   [87.81377448136195, 24.582367284027782],
   [87.85504534180527, 24.529754346908565],
   [88.14334524729725, 24.23218453242052],
   [88.5115402462072, 24.127175909249956],
   [88.72539366129361, 23.961486276685214],
   [89.3074854303443, 23.710339897524896],
   [89.75774379505597, 23.60722268569749],
   [90.1421647597441, 23.26715033513523],
   [90.38378676041935, 22.9843486127382],
   [90.66116522012055, 22.432563897354584],
   [90.92752506469958, 22.548856492400095],
   [91.15785840972656, 22.630272300216063],
   [90.79014168171614, 23.226794447093695],
   [90.47164083568629, 23.488936126396],
   [89.96096999517475, 23.7480397379857],
   [89.92252246213062, 23.92743892374796],
   [89.96643631021006, 24.17206923077994],
   [89.94446602480448, 24.99099954456734],
   [90.12017529698102, 25.844042247245564],
   [90.33995095003816, 25.898418684599672],
   [90.1807113850368, 26.061353911688766],
   [90.16413884053382, 26.209196974738333]]])

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
    geometry= ROI.getInfo(),
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
    geometry= ROI.getInfo(),
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
    geometry= ROI.getInfo(),
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
    geometry= ROI,
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
def ann_SSCL5(img):
  I=ee.Image(img)
  ssc=I.expression(L5L7_reg,
  {'LID':ee.Number(8),'blue':I.select('blue'), 'green': I.select('green'), 'nir': I.select('nir'),'red': I.select('red'),
  'rb':I.select('red')/I.select('blue'),'rg':I.select('red')/I.select('green'),'rn':I.select('red')/I.select('nir'),
   'sp1':(I.select('red')+I.select('green'))/2,'sp2':I.select('red')/I.select('green')+I.select('nir'),
   'month': datetime.datetime.utcfromtimestamp(img['properties']['system:time_start']/1000)})
  ssc=ssc.rename('reg_ssc')
  I=I.addBands(ssc)
  return I

def ann_SSCL7(img):
  I=ee.Image(img)
  ssc=I.expression(L5L7_reg,
  {'LID':ee.Number(8),'blue':I.select('blue'), 'green': I.select('green'), 'nir': I.select('nir'),'red': I.select('red'),
  'rb':I.select('red')/I.select('blue'),'rg':I.select('red')/I.select('green'),'rn':I.select('red')/I.select('nir'),
   'sp1':(I.select('red')+I.select('green'))/2,'sp2':I.select('red')/I.select('green')+I.select('nir'),
   'month': datetime.datetime.utcfromtimestamp(img['properties']['system:time_start']/1000)})
  ssc=ssc.rename('reg_ssc')
  I=I.addBands(ssc)
  return I


def ann_SSCL8(img):
  I = ee.Image(img)
  r = I.select('red').reduceRegion(
    reducer= ee.Reducer.median(),
    geometry= ROI.getInfo(),
    scale= 30,
    maxPixels= 6098838800)
  
  g = I.select('green').reduceRegion(
    reducer= ee.Reducer.median(),
    geometry= ROI.getInfo(),
    scale= 30,
    maxPixels= 6098838800)

  b = I.select('blue').reduceRegion(
    reducer= ee.Reducer.median(),
    geometry= ROI.getInfo(),
    scale= 30,
    maxPixels= 6098838800)
  
  n = I.select('nir').reduceRegion(
    reducer= ee.Reducer.median(),
    geometry= ROI.getInfo(),
    scale= 30,
    maxPixels= 6098838800)
  # arr = r.toArray();

  rv = r.get('red').getInfo()
  bv = b.get('blue').getInfo()
  gv = g.get('green').getInfo()
  nv = n.get('nir').getInfo()

  #~ print(g.get('green').getInfo())
  from datetime import datetime
  utcd = I.get('system:time_start').getInfo()/1000
  dt= datetime.fromtimestamp(utcd)

  #~ print(dt.month)
  input = np.array([[8, bv,rv,gv,nv, bv+gv+rv+nv, rv/bv, rv/gv, rv/nv, (rv+gv)/2, (rv/gv)+nv, dt.month]])
    
  ssc=L8_ann.predict(input)
  print("ANN SSC on {}:{}".format(dt,ssc[0][0]))
  return ssc[0][0]

def reg_avgSSC(img):
  I = ee.Image(img)
  
  temp=I.select('ssc')
  avgSSC = temp.reduceRegion(
    reducer= ee.Reducer.mean(),
    geometry= ROI.getInfo(),
    scale= 90,
    maxPixels= 6098838800)
  return I.set('reg_avgSSC',avgSSC.get('reg_ssc'))

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
  
  
for i in range(-1,0):

	edate = datetime.datetime.now() + datetime.timedelta(days=i)
	sdate = edate + datetime.timedelta(days=-20)

	print(edate)
	if edate.day < 16:
		writedate = datetime.datetime(edate.year,edate.month, 1)
	else:
		writedate = datetime.datetime(edate.year,edate.month, 16)
		
	clipEndDate = ee.Date(edate)  #ee.Date(datttte()) 
	clipStartDate = ee.Date(sdate) # ee.Date('1987-01-01')
  
	print('Checking for new imagery from: {} to {}'.format(sdate.strftime('%Y-%m-%d'),edate.strftime('%Y-%m-%d')))

	L8imgs = (ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterDate(clipStartDate, clipEndDate).filterBounds(ROI))
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
	L8imgs = L8imgs.map(reg_SSCL8)
	#L8imgs = L8imgs.map(ann_SSCL8)
	#~ L8imgs = L8imgs.map(reg_avgSSC)

	## ANN SSC L8
	#~ L8list = L8imgs.toList(L8imgs.size())
	#~ tempann=[]
	#~ for i in range(L8imgs.size().getInfo()):
		#~ print(i)
		#~ image = ee.Image(L8list.get(i))
		#~ ssc = ann_SSCL8(image)
		
		#~ proj = image.projection()
		#~ improj = ee.Image(float(ssc)).rename('ann_ssc').reproject(proj, None, 30)
		#~ pp = image.addBands(improj)
		#~ print(pp.select('ann_ssc').getInfo())
		#~ tempann.append(pp)
		#~ # print(tempann)
	#~ L8imgs = ee.ImageCollection.fromImages(tempann);  
	#~ print(L8imgs.getInfo())




	L8imgs=L8imgs.select(['blue', 'green', 'red', 'nir', 'swir1','swir2', 'clear_mask', 'pixel_qa','reg_ssc'])
	L8imgs_ROI = L8imgs.filter(ee.Filter.gt('ROI_area',ee.Number(ROI.area()).multiply(0.9)))
	L8imgs_ROI = L8imgs_ROI.filter(ee.Filter.gt('CloudAreaRatio',cloud_thresh)) \
	  .filter(ee.Filter.gt('water_pixels',0))

	# Landsat 7
	L7imgs = (ee.ImageCollection('LANDSAT/LE07/C01/T1_SR') \
	  .filterDate(clipStartDate, clipEndDate) \
	  .filterBounds(ROI))
	L7imgs = L7imgs.map(Roundtime)
	L7imgs=L7imgs.select(['B1', 'B2', 'B3','B4','B5','B7', 'sr_cloud_qa','pixel_qa'],
	  ['blue', 'green', 'red', 'nir', 'swir1','swir2', 'sr_cloud_qa','pixel_qa'])
	L7imgs = L7imgs.map(Convert_scale)
	L7imgs = L7imgs.map(etm2oli)
	L7imgs = L7imgs.map(calcArea)
	L7imgs = L7imgs.map(getCloudsL5L7)
	L7imgs=L7imgs.select(['blue', 'green', 'red', 'nir', 'swir1','swir2', 'clear_mask', 'sr_cloud_qa','pixel_qa'])
	L7natural = L7imgs.select(['nir','red','green','blue', 'pixel_qa'])
	L7imgs = L7imgs.map(maskClouds)
	L7imgs = L7imgs.map(calcIndex)
	L7imgs = L7imgs.map(WaterTests)
	L7imgs = L7imgs.map(maskLand)
	L7imgs = L7imgs.map(calcCloudAreaRatio)
	L7imgs = L7imgs.map(reg_SSCL5L7)
	L7imgs = L7imgs.map(reg_avgSSC)
	L7imgs=L7imgs.select(['blue', 'green', 'red', 'nir', 'swir1','swir2', 'clear_mask', 'pixel_qa','reg_ssc'])
	L7imgs_ROI = L7imgs.filter(ee.Filter.gt('ROI_area',ee.Number(ROI.area()).multiply(0.9)))
	L7imgs_ROI = L7imgs_ROI.filter(ee.Filter.gt('CloudAreaRatio',cloud_thresh)) \
	  .filter(ee.Filter.gt('water_pixels',0))

	# Landsat 5
	L5imgs = (ee.ImageCollection('LANDSAT/LT05/C01/T1_SR') \
	  .filterDate(clipStartDate, clipEndDate) \
	  .filterBounds(ROI))
	L5imgs = L5imgs.map(Roundtime)
	L5imgs=L5imgs.select(['B1', 'B2', 'B3','B4','B5','B7', 'sr_cloud_qa','pixel_qa'],
	  ['blue', 'green', 'red', 'nir', 'swir1','swir2', 'sr_cloud_qa','pixel_qa'])
	L5imgs = L5imgs.map(Convert_scale)
	L5imgs = L5imgs.map(etm2oli)
	L5imgs = L5imgs.map(calcArea)
	L5imgs = L5imgs.map(getCloudsL5L7)
	L5natural = L5imgs.select(['nir','red','green','blue', 'pixel_qa'])
	L5imgs=L5imgs.select(['blue', 'green', 'red', 'nir', 'swir1','swir2', 'clear_mask', 'pixel_qa'])
	L5imgs = L5imgs.map(maskClouds)
	L5imgs = L5imgs.map(calcIndex)
	L5imgs = L5imgs.map(WaterTests)
	L5imgs = L5imgs.map(maskLand)
	L5imgs = L5imgs.map(calcCloudAreaRatio)
	L5imgs = L5imgs.map(reg_SSCL5L7)
	L5imgs = L5imgs.map(reg_avgSSC)
	L5imgs=L5imgs.select(['blue', 'green', 'red', 'nir', 'swir1','swir2', 'clear_mask', 'pixel_qa','reg_ssc'])
	L5imgs_ROI = L5imgs.filter(ee.Filter.gt('ROI_area',ee.Number(ROI.area()).multiply(0.9)))
	L5imgs_ROI = L5imgs_ROI.filter(ee.Filter.gt('CloudAreaRatio',cloud_thresh)) \
	  .filter(ee.Filter.gt('water_pixels',0))

	# Merge
	combine_ROI=L8imgs_ROI.merge(L7imgs_ROI).merge(L5imgs_ROI) # merge(L4imgs)
	combine=L8imgs.merge(L7imgs).merge(L5imgs) # merge(L4imgs)
	combine_nat=L8natural.merge(L7natural).merge(L5natural) #merge(L4natural).
	combine_ROI=combine_ROI.sort('system:time_start')
	combine=combine.sort('system:time_start')
	combine_nat=combine_nat.sort('system:time_start')



	#process
	
	img_ssc = ee.Image(combine.select(['reg_ssc']).filterDate(clipStartDate, clipEndDate).median().clip(ROI))
	#~ img_ssc_ann = ee.Image(combine.select(['ann_ssc']).filterDate(clipStartDate, clipEndDate).first().clip(ROI))
	img_natural = ee.Image(combine_nat.select(['red','green','blue']).filterDate(clipStartDate, clipEndDate).median().clip(ROI))


	write_day = writedate.strftime('%Y-%m-%d')

	coll_ssc = combine.select(['reg_ssc']).filterDate(clipStartDate, clipEndDate)
	
	latest_ssc = ee.Image(coll_ssc.sort('system:time_start', False).limit(1, 'system:time_start').first())

	epoch = (latest_ssc.get('system:time_start')).getInfo()
	latest_day = time.strftime('%Y-%m-%d', time.localtime(epoch/1000.0))
	print('--Latest day',latest_day)



	print('--Exporting SSCMap_reg_{} to Drive'.format(write_day))

	task_config = {
		'fileNamePrefix': 'SSCMap_reg_' + write_day,
		'crs': 'EPSG:4326',
		'scale': 100,
		'maxPixels': 1e10,
		'fileFormat': 'GeoTIFF',
		'skipEmptyTiles': True,
		'region': ROI,
		'folder': 'BROSS_SSC_raster'
		}

	task = ee.batch.Export.image.toDrive(img_ssc, str('ssc-export'), **task_config)
	task.start()
	import time
	while task.active():
		time.sleep(30)
		print(task.status())
		
	
	#~ print('--Exporting SSCMap_ann_{} to Drive'.format(write_day))

	#~ task_config = {
		#~ 'fileNamePrefix': 'SSCMap_ann_' + write_day,
		#~ 'crs': 'EPSG:4326',
		#~ 'scale': 100,
		#~ 'maxPixels': 1e10,
		#~ 'fileFormat': 'GeoTIFF',
		#~ 'skipEmptyTiles': True,
		#~ 'region': ROI,
		#~ 'folder': 'BROSS_SSC_raster'
		#~ }

	#~ task = ee.batch.Export.image.toDrive(img_ssc_ann, str('ssc-ann-export'), **task_config)
	#~ task.start()
	#~ import time
	#~ while task.active():
		#~ time.sleep(30)
		#~ print(task.status())

	############################################################################################
	############################################################################################
	#### b04_Download_SedimentVis_fromDrive.py


	day = write_day
	print('Downloading',day)
	SCOPES = ['https://www.googleapis.com/auth/drive']

	creds = None

	if os.path.exists('token.pickle'):
		with open('token.pickle', 'rb') as token:
			creds = pickle.load(token)

	if not creds or not creds.valid:
		if creds and creds.expired and creds.refresh_token:
			creds.refresh(Request())
		else:
			flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
			creds = flow.run_local_server(port=0)
			# Save the credentials for the next run
		with open('token.pickle', 'wb') as token:
			pickle.dump(creds, token)

	service = build('drive', 'v3', credentials=creds)


	page_token = None
	while True:
		response = service.files().list(q="mimeType='image/tiff'",
											  spaces='drive',
											  fields='nextPageToken, files(id, name)',
											  pageToken=page_token).execute()
		for file in response.get('files', []):
			# Process change
			if('reg_' + day in file.get('name')):
				print('Found latest map: %s (%s)' % (file.get('name'), file.get('id')))
				file_id = file.get('id')
				file_nm = file.get('name')
		page_token = response.get('nextPageToken', None)
		if page_token is None:
			break


	print('Downloading ',file_nm)
	request = service.files().get_media(fileId=file_id)
	fh = io.FileIO('Processed/Raster/'+file_nm,mode='wb')
	downloader = MediaIoBaseDownload(fh, request)
	done = False
	while done is False:
		status, done = downloader.next_chunk()
		print(status)
		print ("Completed %d%%." % int(status.progress() * 100))
		
	#download ann
	#~ while True:
		#~ response = service.files().list(q="mimeType='image/tiff'",
											  #~ spaces='drive',
											  #~ fields='nextPageToken, files(id, name)',
											  #~ pageToken=page_token).execute()
		#~ for file in response.get('files', []):
			#~ # Process change
			#~ if('ann_' + day in file.get('name')):
				#~ print('Found latest map: %s (%s)' % (file.get('name'), file.get('id')))
				#~ file_id = file.get('id')
				#~ file_nm = file.get('name')
		#~ page_token = response.get('nextPageToken', None)
		#~ if page_token is None:
			#~ break


	#~ print('Downloading ',file_nm)
	#~ request = service.files().get_media(fileId=file_id)
	#~ fh = io.FileIO('Processed/Raster/'+file_nm,mode='wb')
	#~ downloader = MediaIoBaseDownload(fh, request)
	#~ done = False
	#~ while done is False:
		#~ status, done = downloader.next_chunk()
		#~ print(status)
		#~ print ("Completed %d%%." % int(status.progress() * 100))

