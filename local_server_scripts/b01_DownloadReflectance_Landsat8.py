#!/bin/env python


import ee
import csv
import time, datetime
from datetime import date
import folium
import tensorflow as tf
#~ %tensorflow_version 2.x
import numpy as  np
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize() 

import os
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import layers
from sklearn.model_selection import train_test_split
from sklearn import metrics
import datetime

# Installs geemap package
import subprocess

try:
    import geemap
except ImportError:
    print('geemap package not installed. Installing ...')
    subprocess.check_call(["python", '-m', 'pip', 'install', 'geemap'])

# Checks whether this notebook is running on Google Colab
try:
    import google.colab
    import geemap.eefolium as emap
except:
    import geemap as emap
import math


## ROIs

baruria = ee.Geometry.Polygon(
	[[[89.80950774646578, 23.791402479605882],
	  [89.78890838123141, 23.763754581376407],
	  [89.80693282581149, 23.745528890307813],
	  [89.82109357160516, 23.758098597927344],
	  [89.83422698474703, 23.770667106958186],
	  [89.8240135299103, 23.77969989232177]]])
study_area = ee.Geometry.Polygon(
	[[[90.28474268098603, 26.387078420504142],
	  [89.69135269545075, 26.170313830439515],
	  [89.42762530172024, 25.67616695676571],
	  [89.36169458085533, 25.239616790522636],
	  [89.48254524280833, 24.581850763551575],
	  [89.58015173425827, 24.252502899817316],
	  [89.53470635833526, 23.957106239177833],
	  [89.1562918840259, 24.059019860434688],
	  [89.07511749579908, 24.218055878302096],
	  [88.81781988193529, 24.316781384846273],
	  [88.21008160821498, 24.916326887149705],
	  [87.84526911317437, 25.309161902340232],
	  [87.471661073586, 25.110317139481474],
	  [87.73539691867677, 24.52186757426558],
	  [88.08703228380523, 24.30173790734487],
	  [88.45543175272303, 24.196739172444982],
	  [88.66940534479181, 24.031053963646222],
	  [89.25181366037896, 23.779914713546734],
	  [89.70231218626031, 23.676798087886194],
	  [90.08693285558138, 23.336722835789214],
	  [90.32867565089384, 23.05391768598817],
	  [90.50449061688937, 22.788177292646743],
	  [90.84796078068916, 22.891955860367847],
	  [90.83413858096696, 22.89204433605182],
	  [90.73524751652712, 23.296354964671078],
	  [90.41658618660233, 23.558503646454902],
	  [89.90564775328369, 23.817613996660608],
	  [89.86718113457448, 23.997013465924663],
	  [89.91112100660393, 24.24164344658607],
	  [89.88914608816685, 25.06057392873872],
	  [90.06496667243823, 25.913614998174296],
	  [90.90010655797573, 26.11111943723229],
	  [90.74626893983213, 26.465798912657412]]])
mawa = ee.Geometry.Polygon(
	[[[90.27767523070543, 23.49246965593073],
	  [90.22136841652832, 23.4187675142901],
	  [90.2646273713624, 23.39860144958394],
	  [90.32642779137961, 23.471055813959982]]])
bahadurabad = ee.Geometry.Polygon(
	[[[89.75825388873409, 25.244834931218396],
	  [89.56324656451534, 25.25477158829473],
	  [89.54127390826534, 25.12801840182916],
	  [89.76374705279659, 25.132991598478295]]])
hardinge_bridge = ee.Geometry.Polygon(
	[[[89.0129066086065, 24.076195189413795],
	  [89.01771312716119, 24.063969863276075],
	  [89.04105907442681, 24.072747138484416],
	  [89.03590923311822, 24.08528506105656]]])
naria = ee.Geometry.Polygon(
	[[[90.40167488861573, 23.348435287415562],
	  [90.3863970260669, 23.328733286373964],
	  [90.39854403123695, 23.322547154683622],
	  [90.41137643586853, 23.31541429096687],
	  [90.43051399994386, 23.30429874699429],
	  [90.44613518524659, 23.32495016772666]]])
  



  # Small ROI to test the code over
#~ ROI= ee.Geometry.Polygon([[[89.81473388326633,23.8709610461413], [89.58676757467258,23.77422694334494],[89.81748046529758,23.67993619559253], [89.9122375453757,23.77674042915862],[89.81473388326633,23.8709610461413]]])

def downloadL8(ROI,startDate,endDate,roiID):

	# User-set Thresholds
	cloud_thresh = ee.Number(0.3); # min percent of non-cloudy
	ref_thresh_L5L7= ee.Number(0.4); # for red+blue band
	ref_thresh_L8= ee.Number(0.4); # for red+blue band
	red_thresh= ee.Number(0.4); # for red band


	# Functions to process image data
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

	#*****Calc Vis and NIR Reflectance************************************************************
	def AvgNIR(img):
	  I = ee.Image(img)
	  temp = I.select('nir')
	  avgNIR = temp.reduceRegion(
		reducer= ee.Reducer.median(),
		geometry= ROI,
		scale= 90,
		maxPixels= 6098838800)
	  return I.set('avgNIR',avgNIR.get('nir'))


	def AvgRED(img):
	  I = ee.Image(img)
	  temp = I.select('red')
	  avgRED = temp.reduceRegion(
		reducer= ee.Reducer.median(),
		geometry= ROI,
		scale= 90,
		maxPixels= 6098838800)
	  return I.set('avgRED',avgRED.get('red'))


	def AvgGRN(img):
	  I = ee.Image(img)
	  temp = I.select('green')
	  avgGRN = temp.reduceRegion(
		reducer= ee.Reducer.median(),
		geometry= ROI,
		scale= 90,
		maxPixels= 6098838800)
	  return I.set('avgGRN',avgGRN.get('green'))


	def AvgBLU(img):
	  I = ee.Image(img)
	  temp = I.select('blue')
	  avgBLU = temp.reduceRegion(
		reducer= ee.Reducer.median(),
		geometry= ROI,
		scale= 90,
		maxPixels= 6098838800)
	  return I.set('avgBLU',avgBLU.get('blue'))


	def AvgBrightness(img):
	  I = ee.Image(img)
	  temp = I.select('red')
	  avgBRT = temp.reduceRegion(
		reducer= ee.Reducer.mean(),
		geometry= ROI,
		scale= 90,
		maxPixels= 6098838800)
	  return I.set('avgBright',avgBRT.get('red'))

	def Landsat_ID_8(img):
	  return img.set('LANDSAT_ID',ee.Number(8))

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
	  

	# Functions to process image data
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

	#*****Calc Vis and NIR Reflectance************************************************************
	def AvgNIR(img):
	  I = ee.Image(img)
	  temp = I.select('nir')
	  avgNIR = temp.reduceRegion(
		reducer= ee.Reducer.median(),
		geometry= ROI,
		scale= 90,
		maxPixels= 6098838800)
	  return I.set('avgNIR',avgNIR.get('nir'))


	def AvgRED(img):
	  I = ee.Image(img)
	  temp = I.select('red')
	  avgRED = temp.reduceRegion(
		reducer= ee.Reducer.median(),
		geometry= ROI,
		scale= 90,
		maxPixels= 6098838800)
	  return I.set('avgRED',avgRED.get('red'))


	def AvgGRN(img):
	  I = ee.Image(img)
	  temp = I.select('green')
	  avgGRN = temp.reduceRegion(
		reducer= ee.Reducer.median(),
		geometry= ROI,
		scale= 90,
		maxPixels= 6098838800)
	  return I.set('avgGRN',avgGRN.get('green'))


	def AvgBLU(img):
	  I = ee.Image(img)
	  temp = I.select('blue')
	  avgBLU = temp.reduceRegion(
		reducer= ee.Reducer.median(),
		geometry= ROI,
		scale= 90,
		maxPixels= 6098838800)
	  return I.set('avgBLU',avgBLU.get('blue'))


	def AvgBrightness(img):
	  I = ee.Image(img)
	  temp = I.select('red')
	  avgBRT = temp.reduceRegion(
		reducer= ee.Reducer.mean(),
		geometry= ROI,
		scale= 90,
		maxPixels= 6098838800)
	  return I.set('avgBright',avgBRT.get('red'))

	def Landsat_ID_8(img):
	  return img.set('LANDSAT_ID',ee.Number(8))

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
	  
	  
	 
	# Run functions over the Landsat image collections
	# Landsat 8
	

	L8imgs = (ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterDate(startDate, endDate).filterBounds(ROI))
	if(L8imgs.size().getInfo() > 0):
		
		L8imgs = L8imgs.map(Roundtime)
		L8imgs=L8imgs.select(['B2', 'B3','B4','B5','B6','B7','pixel_qa'],
		  ['blue', 'green', 'red', 'nir', 'swir1','swir2','pixel_qa'])
		L8imgs = L8imgs.map(Convert_scale)
		L8imgs = L8imgs.map(calcArea)
		L8imgs = L8imgs.filter(ee.Filter.gt('ROI_area',ee.Number(ROI.area()).multiply(0.9)))
		L8imgs = L8imgs.map(getCloudsL8)
		L8imgs=L8imgs.select(['blue', 'green', 'red', 'nir', 'swir1','swir2', 'clear_mask', 'pixel_qa'])
		L8natural = L8imgs.select(['nir','red','green','blue', 'pixel_qa'])
		L8imgs = L8imgs.map(maskClouds)
		L8imgs = L8imgs.map(calcIndex)
		L8imgs = L8imgs.map(WaterTests)
		L8imgs = L8imgs.map(maskLand)
		L8imgs = L8imgs.map(calcCloudAreaRatio)
		L8imgs = L8imgs.map(AvgBrightness)
		L8imgs = L8imgs.filter(ee.Filter.gt('CloudAreaRatio',cloud_thresh)) \
		  .filter(ee.Filter.gt('water_pixels',0))
		L8imgs = L8imgs.map(AvgRED);L8imgs = L8imgs.map(AvgGRN); L8imgs = L8imgs.map(AvgBLU); L8imgs = L8imgs.map(AvgNIR)
		L8imgs = L8imgs.map(Landsat_ID_8)

		#Second check after filtering 
		if(L8imgs.size().getInfo() > 0):
			print('Found {} image(s) for {}'.format(L8imgs.size().getInfo(),roiID))
			epoch = (L8imgs.first().get('system:time_start')).getInfo()
			print('--First day',time.strftime('%Y-%m-%d', time.localtime(epoch/1000.0)))

			# Get average of vis and NIR in the ROI; prepare for export
			AverageNIR = ee.Array(L8imgs.aggregate_array('avgNIR'))
			AverageRED = ee.Array(L8imgs.aggregate_array('avgRED'))
			AverageGRN = ee.Array(L8imgs.aggregate_array('avgGRN'))
			AverageBLU = ee.Array(L8imgs.aggregate_array('avgBLU'))
			AverageBright = ee.Array(L8imgs.aggregate_array('avgBright'))
			LandsatID = ee.Array(L8imgs.aggregate_array('LANDSAT_ID'))

			utcdays = ee.Array(L8imgs.aggregate_array('system:time_start')).getInfo()
			Ldays = ee.Array(list(map(lambda x:int(time.strftime('%Y%m%d', time.localtime(x/1000))), utcdays)))
			InputVars=ee.Array.cat([Ldays,AverageBLU, AverageRED, AverageNIR, AverageGRN, LandsatID], 1).getInfo()
			outarr = np.array(InputVars)
			print(outarr[0])
			print('--Writing reflectance to csv for ', roiID)

			with open('Processed/Reflectance/reflectL8_'+roiID+'.csv', "a", newline="") as f:
				writer = csv.writer(f)
				#~ ##writer.writerow(['date', 'blue','red','nir','green','LID'])    #only for new file
				for item in outarr:
					#Write item to outcsv
					writer.writerow([int(item[0]), item[1], item[2],item[3],item[4],item[5]])
		else:
			print('No new imagery found in last 20 days for', roiID)
			
	else:
		print('No new imagery found in last 20 days for', roiID)	
		
		
#### Options###########
edate = datetime.datetime.now()
sdate = datetime.datetime.now() + datetime.timedelta(days=-20)

endDate = ee.Date(edate)  #ee.Date(datttte()) 
startDate = ee.Date(sdate) # ee.Date('1987-01-01')
print('Checking for new imagery from: {} to {}'.format(sdate.strftime('%Y-%m-%d'),edate.strftime('%Y-%m-%d')))


downloadL8(baruria,startDate,endDate,'baruria')
downloadL8(bahadurabad,startDate,endDate,'bahadurabad')
downloadL8(hardinge_bridge,startDate,endDate,'hardinge_bridge')
downloadL8(mawa,startDate,endDate,'mawa')
downloadL8(naria,startDate,endDate,'naria')
