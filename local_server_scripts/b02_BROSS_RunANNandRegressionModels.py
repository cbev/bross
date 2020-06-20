#!/bin/env python



import os
#~ %tensorflow_version 2.x # use the latest TensorFlow version
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import datetime
import pickle



# Code to BROSS files
path_to_BROSS='/home/saswms/ShahryarWork/BROSS/'
reflectance_dir = 'Processed/Reflectance/'
 
os.chdir(path_to_BROSS) 

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def runModels(rs_data_file, roiID):
	# Import the satellite data. See the fourth bullet point in the text block at the top.
	
	# Load dictonary with values to scale the ANN model inputs
	# keys are: 'L8S2_mean', 'L8S2_std', 'L5L7_mean', 'L5L7_std', 'M_mean', 'M_std'
	pickle_in = open("scale_matrix.pkl","rb")
	scale_matrix = pickle.load(pickle_in)

	# Function to mport the datafile and create the additional columns needed for the ANN model input, and put columns in the right order.
	def import_rs_df(filename):
		rs_df=pd.read_csv(filename)
		rs_df['date']=pd.to_datetime(rs_df['date'],format='%Y%m%d')
		if 'flag' in rs_df.columns: rs_df=rs_df[rs_df.flag!='bad']
		rs_df = rs_df.dropna(how='all')
		rs_df.index=rs_df['date'] ##pd.to_datetime(rs_df.index)    
		rs_df = rs_df.groupby(rs_df.index).mean() # take mean of satellite data collected on the same day
		rs_df['total']=rs_df.red+rs_df.blue+rs_df.green;
		rs_df['rn']=rs_df.red/rs_df.nir;
		rs_df['rg']=rs_df.red/rs_df.green;
		rs_df['rb']=rs_df.red/rs_df.blue;
		rs_df['sp1']=(rs_df.red+rs_df.green)/2;
		rs_df['sp2']=(rs_df.red/rs_df.green)+rs_df.nir;
		rs_df['month']=rs_df.index.month
		rs_df=rs_df[['LID', 'blue', 'green', 'nir', 'red', 'total', 'rb', 'rg', 'rn', 'sp1', 'sp2', 'month']]
		rs_df=rs_df.sort_index()
		return rs_df   

	# Import the satellite data
	df=import_rs_df(rs_data_file)
	

	# Define which satellite(s) to get data from- uncomment option: 2=Sentinel-2; 3=MODIS; 5, 7, 8= Landsat 5, 7, 8
	#df.LID=df.LID.astype(int) # Change the satellite ID ('LID') to an integer

	### Landsat 8 and Sentinel-2
	L8S2_df=df[df['LID'].isin([2, 8])] 
	L8S2_dates=L8S2_df.index
	L8S2_df.reset_index(inplace=True, drop=True)
	L8S2_df_scaled=(L8S2_df.values-scale_matrix['L8S2_mean'])/scale_matrix['L8S2_std'] # scale the data

	### Landsat 5 and Landsat 7
	L5L7_df=df[df['LID'].isin([5, 7])] 
	L5L7_dates=L5L7_df.index
	L5L7_df.reset_index(inplace=True, drop=True)
	L5L7_df=L5L7_df.drop(labels=['LID'],axis=1)
	L5L7_df_scaled=(L5L7_df.values-scale_matrix['L5L7_mean'])/scale_matrix['L5L7_std']

	### MODIS
	M_df=df[df['LID'].isin([3])]    
	M_dates=M_df.index
	M_df.reset_index(inplace=True, drop=True)
	M_df=M_df.drop(labels=['LID','total'],axis=1)
	M_df_scaled=(M_df.values-scale_matrix['M_mean'])/scale_matrix['M_std']


	# Compare ANN and regression- Landsat 8, Sentinel-2
	if L8S2_df_scaled.size==0:
		print('No Landsat 8 or Sentinel 2 data in the input data\n')
		ann_output_L8S2=pd.DataFrame(); reg_output_L8S2=pd.DataFrame()
	else: 
		# Run the ANN model using the inputs developed above
		ann_model_L8S2 = tf.keras.models.load_model('ann_L8S2.h5')
		ann_output_L8S2=pd.DataFrame(index=L8S2_dates.values, data=ann_model_L8S2.predict(L8S2_df_scaled), columns=['output'])

		# Run the regression model
		def reg_model_L8S2 (red_over_green):
			ssc=0.161*np.exp(8.871*(red_over_green)-1.15)
			return ssc
		reg_output_L8S2=pd.DataFrame(index=L8S2_dates.values, data=reg_model_L8S2 (L8S2_df.rg.values), columns=['output'])
	  
	  
		#~ fig1, ax1 = plt.subplots(1,1,figsize=(12,4))
		#~ ax1.plot(reg_output_L8S2.index,reg_output_L8S2.output, 'g.-', linewidth=2, label='Regression, r2=0.66, RMSE=135 mg/L')
		#~ ax1.plot(ann_output_L8S2.index,ann_output_L8S2.output,'b.-', linewidth=2, label='ANN, r2=0.74, RMSE=139 mg/L')
		#~ ax1.legend(loc='best')
		#~ ax1.set_title('Landsat 8, Sentinel 2\nTime series of ANN and regression models')
		#~ ax1.set_ylabel('Model output (SSC, mg/L)')
		#~ ax1.set_xlabel('Date');
		#~ plt.show()

		##write to csv
		print('\n Writing SSC timeseries for', roiID)
		print('First 3 rows\n', L8S2_df.head(3))
		output_L8S2_merge = pd.DataFrame(index=L8S2_dates.values)
		output_L8S2_merge['date'] = L8S2_dates.values
		output_L8S2_merge['ann'] = ann_model_L8S2.predict(L8S2_df_scaled)
		output_L8S2_merge['regression'] = reg_model_L8S2(L8S2_df.rg.values)
		output_L8S2_merge.to_csv('Processed/Timeseries/ssc_'+ roiID+'.csv',index=False)


runModels(reflectance_dir + 'reflectL8_baruria.csv','baruria')
runModels(reflectance_dir + 'reflectL8_bahadurabad.csv','bahadurabad')
runModels(reflectance_dir + 'reflectL8_hardinge_bridge.csv','hardinge_bridge')
runModels(reflectance_dir + 'reflectL8_mawa.csv','mawa')
runModels(reflectance_dir + 'reflectL8_naria.csv','naria')
