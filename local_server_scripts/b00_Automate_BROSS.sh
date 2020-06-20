#!/bin/bash


cd /home/saswms/ShahryarWork/BROSS/

nn=1
i=$(date +"%Y" -d "$nn days ago")	 ## "today")			## Year
j=$(date +"%m" -d "$nn days ago")	 ## "today")			## Month
k=$(date +"%d" -d "$nn days ago")	 ##"today")			## Day
j=$((10#$j)) # converting to decimal form octal in case of 08 or 09
#~ k=$((10#$k))

yymm=$i-$(printf %02d $j) #-$(printf %02d $k)
echo $yymm
 

#~ ## Download reflectance for all sites and store in Processing/Reflectance
/home/saswms/anaconda2/bin/python3 b01_DownloadReflectance_Landsat8.py  

## Run ANN/regression models and save SSC timeseries in Processing/Timeseries
/home/saswms/anaconda2/bin/python3 b02_BROSS_RunANNandRegressionModels.py

## Create SSC maps, store in SASWMS drive and download to Processing/Raster
/home/saswms/anaconda2/bin/python3 b03_SedimentVis_ToDrive.py

## Create ANN SSC maps, save to Processing/Raster/
/home/saswms/anaconda2/bin/python3 b04_SedimentVis_ANN.py


cd /home/saswms/ShahryarWork/BROSS/Processed/Timeseries/
wput ssc_*.csv ftp://saswms:S@swe2015@128.95.45.89/../../opt/lampp/htdocs/bross/timeseries/


cd /home/saswms/ShahryarWork/BROSS/Processed/Raster
wput *${yymm}*.tif ftp://saswms:S@swe2015@128.95.45.89/../../opt/lampp/htdocs/bross/maps/
