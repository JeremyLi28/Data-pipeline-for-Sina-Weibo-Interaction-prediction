# Name: framework.py
# Usage: basic operations
# Author: Chen Li
import pandas as pd
import csv
import re

weibo_train_data = None
weibo_predict_data = None

def loadData():
	global weibo_train_data 
	weibo_train_data= pd.read_csv('data/weibo_train_data.txt',sep='\t', 
		quoting=csv.QUOTE_NONE,names=['uid','mid','time','forward_count','comment_count','like_count','context'])
	global weibo_predict_data 
	weibo_predict_data = pd.read_csv('data/weibo_predict_data.txt',sep='\t',
		quoting=csv.QUOTE_NONE,names=['uid','mid','time','context'])

def genResult(file, data):
	data.to_csv('result/'+file+'.csv',sep=',',float_format='%d')
	data.to_csv('result/'+file+'.txt',sep=',',float_format='%d',index=False,header=False)
	f=open('result/'+file+'.txt','r')
	context = f.read()
	f.close()
	context = re.sub(',(?=\w\w)','\t',context)
	context = re.sub(',(?=\d,\d,\d)','\t',context)
	f=open('result/test.txt','w')
	f.write(context)
	f.close()

if __name__ == "__main__":
	loadData()