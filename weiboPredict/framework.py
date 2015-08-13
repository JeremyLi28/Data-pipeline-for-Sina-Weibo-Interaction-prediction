# Name: framework.py
# Usage: basic operations
# Author: Chen Li
import pandas as pd

weibo_train_data = None
weibo_predict_data = None
def loadData():
	global weibo_train_data 
	weibo_train_data= pd.read_csv('data/weibo_train_data.txt',sep='\t', 
		names=['uid','mid','time','forward_count','comment_count','like_count','context'])
	global weibo_predict_data 
	weibo_predict_data = pd.read_csv('data/weibo_predict_data.txt',sep='\t',
		names=['uid','mid','time','context'])

if __name__ == "__main__":
	loadData()