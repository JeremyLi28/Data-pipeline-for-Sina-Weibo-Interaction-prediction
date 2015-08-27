# Name: framework.py
# Usage: basic operations
# Author: Chen Li
import pandas as pd
import numpy as np
import csv
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.externals import joblib

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

def cleanText(contexts):
	f=open('data/stopwords.txt','r')
	stopwords = [l.strip() for l in f.readlines()]
	for i in range(len(stopwords)):
		stopwords[i] = stopwords[i].decode('utf8')
	f.close()

	i=0
	cleans = []
	for context in contexts:
	    context = re.sub("http://.*\w$","",context)
	    #context = re.sub("\[.{0,4}\]","",context)
	    #context = re.sub("\\pP|\\pS", "",context)
	    context = re.sub("\s","",context)
	    context = re.sub("\d","",context)
	    text = jieba.lcut(context)
	    clean = [t for t in text if t not in stopwords]
	    cleans.append(clean)
	    i=i+1
	    if i%10000==0:
	    	print str(i)+'/'+str(len(contexts))
	return pd.Series(cleans)

def train(start,end,label,feature_type,model_type):
	global weibo_train_data
	train_context_clean = Series.from_csv('data/train_context_clean.csv')
	weibo_train_data['context_clean'] = train_context_clean
	if model_type=="LR":
		vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 100) 
		train_features = vectorizer.fit_transform(								/
							weibo_train_data[(weibo_train_data['time']<=end) 		/
							& (fw.weibo_train_data['time']>=start)].context_clean)
		train_features = train_features.toarray()
		train_labels = weibo_train_data[(weibo_train_data['time']<=end) 		/
							& (fw.weibo_train_data['time']>=start)][label]

		model = linear_model.LinearRegression()
		model.fit(train_features,train_labels)
		print '====='+feature_type+'_'+model_type+'====='
	# The coefficients
	print 'Coefficients: \n', model.coef_
	# The mean square error
	print "Residual sum of squares: %.2f" % /
		np.mean((model.predict(train_features) - train_labels) ** 2)
	# Explained variance score: 1 is perfect prediction
	print 'Variance score: %.2f' % model.score(train_features, train_labels)

	joblib.dump(model,feature_type+'_'+model_type+'_'+start+'_'+end+'.model')
	return model






if __name__ == "__main__":
	loadData()