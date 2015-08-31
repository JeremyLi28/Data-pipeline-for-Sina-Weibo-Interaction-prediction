# Name: framework.py
# Usage: basic operations
# Author: Chen Li
import pandas as pd
import numpy as np
import csv
import re
import jieba
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.externals import joblib

weibo_train_data = None
weibo_predict_data = None
train_log = None
test_log = None
features_log = None

def loadData():
	global weibo_train_data 
	weibo_train_data= pd.read_csv('../data/weibo_train_data.txt',sep='\t', 
		quoting=csv.QUOTE_NONE,names=['uid','mid','time','forward_count','comment_count','like_count','context'])
	global weibo_predict_data 
	weibo_predict_data = pd.read_csv('../data/weibo_predict_data.txt',sep='\t',
		quoting=csv.QUOTE_NONE,names=['uid','mid','time','context'])
	weibo_predict_data.ix[78336].time = '2015-01-04'
	global train_log
	train_log = pd.read_csv('../logs/train.log')
	global test_log
	test_log = pd.read_csv('../logs/test.log')
	global features_log
	features_log = pd.read_csv('../logs/features.log')

def genResult(file, data):
	data.to_csv('../results/'+file+'.csv',sep=',',float_format='%d')
	data.to_csv('../results/'+file+'.txt',sep=',',float_format='%d',index=False,header=False)
	f=open('../results/'+file+'.txt','r')
	context = f.read()
	f.close()
	context = re.sub(',(?=\w{16})','\t',context)
	context = re.sub(',(?=\d+,\d+,\d+)','\t',context)
	f=open('../results/'+file+'.txt','w')
	f.write(context)
	f.close()

def cleanText(contexts):
	f=open('../data/stopwords.txt','r')
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
	    cleans = pd.Series(cleans)
	return cleans

def train(train_data,features,model,label,**parameters):
	global weibo_train_data
	train_context_clean = pd.Series.from_csv('../data/train_context_clean.csv')
	weibo_train_data['context_clean'] = train_context_clean
	if model_type=="LR":
		vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             parameters['max_features']) 
		train_features = vectorizer.fit_transform(                  \
							weibo_train_data[(weibo_train_data['time']<=end) 		\
							& (weibo_train_data['time']>=start)].context_clean)
		train_features = train_features.toarray()
		train_labels = weibo_train_data[(weibo_train_data['time']<=end) 		\
							& (weibo_train_data['time']>=start)][label]
		start = time.time() # Start time
		model = linear_model.LinearRegression()
		model.fit(train_features,train_labels)

		end = time.time()
		elapsed = end - start

		print '====='+feature_type+'_'+model_type+'====='
	# The coefficients
	print 'Coefficients: \n', model.coef_
	# The mean square error
	print "Residual sum of squares: %.2f" % \
		np.mean((model.predict(train_features) - train_labels) ** 2)
	# Explained variance score: 1 is perfect prediction
	print 'Variance score: %.2f' % model.score(train_features, train_labels)
	print "Train time: ", elapsed, "seconds."

	model_name = '../models/'+feature_type+'_'+model_type+'_'+label+'_' \
		+start+'_'+end
	vectorizer_name = '../others/'+feature_type+'_'+model_type+'_' \
		+start+'_'+end
	for k, v in parameters:
		model_name += str(k)+'_'+str(v)
		vectorizer_name += str(k)+'_'+str(v)
	model_name +='.model'
	vectorizer_name += '.vectorizer'
	joblib.dump(model,model_name)
	joblib.dump(vectorizer,vectorizer_name)
	return model,vectorizer

def test(data_start,data_end,model_start,model_end,feature_type,model_type,evaluation=True):
	global weibo_train_data
	global weibo_predict_data
	if data_start>'2014-12-31':
		test_data = weibo_predict_data
		test_data['context_clean'] = pd.Series.from_csv('../data/predict_context_clean.csv')
	else:		
		test_data = weibo_train_data
		test_data['context_clean'] = pd.Series.from_csv('../data/train_context_clean.csv')
	vectorizer = joblib.load('../others/'+feature_type+'_'+model_type+'_' \
		+model_start+'_'+model_end+'.vectorizer')
	test_features = vectorizer.transform(                  \
						test_data[(test_data['time']<=data_end) 		\
						& (test_data['time']>=data_start)].context_clean)
	test_features = test_features.toarray()

	if evaluation == True:
		test_labels = test_data[(test_data['time']<=data_end) 		\
							& (test_data['time']>=data_start)]               \
							[['forward_count','comment_count','like_count']]

	forward_model = joblib.load('../models/'+feature_type+'_'+model_type+'_forward_count_' \
		+model_start+'_'+model_end+'.model')
	comment_model = joblib.load('../models/'+feature_type+'_'+model_type+'_comment_count_' \
		+model_start+'_'+model_end+'.model')
	like_model = joblib.load('../models/'+feature_type+'_'+model_type+'_like_count_' \
		+model_start+'_'+model_end+'.model')

	forward_predict = forward_model.predict(test_features)
	forward_predict[forward_predict<0] = 0
	forward_predict = forward_predict.round()
	comment_predict = comment_model.predict(test_features)
	comment_predict[comment_predict<0] = 0
	comment_predict = comment_predict.round()
	like_predict = like_model.predict(test_features)
	like_predict[like_predict<0] = 0
	like_predict = like_predict.round()

	predict = pd.DataFrame({'forward_predict':forward_predict, \
							'comment_predict':comment_predict, \
							'like_predict':like_predict})

	if evaluation == True: 
		dev_f = (predict.forward_predict-test_labels.forward_count)/(test_labels.forward_count+5)
		dev_c = (predict.comment_predict-test_labels.comment_count)/(test_labels.comment_count+3)
		dev_l = (predict.like_predict-test_labels.like_count)/(test_labels.like_count+3)

		precisions = 1 - 0.5*dev_f - 0.25*dev_c -0.25*dev_l
		count = test_labels.forward_count+test_labels.comment_count+test_labels.like_count
		count[count>100] = 100
		count = count + 1

		precisions_sgn = sgn(precisions)
		precision = (count*precisions_sgn).sum()/count.sum()


		print '====='+feature_type+'_'+model_type+'====='
		print "Forward_count"
		print "Residual sum of squares: %.2f" % \
			np.mean((forward_predict - test_labels.forward_count) ** 2)
		print 'Variance score: %.2f' % forward_model.score(test_features, test_labels.forward_count)
		print "Comment_count"
		print "Residual sum of squares: %.2f" % \
			np.mean((comment_predict - test_labels.comment_count) ** 2)
		print 'Variance score: %.2f' % comment_model.score(test_features, test_labels.comment_count)
		print "Like_count"
		print "Residual sum of squares: %.2f" % \
			np.mean((like_predict - test_labels.like_count) ** 2)
		print 'Variance score: %.2f' % like_model.score(test_features, test_labels.like_count)
		print 'Total_precision:'+str(precision)

	return predict

def sgn(x):
	x[x>0] = 1
	x[x<=0] = 0
	return x

def BOW(data_time=['2014-07-01','2014-12-31'],max_features=100,fit=False):
	global weibo_train_data
	global weibo_predict_data
	global features_log
	if data_time[0]>'2014-12-31':
		data = weibo_predict_data
		data['context_clean'] = pd.Series.from_csv('../data/predict_context_clean.csv')
	else:		
		data = weibo_train_data
		data['context_clean'] = pd.Series.from_csv('../data/train_context_clean.csv')
	if fit==True:
		vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features=max_features) 
		features = vectorizer.fit_transform(                  \
							data[(data['time']<=data[1]) 		\
							& (data['time']>=data[0])].context_clean)
		vectorizer.dump(vectorizer,'../others/'+str(max_features)+'.vectorizer')
	else:
		vectorizer = joblib.load('../others/'+str(max_features)+'.vectorizer'))
		features = vectorizer.transform(                  \
					data[(data['time']<=data_time[1]) 		\
					& (data['time']>=data_time[0])].context_clean)
	features = pd.DataFrame(features)
	feature_name = 'BOW_'+'_'.join(data_time)+'_'+str(max_features)
	features.to_csv('../features/'+ feature_name+'.feature')
	description = "Bag of Words in word count from "+str(data_time[0])+" to "+ \
	data_time[1]+" using top "+str(max_features)+" words"
	features_log.loc[len(features_log)]=[feature_name,description,{'max_features':max_features}]
	features_log.to_csv('../logs/features.log')
return features






if __name__ == "__main__":
	loadData()