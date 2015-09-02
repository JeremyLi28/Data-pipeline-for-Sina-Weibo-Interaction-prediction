# Name: framework.py
# Usage: basic operations
# Author: Chen Li
import pandas as pd
import numpy as np
import csv
import re
import jieba
import time
import json
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
	train_log = pd.DataFrame.from_csv('../logs/train.log')
	train_log.features = train_log.features.map(lambda x: json.loads(x))
	train_log.model_parameters = train_log.model_parameters.map(lambda x: json.loads(x))
	train_log.evaluation = train_log.evaluation.map(lambda x: json.loads(x))
	global test_log
	test_log = pd.DataFrame.from_csv('../logs/test.log')
	global features_log
	features_log = pd.DataFrame.from_csv('../logs/features.log')
	features_log.data_time = features_log.data_time.map(lambda x: json.loads(x))
	features_log.parameters = features_log.parameters.map(lambda x:json.loads(x))

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

def train(features,model_type,label,**model_parameters):

	# load features
	print "loading features..."
	global features_log
	train_features = pd.DataFrame.from_csv(features_log[features_log.feature_name==features[0]].feature_address.tolist()[0])
	if len(features) > 1:
		for i in range(1,len(features)):
			tmp = pd.DataFrame.from_csv(features_log[features_log.feature_name==features[i]].feature_address.tolist()[0])
			train_features = pd.concat(train_features,tmp,axis=1)

	# load label
	print "loading label..."
	global weibo_train_data
	train_labels = weibo_train_data[(weibo_train_data['time']<=features_log[features_log.feature_name==features[0]].data_time.tolist()[0][1]) \
									& (weibo_train_data['time']>=features_log[features_log.feature_name==features[0]].data_time.tolist()[0][0])][label]

	# train model
	print "training model..."
	if model_type=="LR":
		start = time.time() # Start time
		model = linear_model.LinearRegression()
		model.fit(train_features,train_labels)
		end = time.time()
		elapsed = end - start

	# write log
	print "writing log..."
	coef = model.coef_
	sos = np.mean((model.predict(train_features) - train_labels) ** 2)
	vs = model.score(train_features, train_labels)
	model_name = '_'.join(features.tolist())+'_'+model_type+'_'
	for k, v in model_parameters:
		model_name += str(k)+'_'+str(v)
	model_name += label	
	model_address ='../models/'+model_name+'.model'
	log = [model_name,json.dumps(features.tolist()),model_type,label,json.dumps(model_parameters), \
						json.dumps({'sos':sos,'vs':vs}),model_address,elapsed]
	if model_name in train_log.model_name.tolist():
		train_log[train_log.model_name==model_name] = log
	else:
		train_log.loc[len(train_log)] = log
	train_log.to_csv('../logs/train.log')

	# save model
	print "saving model..."
	joblib.dump(model,model_address)

	# print results
	print '====='+'Results'+'====='
	print 'Coefficients: \n', coef
	print "Residual sum of squares: %.2f" % sos
	print 'Variance score: %.2f' % vs
	print "Train time: ", elapsed, "seconds."

	return model

def test(features,fmodel,cmodel,lmodel,evaluation=True):

	test_name = '_'.join(features.tolist())+'_'+fmodel+'_'+cmodel+'_'+lmodel
	# load features
	print "loading features..."
	global features_log
	test_features = pd.DataFrame.from_csv(features_log[features_log.feature_name==features[0]].feature_address.tolist()[0])
	if len(features) > 1:
		for i in range(1,len(features)):
			tmp = pd.DataFrame.from_csv(features_log[features_log.feature_name==features[i]].feature_address.tolist()[0])
			test_features = pd.concat(test_features,tmp,axis=1)

	if evaluation == True:
		print "loading labels..."
		global weibo_train_data
		test_labels = weibo_train_data[(weibo_train_data['time']<=features_log[features_log.feature_name==features[0]].data_time.tolist()[0][1]) \
						& (weibo_train_data['time']>=features_log[features_log.feature_name==features[0]].data_time.tolist()[0][0])] \
						['forward_count','comment_count','like_count']

	print "loading models..."
	global train_log
	forward_model = joblib.load(train_log[train_log.model_name==fmodel].model_address.tolist()[0])
	comment_model = joblib.load(train_log[train_log.model_name==cmodel].model_address.tolist()[0])
	like_model = joblib.load(train_log[train_log.model_name==lmodel].model_address.tolist()[0])

	print "predict..."
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
		print 'evaluating...'
		dev_f = (predict.forward_predict-test_labels.forward_count)/(test_labels.forward_count+5)
		dev_c = (predict.comment_predict-test_labels.comment_count)/(test_labels.comment_count+3)
		dev_l = (predict.like_predict-test_labels.like_count)/(test_labels.like_count+3)

		precisions = 1 - 0.5*dev_f - 0.25*dev_c -0.25*dev_l
		count = test_labels.forward_count+test_labels.comment_count+test_labels.like_count
		count[count>100] = 100
		count = count + 1

		precisions_sgn = sgn(precisions)
		precision = (count*precisions_sgn).sum()/count.sum()


		fsos = np.mean((forward_predict - test_labels.forward_count) ** 2)
		fvs = forward_model.score(test_features, test_labels.forward_count)
		csos = np.mean((comment_predict - test_labels.comment_count) ** 2)
		cvs = comment_model.score(test_features, test_labels.comment_count)
		lsos = np.mean((like_predict - test_labels.like_count) ** 2)
		lvs = like_model.score(test_features, test_labels.like_count)

		print "writing logs..."
		global test_log
		log = [test_name,features.tolist(),fmodel,cmodel,lmodel,0.5*dev_f.mean(),0.25*dev_c.mean(),0.25*dev_l.mean(),precision, \
				json.dumps({'fsos':fsos,'fvs':fvs,'csos':csos,'cvs':cvs,'lsos':lsos,'lvs':lvs})]
		if test_name in test_log.test_name.tolist():
			test_log[test_log.test_name==test_name] = log
		else:
			test_log.loc[len(test_log)] = log
		test_log.to_csv('../logs/test.log')


		print '====='+'Results'+'====='
		print "-----Forward_count-----"
		print "Residual sum of squares: %.2f" % fsos		
		print 'Variance score: %.2f' % fvs
		print "-----Comment_count-----"
		print "Residual sum of squares: %.2f" % csos	
		print 'Variance score: %.2f' % cvs
		print "-----Like_count-----"
		print "Residual sum of squares: %.2f" % lsos
		print 'Variance score: %.2f' % lvs
		print '------Total------'
		print "dev_f:%f, dev_c:%f, dev_l:%f" % 0.5*dev_f.mean(), 0.25*dev_c.mean, 0.25*dev_l.mean()
		print 'Total_precision:'+str(precision)
	else:
		print "genelizing results..."
		global weibo_predict_data
		genResult(test_name,pd.concat(weibo_predict_data[['uid','mid']],predict))

	return predict

def sgn(x):
	x[x>0] = 1
	x[x<=0] = 0
	return x

def BOW(data_time=['2014-07-01','2014-12-31'],vec_time=['2014-07-01','2014-12-31'],max_features=100,fit=False):
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
		data_time = vec_time
		vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features=max_features) 
		features = vectorizer.fit_transform(                  \
							data[(data['time']<=data_time[1]) 		\
							& (data['time']>=data_time[0])].context_clean)
		joblib.dump(vectorizer,'../others/'+'_'.join(vec_time)+'_'+str(max_features)+'.vectorizer')
	else:
		vectorizer = joblib.load('../others/'+'_'.join(vec_time)+'_'+str(max_features)+'.vectorizer')
		features = vectorizer.transform(                  \
					data[(data['time']<=data_time[1]) 		\
					& (data['time']>=data_time[0])].context_clean)
	columns = ['BOW_'+str(i+1) for i in range(max_features)]
	features = pd.DataFrame(features.toarray(),columns=columns)
	# write log
	feature_name = 'BOW_'+'_'.join(data_time)+'_'+'_'.join(vec_time)+'_'+str(max_features)
	feature_address = '../features/'+ feature_name+'.feature'
	features.to_csv(feature_address)
	description = "Bag of Words in word count from "+str(data_time[0])+" to "+ \
	data_time[1]+" using top "+str(max_features)+" words"

	log = [feature_name,'BOW',json.dumps(data_time),json.dumps({'max_features':max_features,'vec_time':vec_time}), \
			'I',feature_address,description]
	if feature_name in features_log.feature_name.tolist():
		features_log[features_log.feature_name==feature_name] = log
	else:
		features_log.loc[len(features_log)] = log
	features_log.to_csv('../logs/features.log')

	return features






if __name__ == "__main__":
	loadData()