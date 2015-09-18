 # -*- coding: utf-8 -*-
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
from nltk.corpus import stopwords as e_stopwords
from datetime import datetime, timedelta

weibo_train_data = None
weibo_predict_data = None
train_log = None
test_log = None
features_log = None

def loadData():
	global weibo_train_data 
	weibo_train_data= pd.read_csv('../data/weibo_train_data(new).txt',sep='\t', 
		quoting=csv.QUOTE_NONE,names=['uid','mid','time','forward_count','comment_count','like_count','context'])
	weibo_train_data.time = weibo_train_data.time.apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
	global weibo_predict_data 
	weibo_predict_data = pd.read_csv('../data/weibo_predict_data(new).txt',sep='\t',
		quoting=csv.QUOTE_NONE,names=['uid','mid','time','context'])
	weibo_predict_data.time = weibo_predict_data.time.apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
	#weibo_predict_data.ix[78336].time = '2015-01-04'
	global train_log
	train_log = pd.DataFrame.from_csv('../logs/train.log')
	train_log.features = train_log.features.map(lambda x: json.loads(x))
	train_log.model_parameters = train_log.model_parameters.map(lambda x: json.loads(x))
	train_log.evaluation = train_log.evaluation.map(lambda x: json.loads(x))
	global test_log
	test_log = pd.DataFrame.from_csv('../logs/test.log')
	test_log.f_features = test_log.f_features.map(lambda x: json.loads(x))
	test_log.c_features = test_log.c_features.map(lambda x: json.loads(x))
	test_log.l_features = test_log.l_features.map(lambda x: json.loads(x))
	test_log.model_evaluation = test_log.model_evaluation.map(lambda x: json.loads(x))
	global features_log
	features_log = pd.DataFrame.from_csv('../logs/features.log')
	features_log.data_time = features_log.data_time.map(lambda x: json.loads(x))
	features_log.parameters = features_log.parameters.map(lambda x:json.loads(x))
	features_log.feature_shape = features_log.feature_shape.map(lambda x:json.loads(x))

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
	    # remove links
	    context = re.sub("http://[a-zA-z./\d]*","",context)
	    # remove emojo
	    context = re.sub("\[.{0,12}\]","",context)
	    # extract and remove tag
	    tags = re.findall("#(.{0,30})#",context)
	    context = re.sub("#.{0,30}#","",context)
	    # extract and remove @somebody
	    at = re.findall("@([^@]{0,30})\s",context)
	    context = re.sub("@([^@]{0,30})\s","",context)
	    at+= re.findall("@([^@]{0,30})）",context)
	    context = re.sub("@([^@]{0,30})）","",context)
	    # lower the english characaters
	    context  = context.lower()
	    # extract and remove the english words and characters
	    english = re.findall("[a-z]+",context)
	    context = re.sub("[a-z]+","",context)
	    # remove punctuation
	    context = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), "",context)
	    context = re.sub("[【】╮╯▽╰╭★→「」]+".decode("utf8"),"",context.decode('utf8'))
	    # remove wirte space
	    context = re.sub("\s","",context)
	    # remove digits
	    context = re.sub("\d","",context)
	    # remove ....
	    context = re.sub("\.*","",context)
	    # word segementation
	    text = jieba.lcut(context)
	    # remove chinese stopwords
	    clean = [t for t in text if t not in stopwords]
	    # remove english stopwords and singel characters
	    e_clean = [t for t in english if t not in e_stopwords.words('english') and len(t) is not 1]
	    clean+=e_clean
	    clean+=pd.Series(tags).apply(lambda x:x.decode('utf8')).tolist()
	    clean+=pd.Series(at).apply(lambda x:x.decode('utf8')).tolist()
	    cleans.append(clean)
	    i=i+1
	    if i%10000==0:
	    	print str(i)+'/'+str(len(contexts))
	cleans = pd.Series(cleans)
	return cleans

def train(features,model_type,label,**model_parameters):

	# load features
	print "loading features..."
	train_features = loadFeatures(features)

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
	model_name = '_'.join(features)+'_'+model_type+'_'
	for k, v in model_parameters:
		model_name += str(k)+'_'+str(v)
	model_name += label	
	model_address ='../models/'+model_name+'.model'
	category = features_log[features_log.feature_name==features[0]].category.tolist()[0]
	log = [model_name,features,model_type,label,model_parameters,category,{'sos':sos,'vs':vs},model_address,elapsed]
	writeLog(log,"train_log")

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

def test(f_features, c_features, l_features, f_model, c_model, l_model, evaluation=True):

	test_name = '_'.join(f_features)+'_'.join(c_features)+'_'.join(l_features)+'_'+f_model+'_'+c_model+'_'+l_model
	# load features
	print "loading features..."
	f_feature = loadFeatures(f_features)
	c_feature = loadFeatures(c_features)
	l_feature = loadFeatures(l_features)
	


	if evaluation == True:
		print "loading labels..."
		global weibo_train_data
		test_labels = weibo_train_data[(weibo_train_data['time']<=features_log[features_log.feature_name==f_features[0]].data_time.tolist()[0][1]) \
						& (weibo_train_data['time']>=features_log[features_log.feature_name==f_features[0]].data_time.tolist()[0][0])] \
						[['forward_count','comment_count','like_count']]

	print "loading models..."
	global train_log
	forward_model = joblib.load(train_log[train_log.model_name==f_model].model_address.tolist()[0])
	comment_model = joblib.load(train_log[train_log.model_name==c_model].model_address.tolist()[0])
	like_model = joblib.load(train_log[train_log.model_name==l_model].model_address.tolist()[0])

	print "predicting..."
	forward_predict = forward_model.predict(f_feature)
	forward_predict[forward_predict<0] = 0
	forward_predict = forward_predict.round()
	comment_predict = comment_model.predict(c_feature)
	comment_predict[comment_predict<0] = 0
	comment_predict = comment_predict.round()
	like_predict = like_model.predict(l_feature)
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
		fvs = forward_model.score(f_feature, test_labels.forward_count)
		csos = np.mean((comment_predict - test_labels.comment_count) ** 2)
		cvs = comment_model.score(c_feature, test_labels.comment_count)
		lsos = np.mean((like_predict - test_labels.like_count) ** 2)
		lvs = like_model.score(l_feature, test_labels.like_count)

		print "writing logs..."
		global test_log
		log = [test_name,f_features,c_features,l_features,f_model,c_model,l_model,0.5*dev_f.mean(),0.25*dev_c.mean(),0.25*dev_l.mean(),precision, \
				{'fsos':fsos,'fvs':fvs,'csos':csos,'cvs':cvs,'lsos':lsos,'lvs':lvs},'']
		writeLog(log,"test_log")


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
		print "dev_f:%f, dev_c:%f, dev_l:%f" % (0.5*dev_f.mean(), 0.25*dev_c.mean(), 0.25*dev_l.mean())
		print 'Total_precision:'+str(precision)
	else:
		print "writing logs..."
		global test_log
		result_name = 'result_'+str(datetime.now())
		log = [test_name,f_features,c_features,l_features,f_model,c_model,l_model,0,0,0,0,{},'../results/'+result_name+'.txt']
		writeLog(log,"test_log")
		print "genelizing results..."
		global weibo_predict_data
		genResult(result_name,pd.concat([weibo_predict_data[['uid','mid']],predict],axis=1))

	return predict

def sgn(x):
	x[x>0] = 1
	x[x<=0] = 0
	return x

def writeLog(log,log_type):
	global train_log
	global test_log
	global features_log
	if log_type == "train_log":
		if log[0] in train_log.model_name.tolist():
			train_log.loc[train_log[train_log.model_name==log[0]].index.values[0]] = log
		else:
			train_log.loc[len(train_log)] = log
		tmp_train_log = train_log.copy()
		tmp_train_log.features = tmp_train_log.features.map(lambda x: json.dumps(x))
		tmp_train_log.model_parameters = tmp_train_log.model_parameters.map(lambda x: json.dumps(x))
		tmp_train_log.evaluation = tmp_train_log.evaluation.map(lambda x: json.dumps(x))
		tmp_train_log.to_csv('../logs/train.log')
	elif log_type == "test_log":
		if log[0] in test_log.test_name.tolist():
			test_log.loc[test_log[test_log.test_name==log[0]].index.values[0]] = log
		else:
			test_log.loc[len(test_log)] = log
		tmp_test_log = test_log.copy()
		tmp_test_log.f_features = tmp_test_log.f_features.map(lambda x: json.dumps(x))
		tmp_test_log.c_features = tmp_test_log.c_features.map(lambda x: json.dumps(x))
		tmp_test_log.l_features = tmp_test_log.l_features.map(lambda x: json.dumps(x))
		tmp_test_log.model_evaluation = tmp_test_log.model_evaluation.map(lambda x: json.dumps(x))
		tmp_test_log.to_csv('../logs/test.log')
	elif log_type=="features_log":
		if log[0] in features_log.feature_name.tolist():
			features_log.loc[features_log[features_log.feature_name==log[0]].index.values[0]] = log
		else:
			features_log.loc[len(features_log)] = log
		tmp_features_log = features_log.copy()
		tmp_features_log.data_time = tmp_features_log.data_time.map(lambda x: json.dumps(x))
		tmp_features_log.parameters = tmp_features_log.parameters.map(lambda x:json.dumps(x))
		tmp_features_log.feature_shape = tmp_features_log.feature_shape.map(lambda x:json.dumps(x))
		tmp_features_log.to_csv('../logs/features.log')

def loadFeatures(feature_list):
	global features_log
	features = pd.DataFrame.from_csv(features_log[features_log.feature_name==feature_list[0]].feature_address.tolist()[0])
	if len(feature_list) > 1:
		for i in range(1,len(feature_list)):
			tmp = pd.DataFrame.from_csv(features_log[features_log.feature_name==feature_list[i]].feature_address.tolist()[0])
			features = pd.concat([features,tmp],axis=1)
	return features





def I_BOW(data_time=['2014-07-01','2014-12-31'],vec_time=['2014-07-01','2014-12-31'],max_features=100,fit=False):
	global weibo_train_data
	global weibo_predict_data
	global features_log
	print "loading data..."
	if data_time[0]>'2014-12-31':
		data = weibo_predict_data.copy()
		data['context_clean'] = pd.Series.from_csv('../data/predict_context_clean.csv')
	else:		
		data = weibo_train_data.copy()
		data['context_clean'] = pd.Series.from_csv('../data/train_context_clean.csv')
	data.context_clean = data.context_clean.apply(lambda x: json.loads(x))
	data.context_clean = data.context_clean.apply(lambda x: ' '.join(x))
	if fit==True:
		print 'fitting and transforming...'
		data_time = vec_time
		vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features=max_features)
		features = vectorizer.fit_transform(                  \
							data[(data['time']<=data_time[1]) 		\
							& (data['time']>=data_time[0])].context_clean)
		print 'saving models...'
		joblib.dump(vectorizer,'../others/'+'_'.join(vec_time)+'_'+str(max_features)+'.vectorizer')
	else:
		print 'transforming...'
		vectorizer = joblib.load('../others/'+'_'.join(vec_time)+'_'+str(max_features)+'.vectorizer')
		features = vectorizer.transform(                  \
					data[(data['time']<=data_time[1]) 		\
					& (data['time']>=data_time[0])].context_clean)
	columns = ['I_BOW_'+str(i+1) for i in range(max_features)]
	features = pd.DataFrame(features.toarray(),columns=columns)
	# write log
	print 'saving features...'
	feature_name = 'I_BOW_'+'_'.join(data_time)+'_'+'_'.join(vec_time)+'_'+str(max_features)
	feature_address = '../features/'+ feature_name+'.feature'
	features.to_csv(feature_address)
	usage = "train" if fit==True else "test"
	description = "Bag of Words in word count from "+str(data_time[0])+" to "+ \
	data_time[1]+" using top "+str(max_features)+" words"

	print "writing logs..."
	log = [feature_name,'I_BOW',data_time,{'max_features':max_features,'vec_time':vec_time},'I',feature_address,usage,description,list(features.values.shape)]
	writeLog(log,"features_log")

	return features

def U_AVG(train_time=['2015-02-01 00:00:00','2015-06-30 23:59:59'],test_time =['2015-07-01 00:00:00','2015-07-31 23:59:59'],time_range="All"):
	global weibo_train_data
	global weibo_predict_data
	global features_log

	# data = weibo_train_data
	data = pd.concat([weibo_train_data,weibo_predict_data])
	train_data = data[(data['time']>=train_time[0]) & (data['time']<=train_time[1])]
	test_data = data[(data['time']>=test_time[0]) & (data['time']<=test_time[1])]

	ug = train_data.groupby('uid')
	uavg = ug.sum()/ug.count()[['forward_count','comment_count','like_count']]
	uavg=uavg.applymap(lambda x:round(x))
	uavg.columns = ['U_AVG_f'+'_'+'_'.join(train_time)+'_'.join(train_time),'U_AVG_c'+'_'+'_'.join(train_time)+'_'.join(train_time), \
					'U_AVG_l'+'_'+'_'.join(train_time)+'_'.join(train_time)]

	# train features
	train_features = pd.merge(train_data,uavg,how="inner",left_on='uid',right_index=True)
	train_features = train_features[['U_AVG_f'+'_'+'_'.join(train_time)+'_'.join(train_time),'U_AVG_c'+'_'+'_'.join(train_time)+'_'.join(train_time), \
					'U_AVG_l'+'_'+'_'.join(train_time)+'_'.join(train_time)]]
	train_features.index = range(len(train_features))
	train_feature_name = "U_AVG"+'_'+'_'.join(train_time)+'_'.join(train_time)
	train_feature_address = '../features/'+train_feature_name+'.feature'
	train_features.to_csv(train_feature_address)
	train_description = "User's average forward/comment/like count during"+'-'.join(train_time)
	train_log = [train_feature_name,'U_AVG',train_time,{},'U',train_feature_address,"train",train_description,list(train_features.values.shape)]
	writeLog(train_log,"features_log")

	# test features
	uavg.columns = ['U_AVG_f'+'_'+'_'.join(train_time)+'_'.join(test_time),'U_AVG_c'+'_'+'_'.join(train_time)+'_'.join(test_time), \
					'U_AVG_l'+'_'+'_'.join(train_time)+'_'.join(test_time)]
	test_features = pd.merge(test_data,uavg,left_on='uid',right_index=True,how="left")
	test_features['U_AVG_f'+'_'+'_'.join(train_time)+'_'.join(test_time)].fillna(uavg['U_AVG_f'+'_'+'_'.join(train_time)+'_'.join(test_time)].mean(),inplace=True)
	test_features['U_AVG_c'+'_'+'_'.join(train_time)+'_'.join(test_time)].fillna(uavg['U_AVG_c'+'_'+'_'.join(train_time)+'_'.join(test_time)].mean(),inplace=True)
	test_features['U_AVG_l'+'_'+'_'.join(train_time)+'_'.join(test_time)].fillna(uavg['U_AVG_l'+'_'+'_'.join(train_time)+'_'.join(test_time)].mean(),inplace=True)
	test_features = test_features[['U_AVG_f'+'_'+'_'.join(train_time)+'_'.join(test_time),'U_AVG_c'+'_'+'_'.join(train_time)+'_'.join(test_time), \
					'U_AVG_l'+'_'+'_'.join(train_time)+'_'.join(test_time)]]
	test_features.index = range(len(test_features))
	test_feature_name = "U_AVG"+'_'+'_'.join(train_time)+'_'.join(test_time)
	test_feature_address = '../features/'+test_feature_name+'.feature'
	test_features.to_csv(test_feature_address)
	test_description = "User's average forward/comment/like count during"+'-'.join(test_time)
	test_log = [test_feature_name,'U_AVG',test_time,{},'U',test_feature_address,"test",test_description,list(test_features.values.shape)]
	writeLog(test_log,"features_log")
	return train_features, test_features

# def I_WAVG(train_time, test_time, top):
# 	global weibo_train_data
# 	global weibo_predict_data
# 	global features_log
# 	print "loading data..."
# 	if data_time[0]>'2014-12-31':
# 		data = weibo_predict_data.copy()
# 		data['context_clean'] = pd.Series.from_csv('../data/predict_context_clean.csv')
# 	else:		
# 		data = weibo_train_data.copy()
# 		data['context_clean'] = pd.Series.from_csv('../data/train_context_clean.csv')
# 	data.context_clean = data.context_clean.apply(lambda x: json.loads(x))

# def UI_WEIBO_COUNT(train_time, test_time, direction=1, day=0):
# 	global weibo_train_data
# 	global weibo_predict_data
# 	global features_log

# 	data = pd.concat([weibo_train_data[['uid','mid','time']],weibo_predict_data[['uid','mid','time']]])
# 	data.set_index('time',inplace=True)
# 	train_data = data[train_time[0]:train_time[1]]
# 	test_data = data[test_time[0]):test_time[1]]









if __name__ == "__main__":
	loadData()