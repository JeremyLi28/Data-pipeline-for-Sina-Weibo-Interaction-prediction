# Sina Weibo Interaction-prediction
## Introduction
The Competition's detail can be find [here](http://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.333.11.Til5AJ&raceId=5&_lang=en_US)  
Basically the competition is about analyzing users' behaviors and messages they post on the Chinese micro-blog platform, and predicting the number of forwarding, comment and like on each message.

This project mainly use python and pandas.

The Stage 2 of this competition is still ongoing. Here is the data pipline I built for Stage 1.

## Design
This is a self-designed data pipline. The main thought is **modularizing** the process of a data project. 
* User write methods to generate features, which stored as DataFrame in Pandas in **features** folder, and the **feature.log** will automatically record all existing features and their parameters. 
* User can combine different features and select different models in the **Train** method, the model will be store in **models** folder, the model's information will be stored in **train.log**.
* User choose different combination of features and parameters for testing, the results will be store in *results* folder and the test information will be stored at **test.log**
* Ipython notebooks in **notebooks** folder is for playing around data, watching logs iteratively.
* The code locate in **weiboPredict** package.

When I do data project before, the problems of managing different version of features, models, results and naming them differently are killing me. This simple data pipeline solve my problem, cheers!

## Future Work
Sadly, the second stage of the competition is on Alibaba's ODPS platform with SQL and java, I don't have the chance to develop this framework further right now. The pipeline is still a little problem-specific and I want to build it for more general purpose in the furture.

