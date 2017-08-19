****

#	<center>Zhihu Machine Learning Challenge 2017</center>


****

##	Categories
*	[Abstract](#Abstract)
* 	[Learning To Rank](#learn-to-rank)

****

##	<a name="Abstract"> Abstract </a>

In the **Zhihu Machine Learning Challenge 2017**, we were asked to build a model to automaticly and accurately tag topics for Zhihu contents. Our final submission was a 2-stage process and scored **0.43436** on Public LB and **0.43273** on Private LB, **ranking 3rd** out of all teams. This documents describes our team's solution which can be dived into two parts:

1. Deep Learning: build variance DL models to sort all topics.
2. Learning To Rank: build RankGBM model to sort ten of most possible topics.

****

## <a name="learn-to-rank"> Learning To Rank </a>

In the first stage, we can get the DL model prediction results for each <instance, topic> pairs. In the second stage, we will vote for all instances based on ML model results. After voting, each instance is associated with ten of the most likely topics. Then, build a RankGBM model to sort ten of most possible topics.

The above description can be done by the following steps:

1. Enter root directory of the project:
	
	```shell
	cd zhihu-machine-learning-challenge-2017/
	```

2. Vote for offline dataset and online dataset:

	```shell
	python -m bin.rank.vote conf/rank_v29.conf vote offline
	python -m bin.rank.vote conf/rank_v29.conf vote online
	```

3. Generate features for offline dataset and online dataset:

		# generate <instance, topic> pair features
		python -m bin.rank.feature conf/rank_v29.conf generate_featwheel_feature_from_model offline
		python -m bin.rank.feature conf/rank_v29.conf generate_featwheel_feature_from_model online
		# generate <instance> features
		python -m bin.rank.feature conf/rank_v29.conf generate_featwheel_feature_from_instance offline
		python -m bin.rank.feature conf/rank_v29.conf generate_featwheel_feature_from_instance online
		# generate <topic> features
		python -m bin.rank.feature conf/rank_v29.conf generate_featwheel_feature_from_topic offline
		python -m bin.rank.feature conf/rank_v29.conf generate_featwheel_feature_from_topic online
