## 安然提交开放式问题


1.向我们总结此项目的目标以及机器学习对于实现此目标有何帮助。作为答案的部分，提供一些数据背景信息以及这些信息如何用于回答项目问题。你在获得数据时它们是否包含异常值，你是如何处理的？


* 项目目标：通过机器学习识别并提取有用特征，构建算法，通过公开的安然财务和邮件数据集，找出有欺诈嫌疑的安然雇员.
* 理解数据集和问题：
	- 数据点总数：该数据一共包含146个数据
	
	```
	print("Before Clean:", len(data_dict))
	
	>>>
	Before Clean: 146
	```
	
	- 类之间的分配：数据集中一共18名人员被标记为POI，128名人员不是POI。
	```
	poi_list = []
	not_poi_list = []
	for name in data_dict:
    if data_dict[name]["poi"]:
        poi_list.append(name)
    else:
        not_poi_list.append(name)
	print("There are", len(poi_list),"poi in this dataset")
	print("There are", len(not_poi_list), "Non-poi in this dataset")
	
	>>>
	There are 18 poi in this dataset
	There are 128 Non-poi in this dataset
	```
	
	- 使用的特征数量：数据集中包含安然员工的财务以及邮件数据共有20个特征。
	```
	all_features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                     'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                     'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                     'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',
                     'from_this_person_to_poi', 'shared_receipt_with_poi','email_address']
	```
	
	- 异常值:有些异常值是由于报表造成, 如"TOTAL","THE TRAVEL AGENCY IN THE PARK"; 有的则是由于所有特征全部为"NaN",如"LOCKHART EUGENE E", 无有用信息，可删除。
	
	- 异常值处理：对于因报表，将其找出并去掉，而对于数据缺失的情况，为了保证数据集的数据量，选择将“NaN”转成0，而不是删去；而正常数据则将其保留，并需要加以关注.
	```
	data_dict.pop("TOTAL", 0)
	data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
	data_dict.pop("LOCKHART EUGENE E", 0)
	```

***


2.你最终在你的POI标识符中使用了什么特征，你使用了什么筛选过程来挑选他们？你是否需要进行任何缩放？为什么？作为任务的一部分，你应该设计自己的特征，而非使用数据集中现成的--解释你尝试创建的特征及其基本原理。（你不一定要在最后的分析中使用它，而只是设计并测试它）。在你的特征选择步骤，如果你使用了算法（如决策树），请也给出所使用的特征重要性；如果你使用了自动特征选择函数（如SelectBest)，请在报告特征得分及你所选的参数的原因

* 最终选择了"bonus", "exercised_stock_options", "total_stock_value"这三个特征。
* 使用SelectKBest()方法来初步筛选，然后自己带入去验证，最后选取了评分最高的特征
* 尝试进行了特征缩放，因为我选择的算法中KNN算法与距离相关，会受到缩放影响。
* 调查员工奖金与薪水比，创建了新特征bonus_salary = bonus / salary

* 优化特征选择过程:
	- 使用SelectKBest，DecisionTree初步筛选：
	```
	selector = SelectKBest()
	selector.fit_transform(features, labels)
	
	DT_selector = DecisionTreeClassifier()
	DT_selector.fit(features, labels)
	
	features_scores = {feature : score for feature, score in zip(all_features_list[1:], DT_selector.feature_importances_)}
	sorted_features_scores = sorted(features_scores.items(), key=lambda a:a[1], reverse=True)
	pprint.pprint(sorted_features_scores[:5])
	
	>>> Score排序：
	[('exercised_stock_options', 25.097541528735491),
	 ('total_stock_value', 24.467654047526391),
	 ('bonus', 21.060001707536578),
	 ('salary', 18.575703268041778),
	 ('deferred_income', 11.595547659732164)]
	
	>>> DecisionTreeClassifier的feature_importance排序：
	[('exercised_stock_options', 0.19984012789768188),
	 ('bonus', 0.19387255448166713),
	 ('restricted_stock', 0.12054494260342233),
	 ('total_payments', 0.11298765432098762),
	 ('long_term_incentive', 0.10895238095238094)]
	```
	
	- 选取Score前三的特征进行Test, 测试使用了KNeighborsClassifier, Decision Tree, Nearest Neighbor, RandomForest, AdaBoost五种算法, 结果几个算法都还理想，KNN的Recall稍低。继续从这五个特征中选取特征测试，结果相差不大。
	```
	names = ["Naive Bayes", "Decision Tree", "Nearest Neighbor", "Random Forest", "AdaBoost"]
	classifiers = [GaussianNB(),
               DecisionTreeClassifier(),
               KNeighborsClassifier(n_neighbors=3),
               RandomForestClassifier(),
               AdaBoostClassifier(base_estimator=DecisionTreeClassifier())]
	
	for name, clf in zip(names, classifiers):
		print("feature:", my_features)
		test_classifier(clf, my_dataset, my_features, folds=1000)
	
	>>>
	feature: ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus']
	GaussianNB(priors=None) Precision: 0.48581	Recall: 0.35100
	DecisionTreeClassifier() Precision: 0.36144	Recall: 0.38150
	KNeighborsClassifier(n_neighbors=3) Precision: 0.61627	Recall: 0.29550
	RandomForestClassifier() Precision: 0.60529	Recall: 0.30900
	AdaBoostClassifier(base_estimator=DecisionTreeClassifier()) Precision: 0.36682	Recall: 0.38700
	```
	
	- 特征缩放. 特征缩放适用于与距离相关的算法，因为选取了KNN算法, 故尝试使用特征缩放。
	
	```
	scaler = MinMaxScaler()
	data = scaler.fit_transform(data)
	
	>>> KNN算法结果Precision稍降，Recall稍有提高
	feature: ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus']
	KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=3, p=2,
           weights='uniform')
	Accuracy: 0.86177	Precision: 0.60181	Recall: 0.30000	F1: 0.40040	F2: 0.33344
	Total predictions: 13000	True positives:  600	False positives:  397	False negatives: 1400	True negatives: 10603
	```
	
	- 新特征的创建与实现：
		- 新特征bonus_salary：bonus和salary之比
		```
		# 调查员工的奖金与薪水比，创建新特征bonus_salary
		for name in data_dict:
			try:
				data_dict[name]["bonus_salary"] = data_dict[name]["bonus"] / data_dict[name]["salary"]
			except:
				data_dict[name]["bonus_salary"] = 0
		```
		
		- 创建理由：a.bonus和salary在之前测试发现得分都较高。b.猜测poi的bonus和salary可能与非poi会有一些异常
		- 测试新特征, 结果看起来不理想，排序靠后：
		```
		# SelcectKBest和feature_importance测试
		>>> SelectKBest 得分排名第6
		[('exercised_stock_options', 24.815079733218194),
		 ('total_stock_value', 24.182898678566872),
		 ('bonus', 20.792252047181538),
		 ('salary', 18.289684043404513),
		 ('deferred_income', 11.458476579280697),
		 ('bonus_salary', 10.78358470816082),
		 ('long_term_incentive', 9.9221860131898385),
		 ('restricted_stock', 9.212810621977086),
		 ('total_payments', 8.7727777300916809),
		 ('shared_receipt_with_poi', 8.5894207316823774)]
		 
		>>> feature_importances_ 排名第10
		[('exercised_stock_options', 0.19984012789768188),
		 ('restricted_stock', 0.12054494260342231),
		 ('total_payments', 0.1129876543209876),
		 ('bonus', 0.10913181374092638),
		 ('long_term_incentive', 0.10895238095238093),
		 ('from_this_person_to_poi', 0.065122641509433823),
		 ('expenses', 0.064896989949621592),
		 ('from_poi_to_this_person', 0.055611111111111097),
		 ('deferred_income', 0.052962962962962955),
		 ('bonus_salary', 0.048148148148148134)]
		```
		
		- 将新特征加入测试, 并调整特征, 结果与之前比, 部分算法得分提高了, 但整体结果并不比之前好，因此不打算采用作为最终特征:
		```
		feature: ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus_salary']
		GaussianNB(priors=None) Precision: 0.41945	Recall: 0.28900
		DecisionTreeClassifier() Precision: 0.37718	Recall: 0.36850
		KNeighborsClassifier(n_neighbors=3) Precision: 0.71083	Recall: 0.30850
		RandomForestClassifier() Precision: 0.52778	Recall: 0.21850
		AdaBoostClassifier(base_estimator=DecisionTreeClassifier()) Precision: 0.38073	Recall: 0.36550
		```

***

3.你最终选用了什么算法？你还尝试了其他什么算法？不同算法之间的模型性能有何差异？

* 我最终选择了Naive Bayes算法。
* 还尝试了Decision Tree,Nearest Neighbor(KNN), RandomForest, AdaBoost。
* 对于所选的特征，Naive bayes综合Precision，Recall最好，决策树其次, KNN 的Precision最高，但Recall偏低 

***

4.调整算法的参数是什么意思，如果你不这样做会发生什么？你是如何调整特定算法的参数的?（一些算法没有需要调整的参数，指明并简要解释对于你最终为选择的模型或需要调整的不同模型，例如决策树分类器，你会怎么做）。

* 对于每个算法，基于它们不同的作用原理，它们有相对应的参数，这些参数决定了该算法处理特征的性能，为了获得算法的最佳性能，我们需要调整算法的参数。
* 如果不调整参数，尽管我们可能选择了合适的算法，也无法得到满意的结果。
* 在项目中使用了分别用了GridSearchCV()，手动调参来调整参数：
	- GridSearchCV调参，将得到的最佳参数导入Test，发现结果不如人意。
	```
	names = ["Naive Bayes", "Decision Tree", "Nearest Neighbor", "Random Forest", "AdaBoost"]
	classifiers = [GaussianNB(),
				   DecisionTreeClassifier(),
				   KNeighborsClassifier(),
				   RandomForestClassifier(),
				   AdaBoostClassifier()]
	parameters = {"Naive Bayes":{},
				  "Decision Tree":{"max_depth": range(5,15),
								   "min_samples_leaf": range(1,5)},
				  "Nearest Neighbor":{"n_neighbors": range(1, 10),
									  "weights":("uniform", "distance"),
									  "algorithm":("auto", "ball_tree", "kd_tree", "brute")},
				  "Random Forest":{"n_estimators": range(2, 5),
								   "min_samples_split": range(2, 5),
								   "max_depth": range(2, 15),
								   "min_samples_leaf": range(1, 5),
								   "random_state": [0, 10, 23, 36, 42],
								   "criterion": ["entropy", "gini"]},
				  "AdaBoost":{"n_estimators": range(2, 5),
							  "algorithm":("SAMME", "SAMME.R"),
							  "random_state":[0, 10, 23, 36, 42]}}

	print("feature:", my_features)
	for name, clf in zip(names, classifiers):
		grid_clf = GridSearchCV(clf, parameters[name])
		grid_clf.fit(features_train, labels_train)
		pprint.pprint(grid_clf.best_params_)

	>>> 获得的最佳匹配的参数：
	{}
	{'max_depth': 5, 'min_samples_leaf': 3}
	{'algorithm': 'auto', 'n_neighbors': 4, 'weights': 'uniform'}
	{'criterion': 'gini',
	 'max_depth': 4,
	 'min_samples_leaf': 2,
	 'min_samples_split': 2,
	 'n_estimators': 3,
	 'random_state': 42}
	{'algorithm': 'SAMME', 'n_estimators': 4, 'random_state': 23}
	
	
	>>> 参数导入Tester后的测试结果：
	feature: ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus_salary']
		GaussianNB(priors=None) Precision: 0.48581	Recall: 0.35100
		DecisionTreeClassifier() Precision: 0.37740	Recall: 0.26550
		KNeighborsClassifier(n_neighbors=3) Precision: 0.79960	Recall: 0.19750
		RandomForestClassifier() Precision: 0.48871	Recall: 0.11900
		AdaBoostClassifier(base_estimator=DecisionTreeClassifier()) Precision: 0.37246	Recall: 0.16500
	```
	
	- 猜测GridSearchCV结果不好的原因可能是由于该算法评估标准是Accuracy，而在这里我们以Precision和Recall为标准，故不能满足要求
	- 手动调参: 主要调整了决策树和KNN的参数
		- 调整决策树参数，结果默认参数最好：
		```
		>>>默认参数：
		Accuracy: 0.80208	Precision: 0.36441	Recall: 0.38500	F1: 0.37442	F2: 0.38070
		>>> 调整min_samples_split=10，20，50
		10：Accuracy: 0.81585	Precision: 0.38357	Recall: 0.32450	F1: 0.35157	F2: 0.33481
		20：Accuracy: 0.80169	Precision: 0.19706	Recall: 0.09400	F1: 0.12729	F2: 0.10498
		50：Accuracy: 0.82262	Precision: 0.25559	Recall: 0.08000	F1: 0.12186	F2: 0.09274
		```
		- 调整KNN参数,经调整n_neighbor影响最大，n_neighbor=3结果最好：
		```
		>>> 默认参数n_neighbor=5：
		Accuracy: 0.87785	Precision: 0.79683	Recall: 0.27650	F1: 0.41054	F2: 0.31804
		>>> 调整n_neighbor = 2,3,4,6
		2: Accuracy: 0.86931	Precision: 0.77615	Recall: 0.21150	F1: 0.33242	F2: 0.24751
		3: Accuracy: 0.86177	Precision: 0.60181	Recall: 0.30000	F1: 0.40040	F2: 0.33344
		4: Accuracy: 0.86892	Precision: 0.79960	Recall: 0.19750	F1: 0.31676	F2: 0.23252
		6: Accuracy: 0.84577	Precision: 0.48276	Recall: 0.03500	F1: 0.06527	F2: 0.04297
		```
	

***
5.什么是验证？未执行情况下的典型错误是什么？你是如何验证你的分析的？

* 验证就是评估选用的算法是否能够达到目标，使用各自的训练和测试数据，得以基于独立数据集来评估分类器或回归的性能，或者作为过度拟合的检查因素。
* 未执行时的典型错误就是过拟合。
* 验证方法：将数据集拆分为训练和测试集，使用交叉验证。
	- 这个数据集很不平衡，poi在数据集中占少数，因此使用train_test_split()交叉验证不是很好。
	- 本项目使用了Stratified Shuffle Split方式将数据集分为训练和测试集。这个交叉验证是一种分层随机分割的方式，通过保留每个类别的样本的百分比来进行折叠，这种方式能尽量保证每次取出的样本不同，尽量用到所有数据。

***
6.给出至少两个评估变量并说明每个的平均性能。解释对用简单的语言表明算法性能的度量的解读

* 精确度Precision：预测poi正确/(预测poi正确 + 预测poi错误)
	- 这个评估变量表示针对所有预测为poi的数据中的命中率
* 召回率Recall: 预测poi正确/(预测poi正确 + 预测非poi错误)
	- 这个评估变量表示针对所有真实的poi，这次预测的准确率

***

