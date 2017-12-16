#!/usr/bin/python

import pickle
import pprint

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tester import test_classifier

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".


# features_list = ["poi","bonus", "exercised_stock_options", "director_fees"]  # You will need to use more features
features_list = ["poi", "bonus", "exercised_stock_options", "total_stock_value"]
# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
print("Before Clean:", len(data_dict))
poi_list = []
not_poi_list = []
for name in data_dict:
    if data_dict[name]["poi"]:
        poi_list.append(name)
    else:
        not_poi_list.append(name)
print("There are", len(poi_list),"poi in this dataset")
print("There are", len(not_poi_list), "Non-poi in this dataset")

data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)


# Task 3: Create new feature(s)
# 调查员工的奖金与薪水比，创建新特征bonus_salary
for name in data_dict:
    try:
        data_dict[name]["bonus_salary"] = data_dict[name]["bonus"] / data_dict[name]["salary"]
    except:
        data_dict[name]["bonus_salary"] = 0


# Store to my_dataset for easy export below.
my_features = ["poi","bonus", "exercised_stock_options", "total_stock_value"]
my_dataset = data_dict

all_features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                     'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                     'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                     'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',
                     'from_this_person_to_poi', 'shared_receipt_with_poi','bonus_salary']
# print(len(all_features_list))
# Extract features and labels from dataset for local testing
"""
featureFormat(dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False,
                  sort_keys=False):
 convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
"""
data = featureFormat(my_dataset, all_features_list, sort_keys=True, remove_all_zeroes=False, remove_any_zeroes=False)


print("After Clean:", len(data))
labels, features = targetFeatureSplit(data)

# 使用SelectKBest选取特征
selector = SelectKBest()
selector.fit_transform(features, labels)
DT_selector = DecisionTreeClassifier()
DT_selector.fit(features, labels)

# 提取分数，打印按分数排序的特征列表
features_scores = {feature : score for feature, score in zip(all_features_list[1:], selector.scores_)}
sorted_features_scores = sorted(features_scores.items(), key=lambda a:a[1], reverse=True)
pprint.pprint(sorted_features_scores[:5])


# Task 4: Try a variety of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# clf = KNeighborsClassifier(n_neighbors=3)
#
names = ["Naive Bayes", "Decision Tree", "Nearest Neighbor", "Random Forest", "AdaBoost"]
classifiers = [GaussianNB(priors=None),
               DecisionTreeClassifier(min_samples_split=50),
               KNeighborsClassifier(n_neighbors=7),
               RandomForestClassifier(),
               AdaBoostClassifier()]
# parameters = {"Naive Bayes":{},
#               "Decision Tree":{"max_depth": range(5,15),
#                                "min_samples_leaf": range(1,5)},
#               "Nearest Neighbor":{"n_neighbors": range(1, 10),
#                                   "weights":("uniform", "distance"),
#                                   "algorithm":("auto", "ball_tree", "kd_tree", "brute")},
#               "Random Forest":{"n_estimators": range(2, 5),
#                                "min_samples_split": range(2, 5),
#                                "max_depth": range(2, 15),
#                                "min_samples_leaf": range(1, 5),
#                                "random_state": [0, 10, 23, 36, 42],
#                                "criterion": ["entropy", "gini"]},
#               "AdaBoost":{"n_estimators": range(2, 5),
#                           "algorithm":("SAMME", "SAMME.R"),
#                           "random_state":[0, 10, 23, 36, 42]}}

# clf = Pipeline(
#     [("feature_selection", SelectFromModel(LinearSVC("l1"))), ("reduce_dim", PCA()), ("clf_GaussianNB", GaussianNB()),
#      ("clf_DecisionTree", DecisionTreeClassifier()),
#      ("clf_svc", SVC()), ("clf_neighbor", NearestNeighbors()), ("clf_RandomForest", RandomForestClassifier()),
#      ("clf_AdaBoost", AdaBoostClassifier())])

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)
#
# print("feature:", my_features)
# for name, clf in zip(names, classifiers):
#     # grid_clf = GridSearchCV(clf, parameters[name])
#     # grid_clf.fit(features_train, labels_train)
#     # pprint.pprint(grid_clf.best_params_)
#     test_classifier(clf, my_dataset, my_features, folds=1000)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.
clf = GaussianNB()
test_classifier(clf, my_dataset, my_features, folds=1000)
dump_classifier_and_data(clf, my_dataset, my_features)
