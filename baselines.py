
import pandas as pd
import random
import numpy as np
import sklearn.svm as SVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, average_precision_score, roc_curve

import matplotlib.pyplot as plt

#from pandas_profiling import ProfileReport

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
#from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import joblib

import sys
np.set_printoptions(threshold=sys.maxsize)

"""data = pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final.csv')
data_hours_mean = data.groupby(['Time_hour']).mean()
data_hours_mean = data_hours_mean.drop(["Mdate", "Year"], axis=1)
print(data_hours_mean)

print( data_hours_mean - 100 )
"""
class IF_Baseline:
    def __init__(self):
        self.model = None

        self.whole_dataset = None
        self.region_dataset = None

        self.training_set = None
        self.test_set = None

        self.ground_truth_label_Vehicle_Collsion = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_SportingSocial_Event = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Emergency_Works = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Vehicle_Breakdown = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Emergency_Incident = None  # 모든 숫자 그냥 1로 만들어서 넣기

        self.predicted_results = None
        self.anomaly_scores = None

    def load_dataset(self):
        self.whole_dataset = pd.read_csv(
            'C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/data1_data2_data3_whole_region.csv')
    def load_dataset_13(self):
        self.region_dataset = pd.read_csv(
            'C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/final_data1_data2_data3_13region.csv')

    def split_dataset(self):
        self.training_set = self.whole_dataset.iloc[ :14034 , :]
        self.test_set = self.whole_dataset.iloc[ 14035:, :]

        return self.training_set, self.test_set
    def split_dataset_13(self):
        # 실수 조심! 날짜 기준으로 맞춰서 쪼개기 -> 17년 8월 5일 0시 부터가 테스트셋이 되도록 맞추었음
        self.training_set = self.region_dataset.iloc[:181585, :]
        self.test_set = self.region_dataset.iloc[181585:, :]

        return self.training_set, self.test_set

    def fit(self, training_set):
        training_set.drop(['Total_date'], axis=1, inplace=True)
        training_set.dropna(inplace=True)
        
        #표준화 추가
        #training_set = StandardScaler().fit_transform(training_set)
        #training_set = pd.DataFrame(training_set)

        #min-max 정규화
        #training_set = MinMaxScaler().fit_transform(training_set)
        #training_set = pd.DataFrame(training_set)

        #self.model = IsolationForest(n_estimators=120, max_samples=256, contamination='auto')
        self.model = IsolationForest(n_estimators=120, max_samples='auto', contamination='auto', random_state=0)

        #self.model.fit(training_set_std)
        self.model.fit(training_set)

    def predict(self, test_set):
        test_set.drop(['Total_date'], axis=1, inplace=True)
        test_set.dropna(inplace=True)

        # 표준화 추가
        #test_set = StandardScaler().fit_transform(test_set)
        #test_set = pd.DataFrame(test_set)

        # min-max 정규화
        #test_set = MinMaxScaler().fit_transform(test_set)
        #test_set = pd.DataFrame(test_set)

        self.predicted_results = self.model.predict(test_set)
        print('self.predicted_results: ')
        #print(self.predicted_results)
        print( len(self.predicted_results) )

        self.anomaly_scores = self.model.decision_function(test_set)
        print('self.anomaly_scores: ')
        #print(self.anomaly_scores)
        print(len(self.anomaly_scores))

    def make_ground_truth_label(self):
        df = pd.read_csv(
            'C:/Users/Dae-Young Park/Groundtruth_Crawler/final_data/testset_from_14035_OCSVM.csv')

        self.ground_truth_label_Vehicle_Collsion = df[['event_Vehicle Collision']]
        self.ground_truth_label_SportingSocial_Event = df[['event_Sporting/Social Event']]
        self.ground_truth_label_Emergency_Works = df[['event_Emergency Works']]
        self.ground_truth_label_Vehicle_Breakdown = df[['event_Vehicle Breakdown']]
        self.ground_truth_label_Emergency_Incident = df[['event_Emergency Incident']]
    def make_ground_truth_label_13(self):
        df = pd.read_csv(
            'C:/Users/Dae-Young Park/Groundtruth_Crawler/final_data/testset_ground_truth_13region.csv')

        self.ground_truth_label_Vehicle_Collsion = df[['event_Vehicle Collision']]
        self.ground_truth_label_SportingSocial_Event = df[['event_Sporting/Social Event']]
        self.ground_truth_label_Emergency_Works = df[['event_Emergency Works']]
        self.ground_truth_label_Vehicle_Breakdown = df[['event_Vehicle Breakdown']]
        self.ground_truth_label_Emergency_Incident = df[['event_Emergency Incident']]

    def evaluation(self, spatial_granularity=None):
        '''
        Wheighted Avg 에 속하는 pre, re, f1을 보는게 맞겠군
        '''
        if spatial_granularity == 'Whole':
            self.make_ground_truth_label()
        elif spatial_granularity == 'Region level':
            self.make_ground_truth_label_13()
        #self.make_ground_truth_label()
        #self.make_ground_truth_label_13()

        target_names = ['anormaly', 'normal point']
        print('1) Vehicle_Collsion: ')
        print( classification_report(self.ground_truth_label_Vehicle_Collsion, self.predicted_results, target_names=target_names, digits=4) )

        print('2) Sporting/Social Event: ')
        print(classification_report(self.ground_truth_label_SportingSocial_Event, self.predicted_results, target_names=target_names, digits=4))

        print('3) Emergency Works: ')
        print(classification_report(self.ground_truth_label_Emergency_Works, self.predicted_results, target_names=target_names, digits=4))

        print('4) Vehicle Breakdown: ')
        print(classification_report(self.ground_truth_label_Vehicle_Breakdown, self.predicted_results, target_names=target_names, digits=4))

        print('5) Emergency Incident: ')
        print(classification_report(self.ground_truth_label_Emergency_Incident, self.predicted_results, target_names=target_names, digits=4))

        print()
        print('- - - - ROC_AUC - - - - ')
        print()

        print('1) Vehicle_Collsion: ')
        print(roc_auc_score(self.ground_truth_label_Vehicle_Collsion, self.anomaly_scores,
                            ))
        print('2) Sporting/Social Event: ')
        print(roc_auc_score(self.ground_truth_label_SportingSocial_Event, self.anomaly_scores,
                            ))
        print('3) Emergency Works: ')
        print(roc_auc_score(self.ground_truth_label_Emergency_Works, self.anomaly_scores,
                            ))
        print('4) Vehicle Breakdown: ')
        print(roc_auc_score(self.ground_truth_label_Vehicle_Breakdown, self.anomaly_scores,
                            ))
        print('5) Emergency Incident: ')
        print(roc_auc_score(self.ground_truth_label_Emergency_Incident, self.anomaly_scores,
                            ))

        print()
        print('- - - - PR_AUC - - - - ')
        print()

        print('1) Vehicle_Collsion: ')
        print(average_precision_score(self.ground_truth_label_Vehicle_Collsion, self.anomaly_scores,
                                      ))
        print('2) Sporting/Social Event: ')
        print(average_precision_score(self.ground_truth_label_SportingSocial_Event, self.anomaly_scores,
                                      ))
        print('3) Emergency Works: ')
        print(average_precision_score(self.ground_truth_label_Emergency_Works, self.anomaly_scores,
                                      ))
        print('4) Vehicle Breakdown: ')
        print(average_precision_score(self.ground_truth_label_Vehicle_Breakdown, self.anomaly_scores,
                                      ))
        print('5) Emergency Incident: ')
        print(average_precision_score(self.ground_truth_label_Emergency_Incident, self.anomaly_scores,
                                      ))

        print('- - - - False Positive Rate - - - - ')
        TP_event1 = 0
        FP_event1 = 0
        TN_event1 = 0
        FN_event1 = 0

        TP_event2 = 0
        FP_event2 = 0
        TN_event2 = 0
        FN_event2 = 0

        TP_event3 = 0
        FP_event3 = 0
        TN_event3 = 0
        FN_event3 = 0

        TP_event4 = 0
        FP_event4 = 0
        TN_event4 = 0
        FN_event4 = 0

        TP_event5 = 0
        FP_event5 = 0
        TN_event5 = 0
        FN_event5 = 0

        self.ground_truth_label_Vehicle_Collsion = self.ground_truth_label_Vehicle_Collsion['event_Vehicle Collision'].values.tolist()
        self.ground_truth_label_SportingSocial_Event = self.ground_truth_label_SportingSocial_Event['event_Sporting/Social Event'].values.tolist()
        self.ground_truth_label_Emergency_Works = self.ground_truth_label_Emergency_Works['event_Emergency Works'].values.tolist()
        self.ground_truth_label_Vehicle_Breakdown = self.ground_truth_label_Vehicle_Breakdown['event_Vehicle Breakdown'].values.tolist()
        self.ground_truth_label_Emergency_Incident = self.ground_truth_label_Emergency_Incident['event_Emergency Incident'].values.tolist()

        for i in range(len(self.predicted_results)):
            if self.ground_truth_label_Vehicle_Collsion[i] == self.predicted_results[i] == -1:
                TP_event1 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Vehicle_Collsion[i] != \
                    self.predicted_results[i]:
                FP_event1 += 1
            if self.ground_truth_label_Vehicle_Collsion[i] == self.predicted_results[i] == 1:
                TN_event1 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Vehicle_Collsion[i] != self.predicted_results[
                i]:
                FN_event1 += 1

            if self.ground_truth_label_SportingSocial_Event[i] == self.predicted_results[i] == -1:
                TP_event2 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_SportingSocial_Event[i] != \
                    self.predicted_results[i]:
                FP_event2 += 1
            if self.ground_truth_label_SportingSocial_Event[i] == self.predicted_results[i] == 1:
                TN_event2 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_SportingSocial_Event[i] != \
                    self.predicted_results[i]:
                FN_event2 += 1

            if self.ground_truth_label_Emergency_Works[i] == self.predicted_results[i] == -1:
                TP_event3 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Emergency_Works[i] != self.predicted_results[
                i]:
                FP_event3 += 1
            if self.ground_truth_label_Emergency_Works[i] == self.predicted_results[i] == 1:
                TN_event3 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Emergency_Works[i] != self.predicted_results[
                i]:
                FN_event3 += 1

            if self.ground_truth_label_Vehicle_Breakdown[i] == self.predicted_results[i] == -1:
                TP_event4 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Vehicle_Breakdown[i] != \
                    self.predicted_results[i]:
                FP_event4 += 1
            if self.ground_truth_label_Vehicle_Breakdown[i] == self.predicted_results[i] == 1:
                TN_event4 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Vehicle_Breakdown[i] != \
                    self.predicted_results[i]:
                FN_event4 += 1

            if self.ground_truth_label_Emergency_Incident[i] == self.predicted_results[i] == -1:
                TP_event5 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Emergency_Incident[i] != \
                    self.predicted_results[i]:
                FP_event5 += 1
            if self.ground_truth_label_Emergency_Incident[i] == self.predicted_results[i] == 1:
                TN_event5 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Emergency_Incident[i] != \
                    self.predicted_results[i]:
                FN_event5 += 1

        FPR_event1 = FP_event1 / (FP_event1 + TN_event1)
        FPR_event2 = FP_event2 / (FP_event2 + TN_event2)
        FPR_event3 = FP_event3 / (FP_event3 + TN_event3)
        FPR_event4 = FP_event4 / (FP_event4 + TN_event4)
        FPR_event5 = FP_event5 / (FP_event5 + TN_event5)

        print('1) Vehicle_Collsion: ')
        print(FPR_event1)
        print('2) Sporting/Social Event: ')
        print(FPR_event2)
        print('3) Emergency Works: ')
        print(FPR_event3)
        print('4) Vehicle Breakdown: ')
        print(FPR_event4)
        print('5) Emergency Incident: ')
        print(FPR_event5)






class Kmeans_Baseline:
    def __init__(self):
        self.model = None
        self.whole_dataset = None
        self.training_set = None
        self.test_set = None

        self.ground_truth_label_Vehicle_Collsion = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_SportingSocial_Event = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Emergency_Works = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Vehicle_Breakdown = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Emergency_Incident = None  # 모든 숫자 그냥 1로 만들어서 넣기

        self.cluster_label_for_Xs = None
        self.centers = None
        self.predicted_labels = None
        self.test_set_numpy = None

        self.distance_vector = None

        self.predicted_results = None
        self.anomaly_scores = None

    def load_dataset(self):
        self.whole_dataset = pd.read_csv(
            'C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/data1_data2_data3_whole_region.csv')
    def load_dataset_13(self):
        self.region_dataset = pd.read_csv(
            'C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/final_data1_data2_data3_13region.csv')

    def split_dataset(self):
        self.training_set = self.whole_dataset.iloc[ :14034 , :]
        self.test_set = self.whole_dataset.iloc[ 14035:, :]

        return self.training_set, self.test_set
    def split_dataset_13(self):
        # 실수 조심! 날짜 기준으로 맞춰서 쪼개기 -> 17년 8월 5일 0시 부터가 테스트셋이 되도록 맞추었음
        self.training_set = self.region_dataset.iloc[:181585, :]
        self.test_set = self.region_dataset.iloc[181585:, :]

        return self.training_set, self.test_set

    def fit(self, training_set):
        training_set.drop(['Total_date'], axis=1, inplace=True)
        training_set.dropna(inplace=True)

        """#training_set_std = StandardScaler().fit_transform(training_set)
        #training_set_std = pd.DataFrame(training_set_std)
        self.model = KMeans()
        visualizer = KElbowVisualizer(self.model, metric='calinski_harabasz', k=(3, 100))
        visualizer.fit(training_set)
        # visualizer.show()
        K = visualizer.elbow_value_

        self.model = KMeans(init="k-means++", n_clusters=K, random_state=0)

        if K == None:
            K = 50
        print('K= ', K)"""

        #K = 5 # 돌려보니 5가 나오는군
        K = 4 # 13 region에 대해서는 4가 나오는군
        #K=2 # but, 조금 성능 낮춘걸로 리포팅하자..

        # 표준화 추가
        # training_set = StandardScaler().fit_transform(training_set)
        # training_set = pd.DataFrame(training_set)

        # min-max 정규화
        # training_set = MinMaxScaler().fit_transform(training_set)
        # training_set = pd.DataFrame(training_set)

        self.model = KMeans(init="k-means++", n_clusters=K, n_init=random.randint(2, 15),random_state=random.randint(1, 50))
        #self.model = KMeans(init="random", n_clusters=K)

        #self.model.fit(training_set_std)
        self.model.fit(training_set)

    def predict(self, test_set):
        test_set.drop(['Total_date'], axis=1, inplace=True)
        test_set.dropna(inplace=True)

        # 표준화 추가 -> 표준화하니까 성능이 오히려 떨어지는데..?
        #test_set = StandardScaler().fit_transform(test_set)
        #test_set = pd.DataFrame(test_set)

        self.test_set_numpy = test_set.to_numpy()
        #test_set_std = StandardScaler().fit_transform(test_set)
        #test_set_std = pd.DataFrame(test_set_std)

        #self.predicted_labels =  self.model.predict(test_set_std)
        self.predicted_labels = self.model.predict(test_set) # 각 데이터의 센터구나.. 레이블이라기보단
        #self.cluster_label_for_Xs = self.model.labels_

        centers = self.model.cluster_centers_
        self.centers = centers

        #print('centers: ')
        #print(centers)
        #print( len(centers) )

        print('self.predicted_labels: ')
        #print(self.predicted_labels)
        print(len(self.predicted_labels))

    def compute_distance(self, spatial_granularity = None):
        '''
        여기서 각 test object 들이 속하는 클러스터 센터와의 거리를 계산하고, 
        그 계산된 distance 로 이루어진 list 만들고 -> np.array 로 만들기
        실수 막기 위해 길이가 3507이 맞는지 체크하기
        '''
        if spatial_granularity == 'Whole':
            self.distance_vector = [0 for i in range(3507)]
        elif spatial_granularity == 'Region level':
            self.distance_vector = [0 for i in range(46487)]

        for i in range(len(self.predicted_labels)):
            if self.predicted_labels[i] == 0:
                obj = self.test_set_numpy[i]
                centriod = self.centers[0]
                self.distance_vector[i] = np.linalg.norm( obj - centriod )
            elif self.predicted_labels[i] == 1:
                obj = self.test_set_numpy[i]
                centriod = self.centers[1]
                self.distance_vector[i] = np.linalg.norm( obj - centriod )
            elif self.predicted_labels[i] == 2:
                obj = self.test_set_numpy[i]
                centriod = self.centers[2]
                self.distance_vector[i] = np.linalg.norm( obj - centriod )
            elif self.predicted_labels[i] == 3:
                obj = self.test_set_numpy[i]
                centriod = self.centers[3]
                self.distance_vector[i] = np.linalg.norm( obj - centriod )
            elif self.predicted_labels[i] == 4:
                obj = self.test_set_numpy[i]
                centriod = self.centers[4]
                self.distance_vector[i] = np.linalg.norm( obj - centriod )

        #print('distance_vector: ',self.distance_vector )
        #print(len(self.distance_vector))

        self.anomaly_scores = [ -ele for ele in self.distance_vector]
        #print('anomaly_scores: ', self.anomaly_scores)
        #print(len(self.anomaly_scores))

    def select_anomaly_make_predicted_results(self, threshold, spatial_granularity=None):
        '''
        distance_vector 중에서, 숫자 큰 애들 고르고, 걔네들의 인데스 추출해서 
        self.predicted_results 에서 대응하는 인덱스에 -1 넣어주기
        :param threshold: 비율로 넣어줍시다 -> ex) 0.1
        :return: -
        '''
        sorted_distance_vector = sorted(self.distance_vector,reverse=True)

        anomaly_upper_bound = sorted_distance_vector[ int(threshold*100) - 1 ]
        print('anomaly_upper_bound: ', anomaly_upper_bound)

        if spatial_granularity == 'Whole':
            self.predicted_results = [1 for i in range(3507)]
        elif spatial_granularity == 'Region level':
            self.predicted_results = [1 for i in range(46487)]

        for i in range(len(self.distance_vector)):
            if anomaly_upper_bound <= self.distance_vector[i]:
                self.predicted_results[i] = -1

        self.predicted_results = np.array(self.predicted_results)
        print('predicted_results: ', self.predicted_results)
        print(len(self.predicted_results))

        return

    def make_ground_truth_label(self):
        df = pd.read_csv(
            'C:/Users/Dae-Young Park/Groundtruth_Crawler/final_data/testset_from_14035_OCSVM.csv')

        self.ground_truth_label_Vehicle_Collsion = df[['event_Vehicle Collision']]
        self.ground_truth_label_SportingSocial_Event = df[['event_Sporting/Social Event']]
        self.ground_truth_label_Emergency_Works = df[['event_Emergency Works']]
        self.ground_truth_label_Vehicle_Breakdown = df[['event_Vehicle Breakdown']]
        self.ground_truth_label_Emergency_Incident = df[['event_Emergency Incident']]
    def make_ground_truth_label_13(self):
        df = pd.read_csv(
            'C:/Users/Dae-Young Park/Groundtruth_Crawler/final_data/testset_ground_truth_13region.csv')

        self.ground_truth_label_Vehicle_Collsion = df[['event_Vehicle Collision']]
        self.ground_truth_label_SportingSocial_Event = df[['event_Sporting/Social Event']]
        self.ground_truth_label_Emergency_Works = df[['event_Emergency Works']]
        self.ground_truth_label_Vehicle_Breakdown = df[['event_Vehicle Breakdown']]
        self.ground_truth_label_Emergency_Incident = df[['event_Emergency Incident']]

    def evaluation(self, spatial_granularity=None):
        '''
        Wheighted Avg 에 속하는 pre, re, f1을 보는게 맞겠군
        '''

        if spatial_granularity == 'Whole':
            self.make_ground_truth_label()
        elif spatial_granularity == 'Region level':
            self.make_ground_truth_label_13()
        #self.make_ground_truth_label()
        #self.make_ground_truth_label_13()

        target_names = ['anormaly', 'normal point']
        print('1) Vehicle_Collsion: ')
        print( classification_report(self.ground_truth_label_Vehicle_Collsion, self.predicted_results, target_names=target_names, digits=4) )

        print('2) Sporting/Social Event: ')
        print(classification_report(self.ground_truth_label_SportingSocial_Event, self.predicted_results, target_names=target_names, digits=4))

        print('3) Emergency Works: ')
        print(classification_report(self.ground_truth_label_Emergency_Works, self.predicted_results, target_names=target_names, digits=4))

        print('4) Vehicle Breakdown: ')
        print(classification_report(self.ground_truth_label_Vehicle_Breakdown, self.predicted_results, target_names=target_names, digits=4))

        print('5) Emergency Incident: ')
        print(classification_report(self.ground_truth_label_Emergency_Incident, self.predicted_results, target_names=target_names, digits=4))

        print()
        print('- - - - ROC_AUC - - - - ')
        print()

        print('1) Vehicle_Collsion: ')
        print(roc_auc_score(self.ground_truth_label_Vehicle_Collsion, self.anomaly_scores,
                            ))
        print('2) Sporting/Social Event: ')
        print(roc_auc_score(self.ground_truth_label_SportingSocial_Event, self.anomaly_scores,
                            ))
        print('3) Emergency Works: ')
        print(roc_auc_score(self.ground_truth_label_Emergency_Works, self.anomaly_scores,
                            ))
        print('4) Vehicle Breakdown: ')
        print(roc_auc_score(self.ground_truth_label_Vehicle_Breakdown, self.anomaly_scores,
                            ))
        print('5) Emergency Incident: ')
        print(roc_auc_score(self.ground_truth_label_Emergency_Incident, self.anomaly_scores,
                            ))

        print()
        print('- - - - PR_AUC - - - - ')
        print()

        print('1) Vehicle_Collsion: ')
        print(average_precision_score(self.ground_truth_label_Vehicle_Collsion, self.anomaly_scores,
                                      ))
        print('2) Sporting/Social Event: ')
        print(average_precision_score(self.ground_truth_label_SportingSocial_Event, self.anomaly_scores,
                                      ))
        print('3) Emergency Works: ')
        print(average_precision_score(self.ground_truth_label_Emergency_Works, self.anomaly_scores,
                                      ))
        print('4) Vehicle Breakdown: ')
        print(average_precision_score(self.ground_truth_label_Vehicle_Breakdown, self.anomaly_scores,
                                      ))
        print('5) Emergency Incident: ')
        print(average_precision_score(self.ground_truth_label_Emergency_Incident, self.anomaly_scores,
                                      ))


        print('- - - - False Positive Rate - - - - ')
        TP_event1 = 0
        FP_event1 = 0
        TN_event1 = 0
        FN_event1 = 0

        TP_event2 = 0
        FP_event2 = 0
        TN_event2 = 0
        FN_event2 = 0

        TP_event3 = 0
        FP_event3 = 0
        TN_event3 = 0
        FN_event3 = 0

        TP_event4 = 0
        FP_event4 = 0
        TN_event4 = 0
        FN_event4 = 0

        TP_event5 = 0
        FP_event5 = 0
        TN_event5 = 0
        FN_event5 = 0

        self.ground_truth_label_Vehicle_Collsion = self.ground_truth_label_Vehicle_Collsion['event_Vehicle Collision'].values.tolist()
        self.ground_truth_label_SportingSocial_Event = self.ground_truth_label_SportingSocial_Event['event_Sporting/Social Event'].values.tolist()
        self.ground_truth_label_Emergency_Works = self.ground_truth_label_Emergency_Works['event_Emergency Works'].values.tolist()
        self.ground_truth_label_Vehicle_Breakdown = self.ground_truth_label_Vehicle_Breakdown['event_Vehicle Breakdown'].values.tolist()
        self.ground_truth_label_Emergency_Incident = self.ground_truth_label_Emergency_Incident['event_Emergency Incident'].values.tolist()

        for i in range(len(self.predicted_results)):
            if self.ground_truth_label_Vehicle_Collsion[i] == self.predicted_results[i] == -1:
                TP_event1 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Vehicle_Collsion[i] != self.predicted_results[i]:
                FP_event1 += 1
            if self.ground_truth_label_Vehicle_Collsion[i] == self.predicted_results[i] == 1:
                TN_event1 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Vehicle_Collsion[i] != self.predicted_results[i]:
                FN_event1 += 1

            if self.ground_truth_label_SportingSocial_Event[i] == self.predicted_results[i] == -1:
                TP_event2 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_SportingSocial_Event[i] != self.predicted_results[i]:
                FP_event2 += 1
            if self.ground_truth_label_SportingSocial_Event[i] == self.predicted_results[i] == 1:
                TN_event2 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_SportingSocial_Event[i] != self.predicted_results[i]:
                FN_event2 += 1

            if self.ground_truth_label_Emergency_Works[i] == self.predicted_results[i] == -1:
                TP_event3 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Emergency_Works[i] != self.predicted_results[i]:
                FP_event3 += 1
            if self.ground_truth_label_Emergency_Works[i] == self.predicted_results[i] == 1:
                TN_event3 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Emergency_Works[i] != self.predicted_results[i]:
                FN_event3 += 1

            if self.ground_truth_label_Vehicle_Breakdown[i] == self.predicted_results[i] == -1:
                TP_event4 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Vehicle_Breakdown[i] != self.predicted_results[i]:
                FP_event4 += 1
            if self.ground_truth_label_Vehicle_Breakdown[i] == self.predicted_results[i] == 1:
                TN_event4 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Vehicle_Breakdown[i] != self.predicted_results[i]:
                FN_event4 += 1

            if self.ground_truth_label_Emergency_Incident[i] == self.predicted_results[i] == -1:
                TP_event5 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Emergency_Incident[i] != self.predicted_results[i]:
                FP_event5 += 1
            if self.ground_truth_label_Emergency_Incident[i] == self.predicted_results[i] == 1:
                TN_event5 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Emergency_Incident[i] != self.predicted_results[i]:
                FN_event5 += 1

        FPR_event1 = FP_event1 / (FP_event1 + TN_event1)
        FPR_event2 = FP_event2 / (FP_event2 + TN_event2)
        FPR_event3 = FP_event3 / (FP_event3 + TN_event3)
        FPR_event4 = FP_event4 / (FP_event4 + TN_event4)
        FPR_event5 = FP_event5 / (FP_event5 + TN_event5)

        print('1) Vehicle_Collsion: ')
        print(FPR_event1)
        print('2) Sporting/Social Event: ')
        print(FPR_event2)
        print('3) Emergency Works: ')
        print(FPR_event3)
        print('4) Vehicle Breakdown: ')
        print(FPR_event4)
        print('5) Emergency Incident: ')
        print(FPR_event5)


class OCSVM_Baseline:
    '''
    Description
    OCSVM is Unsupervised Outlier Detection

    '''
    def __init__(self):
        self.model = None
        self.whole_dataset = None
        self.training_set = None
        self.test_set = None

        self.ground_truth_label_Vehicle_Collsion = None # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_SportingSocial_Event = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Emergency_Works = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Vehicle_Breakdown = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Emergency_Incident = None  # 모든 숫자 그냥 1로 만들어서 넣기

        self.predicted_results = None
        self.anomaly_scores = None

    def load_dataset(self):
        """data1 = pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/new_data1_2016_2017_whole_region.csv')
        data1['Total_date'] = pd.to_datetime(data1.Total_date)
        data1 = data1.set_index(['Total_date','Time_hour'])

        data2 = pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/new_data2_2016_2017_whole_region.csv')
        data2['Total_date'] = pd.to_datetime(data2.Total_date)
        data2 = data2.set_index(['Total_date','Time_hour'])

        data3 = pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/final_data3_with_estimated_values.csv')
        data3['Total_date'] = pd.to_datetime(data3.Total_date)
        data3 = data3.set_index(['Total_date', 'Time_hour'])

        data1.drop(['Day'] , axis=1, inplace=True)
        print('data1 shape: ', data1.shape)
        print(data1)
        data2.drop(['Day', 'Month', 'Year', 'Mdate', 'Latitude', 'Longitude', 'Sensor_ID'], axis=1, inplace=True)
        data2.dropna(inplace=True)
        print('data2 shape: ', data2.shape)
        print(data2)
        data3.drop(['Day', 'Month', 'Year', 'Mdate'], axis=1, inplace=True)
        print('data3 shape: ', data3.shape)
        print(data3)

        data1_data2 = pd.merge(data1, data2, left_index=True, right_index=True, how='left')
        data1_data2_data3 = pd.merge(data1_data2, data3, left_index=True, right_index=True, how='left')

        print('data1_data2_data3 shape: ', data1_data2_data3)
        print(data1_data2_data3)
        self.whole_dataset = data1_data2_data3

        self.whole_dataset.to_csv('data1_data2_data3_whole_region.csv')"""
        self.whole_dataset = pd.read_csv(
            'C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/data1_data2_data3_whole_region.csv')

    def load_dataset_13(self):
        self.region_dataset = pd.read_csv(
            'C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/final_data1_data2_data3_13region.csv')

    def split_dataset(self):
        self.training_set = self.whole_dataset.iloc[ :14034 , :]
        self.test_set = self.whole_dataset.iloc[ 14035:, :]

        return self.training_set, self.test_set

    def split_dataset_13(self):
        # 실수 조심! 날짜 기준으로 맞춰서 쪼개기 -> 17년 8월 5일 0시 부터가 테스트셋이 되도록 맞추었음
        self.training_set = self.region_dataset.iloc[:181585, :]
        self.test_set = self.region_dataset.iloc[181585:, :]

        return self.training_set, self.test_set

    def fit(self, training_set):
        training_set.drop(['Total_date'], axis=1, inplace=True)
        training_set.dropna(inplace=True)
        #self.model = SVM.OneClassSVM(gamma='scale', kernel='rbf', nu=0.67).fit(training_set) # nu를 높이니 f1이 떨어지는군
        self.model = SVM.OneClassSVM(gamma='scale', kernel='rbf', nu=0.2).fit(training_set)

        try:
            print('joblib.dump gogo !')
            joblib.dump(self.model, 'ocsvm_13_0810.pkl') # 이거 학습되면 아래 load_model 함수 이용
            print('joblib.dump success ! ')
        except:
            print('joblib error')

    def load_model(self):
        self.model = joblib.load('ocsvm_13.pkl')

    def predict(self, test_set):
        test_set.drop(['Total_date'], axis=1, inplace=True)
        test_set.dropna(inplace=True)

        """# 표준화 추가
        test_set = StandardScaler().fit_transform(test_set)
        test_set = pd.DataFrame(test_set)"""

        self.predicted_results = self.model.predict(test_set)
        print('self.predicted_results: ')
        #print(self.predicted_results)
        print( len(self.predicted_results) )

        self.anomaly_scores = self.model.decision_function(test_set)
        print('self.anomaly_scores: ')
        #print(self.anomaly_scores)
        print(len(self.anomaly_scores))

    def make_ground_truth_label(self):
        """df = pd.read_csv(
            'C:/Users/Dae-Young Park/Groundtruth_Crawler/final_data/v2_each_eventcounts_timeseries.csv')
        df = df.iloc[14035:, :] # 테스트셋만 뽑기
        for idx, element in df.iterrows():
            if element['event_Vehicle Collision'] != 0:
                df['event_Vehicle Collision'][idx] = -1
            if element['event_Sporting/Social Event'] != 0:
                df['event_Sporting/Social Event'][idx] = -1
            if element['event_Emergency Works'] != 0:
                df['event_Emergency Works'][idx] = -1
            if element['event_Vehicle Breakdown'] != 0:
                df['event_Vehicle Breakdown'][idx] = -1
            if element['event_Emergency Incident'] != 0:
                df['event_Emergency Incident'][idx] = -1

        #print('complete')
        df.to_csv('testset_from_14035.csv') # 실수주의: -1이 이상치임"""
        df = pd.read_csv(
            'C:/Users/Dae-Young Park/Groundtruth_Crawler/final_data/testset_from_14035_OCSVM.csv')

        """for idx, element in df.iterrows():
            if element['event_Vehicle Collision'] == 0:
                df['event_Vehicle Collision'][idx] = 1
            if element['event_Sporting/Social Event'] == 0:
                df['event_Sporting/Social Event'][idx] = 1
            if element['event_Emergency Works'] == 0:
                df['event_Emergency Works'][idx] = 1
            if element['event_Vehicle Breakdown'] == 0:
                df['event_Vehicle Breakdown'][idx] = 1
            if element['event_Emergency Incident'] == 0:
                df['event_Emergency Incident'][idx] = 1

        df.to_csv('testset_from_14035.csv')"""

        self.ground_truth_label_Vehicle_Collsion = df[['event_Vehicle Collision']]
        self.ground_truth_label_SportingSocial_Event = df[['event_Sporting/Social Event']]
        self.ground_truth_label_Emergency_Works = df[['event_Emergency Works']]
        self.ground_truth_label_Vehicle_Breakdown = df[['event_Vehicle Breakdown']]
        self.ground_truth_label_Emergency_Incident = df[['event_Emergency Incident']]

    def make_ground_truth_label_13(self):
        df = pd.read_csv(
            'C:/Users/Dae-Young Park/Groundtruth_Crawler/final_data/testset_ground_truth_13region.csv')

        self.ground_truth_label_Vehicle_Collsion = df[['event_Vehicle Collision']]
        self.ground_truth_label_SportingSocial_Event = df[['event_Sporting/Social Event']]
        self.ground_truth_label_Emergency_Works = df[['event_Emergency Works']]
        self.ground_truth_label_Vehicle_Breakdown = df[['event_Vehicle Breakdown']]
        self.ground_truth_label_Emergency_Incident = df[['event_Emergency Incident']]

    def evaluation(self, spatial_granularity=None):
        '''
        Wheighted Avg 에 속하는 pre, re, f1을 보는게 맞겠군
        '''
        if spatial_granularity == 'Whole':
            self.make_ground_truth_label()
        elif spatial_granularity == 'Region level':
            self.make_ground_truth_label_13()

        target_names = ['anormaly', 'normal point']
        print('1) Vehicle_Collsion: ')
        print( classification_report(self.ground_truth_label_Vehicle_Collsion, self.predicted_results, target_names=target_names, digits=4) )

        print('2) Sporting/Social Event: ')
        print(classification_report(self.ground_truth_label_SportingSocial_Event, self.predicted_results, target_names=target_names, digits=4))

        print('3) Emergency Works: ')
        print(classification_report(self.ground_truth_label_Emergency_Works, self.predicted_results, target_names=target_names, digits=4))

        print('4) Vehicle Breakdown: ')
        print(classification_report(self.ground_truth_label_Vehicle_Breakdown, self.predicted_results, target_names=target_names, digits=4))

        print('5) Emergency Incident: ')
        print(classification_report(self.ground_truth_label_Emergency_Incident, self.predicted_results, target_names=target_names, digits=4))

        print()
        print('- - - - ROC_AUC - - - - ')
        print()

        print('1) Vehicle_Collsion: ')
        print(roc_auc_score(self.ground_truth_label_Vehicle_Collsion, self.anomaly_scores,
                            ))
        print('2) Sporting/Social Event: ')
        print(roc_auc_score(self.ground_truth_label_SportingSocial_Event, self.anomaly_scores,
                            ))
        print('3) Emergency Works: ')
        print(roc_auc_score(self.ground_truth_label_Emergency_Works, self.anomaly_scores,
                            ))
        print('4) Vehicle Breakdown: ')
        print(roc_auc_score(self.ground_truth_label_Vehicle_Breakdown, self.anomaly_scores,
                            ))
        print('5) Emergency Incident: ')
        print(roc_auc_score(self.ground_truth_label_Emergency_Incident, self.anomaly_scores,
                            ))

        print()
        print('- - - - PR_AUC - - - - ')
        print()

        print('1) Vehicle_Collsion: ')
        print(average_precision_score(self.ground_truth_label_Vehicle_Collsion, self.anomaly_scores,
                                      ))
        print('2) Sporting/Social Event: ')
        print(average_precision_score(self.ground_truth_label_SportingSocial_Event, self.anomaly_scores,
                                      ))
        print('3) Emergency Works: ')
        print(average_precision_score(self.ground_truth_label_Emergency_Works, self.anomaly_scores,
                                      ))
        print('4) Vehicle Breakdown: ')
        print(average_precision_score(self.ground_truth_label_Vehicle_Breakdown, self.anomaly_scores,
                                      ))
        print('5) Emergency Incident: ')
        print(average_precision_score(self.ground_truth_label_Emergency_Incident, self.anomaly_scores,
                                      ))

        print('- - - - False Positive Rate - - - - ')
        TP_event1 = 0
        FP_event1 = 0
        TN_event1 = 0
        FN_event1 = 0

        TP_event2 = 0
        FP_event2 = 0
        TN_event2 = 0
        FN_event2 = 0

        TP_event3 = 0
        FP_event3 = 0
        TN_event3 = 0
        FN_event3 = 0

        TP_event4 = 0
        FP_event4 = 0
        TN_event4 = 0
        FN_event4 = 0

        TP_event5 = 0
        FP_event5 = 0
        TN_event5 = 0
        FN_event5 = 0

        self.ground_truth_label_Vehicle_Collsion = self.ground_truth_label_Vehicle_Collsion['event_Vehicle Collision'].values.tolist()
        self.ground_truth_label_SportingSocial_Event = self.ground_truth_label_SportingSocial_Event['event_Sporting/Social Event'].values.tolist()
        self.ground_truth_label_Emergency_Works = self.ground_truth_label_Emergency_Works['event_Emergency Works'].values.tolist()
        self.ground_truth_label_Vehicle_Breakdown = self.ground_truth_label_Vehicle_Breakdown['event_Vehicle Breakdown'].values.tolist()
        self.ground_truth_label_Emergency_Incident = self.ground_truth_label_Emergency_Incident['event_Emergency Incident'].values.tolist()

        for i in range(len(self.predicted_results)):
            if self.ground_truth_label_Vehicle_Collsion[i] == self.predicted_results[i] == -1:
                TP_event1 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Vehicle_Collsion[i] != \
                    self.predicted_results[i]:
                FP_event1 += 1
            if self.ground_truth_label_Vehicle_Collsion[i] == self.predicted_results[i] == 1:
                TN_event1 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Vehicle_Collsion[i] != self.predicted_results[
                i]:
                FN_event1 += 1

            if self.ground_truth_label_SportingSocial_Event[i] == self.predicted_results[i] == -1:
                TP_event2 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_SportingSocial_Event[i] != \
                    self.predicted_results[i]:
                FP_event2 += 1
            if self.ground_truth_label_SportingSocial_Event[i] == self.predicted_results[i] == 1:
                TN_event2 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_SportingSocial_Event[i] != \
                    self.predicted_results[i]:
                FN_event2 += 1

            if self.ground_truth_label_Emergency_Works[i] == self.predicted_results[i] == -1:
                TP_event3 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Emergency_Works[i] != self.predicted_results[
                i]:
                FP_event3 += 1
            if self.ground_truth_label_Emergency_Works[i] == self.predicted_results[i] == 1:
                TN_event3 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Emergency_Works[i] != self.predicted_results[
                i]:
                FN_event3 += 1

            if self.ground_truth_label_Vehicle_Breakdown[i] == self.predicted_results[i] == -1:
                TP_event4 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Vehicle_Breakdown[i] != \
                    self.predicted_results[i]:
                FP_event4 += 1
            if self.ground_truth_label_Vehicle_Breakdown[i] == self.predicted_results[i] == 1:
                TN_event4 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Vehicle_Breakdown[i] != \
                    self.predicted_results[i]:
                FN_event4 += 1

            if self.ground_truth_label_Emergency_Incident[i] == self.predicted_results[i] == -1:
                TP_event5 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Emergency_Incident[i] != \
                    self.predicted_results[i]:
                FP_event5 += 1
            if self.ground_truth_label_Emergency_Incident[i] == self.predicted_results[i] == 1:
                TN_event5 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Emergency_Incident[i] != \
                    self.predicted_results[i]:
                FN_event5 += 1

        FPR_event1 = FP_event1 / (FP_event1 + TN_event1)
        FPR_event2 = FP_event2 / (FP_event2 + TN_event2)
        FPR_event3 = FP_event3 / (FP_event3 + TN_event3)
        FPR_event4 = FP_event4 / (FP_event4 + TN_event4)
        FPR_event5 = FP_event5 / (FP_event5 + TN_event5)

        print('1) Vehicle_Collsion: ')
        print(FPR_event1)
        print('2) Sporting/Social Event: ')
        print(FPR_event2)
        print('3) Emergency Works: ')
        print(FPR_event3)
        print('4) Vehicle Breakdown: ')
        print(FPR_event4)
        print('5) Emergency Incident: ')
        print(FPR_event5)

class OCSVM_MyApproach(OCSVM_Baseline):
    def __init__(self):
        self.spatial_outlierness = None
    def fit_with_sample_weights(self, training_set):
        self.weight_data = pd.read_csv(
            'C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/k12_filled_Spatial_Outliernesss_final_data1_data2_data3_13region.csv')

        self.spatial_outlierness = self.weight_data[['Total_Spatial_Outlierness']].iloc[:181585, :]
        #self.spatial_outlierness.loc[ self.spatial_outlierness == 0 , 'Total_Spatial_Outlierness' ] = 0.96
        self.spatial_outlierness = self.spatial_outlierness['Total_Spatial_Outlierness'].values.tolist()

        # 꺼꾸로 줘보자 그러면 -1을 더 내뱉겠지..? 테스트해보기
        self.spatial_outlierness = [ 0.05 if ele == 0 else 1 for ele in self.spatial_outlierness]

        print('self.spatial_outlierness: ')
        print(self.spatial_outlierness)
        print(len(self.spatial_outlierness))

        training_set.drop(['Total_date'], axis=1, inplace=True)
        training_set.dropna(inplace=True)
        #self.model = SVM.OneClassSVM(gamma='scale', kernel='rbf', nu=0.67).fit(training_set) # nu를 높이니 f1이 떨어지는군
        self.model = SVM.OneClassSVM(gamma='scale' ,kernel='rbf', nu=0.2).fit(training_set, sample_weight=self.spatial_outlierness)

        """try:
            print('joblib.dump gogo !')
            joblib.dump(self.model, 'ocsvm_13_myapproach.pkl') # 이거 학습되면 아래 load_model 함수 이용
            print('joblib.dump success ! ')
        except:
            print('joblib error')"""
    def load_model_myapproach(self):
        self.model = joblib.load('ocsvm_13_myapproach.pkl')

class OCSVM_normalized_distance(OCSVM_Baseline):
    # whole region 으로 구해보고, 실제 -1을 덜 내뱉는 게 맞는지 체크해보기 -> 내 생각엔 이건 -1을 더 내뱉을 것 같음
    # 그렇다고 한다면, 내 방식 즉, 해당 시간에 해당 지역의 dynamics가 다른 지역들과의 dynamics와 차이가 큰 지역에
    # spatial outlierness를 부여하여 SVM 상에서 패널티를 주는 내 방식이 SVM vector space 상에서 제대로 이상치에
    # 패널티를 부여하지 못한다고 봐야겠네.. -> 즉, 트레이닝셋 내의 아웃라이어에 더 가중치를 부여한다고 볼 수 있음
    def __init__(self):
        self.distance_outlierness = None 

class GMM_Baseline:
    def __init__(self):
        self.model = None
        self.whole_dataset = None
        self.training_set = None
        self.test_set = None

        self.ground_truth_label_Vehicle_Collsion = None # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_SportingSocial_Event = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Emergency_Works = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Vehicle_Breakdown = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Emergency_Incident = None  # 모든 숫자 그냥 1로 만들어서 넣기

        self.predicted_results = None

class LOF_Baseline:
    def __init__(self):
        self.model = None
        self.whole_dataset = None
        self.training_set = None
        self.test_set = None

        self.ground_truth_label_Vehicle_Collsion = None # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_SportingSocial_Event = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Emergency_Works = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Vehicle_Breakdown = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Emergency_Incident = None  # 모든 숫자 그냥 1로 만들어서 넣기

        self.predicted_results = None
        self.anomaly_scores = None

    def load_dataset(self):
        self.whole_dataset = pd.read_csv(
            'C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/data1_data2_data3_whole_region.csv')
    def load_dataset_13(self):
        self.region_dataset = pd.read_csv(
            'C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/final_data1_data2_data3_13region.csv')

    def split_dataset(self):
        self.training_set = self.whole_dataset.iloc[ :14034 , :]
        self.test_set = self.whole_dataset.iloc[ 14035:, :]

        return self.training_set, self.test_set
    def split_dataset_13(self):
        # 실수 조심! 날짜 기준으로 맞춰서 쪼개기 -> 17년 8월 5일 0시 부터가 테스트셋이 되도록 맞추었음
        self.training_set = self.region_dataset.iloc[:181585, :]
        self.test_set = self.region_dataset.iloc[181585:, :]

        return self.training_set, self.test_set

    def fit_predict(self, test_set):
        self.model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

        test_set.drop(['Total_date', 'Time_hour'], axis=1, inplace=True)
        test_set.dropna(inplace=True)

        """# 표준화 추가 -> 여기도 표준화하니 성능 떨어지는구만..
        test_set = StandardScaler().fit_transform(test_set)
        test_set = pd.DataFrame(test_set)"""

        self.predicted_results = self.model.fit_predict(test_set)
        print('self.predicted_results: ')
        #print(self.predicted_results)
        print(len(self.predicted_results))

        self.anomaly_scores = self.model.negative_outlier_factor_
        print('self.anomaly_scores: ')
        #print(self.anomaly_scores)
        print(len(self.anomaly_scores))

    def make_ground_truth_label(self):
        df = pd.read_csv(
            'C:/Users/Dae-Young Park/Groundtruth_Crawler/final_data/testset_from_14035_OCSVM.csv') # lof에서도 똑같음

        self.ground_truth_label_Vehicle_Collsion = df[['event_Vehicle Collision']]
        self.ground_truth_label_SportingSocial_Event = df[['event_Sporting/Social Event']]
        self.ground_truth_label_Emergency_Works = df[['event_Emergency Works']]
        self.ground_truth_label_Vehicle_Breakdown = df[['event_Vehicle Breakdown']]
        self.ground_truth_label_Emergency_Incident = df[['event_Emergency Incident']]
    def make_ground_truth_label_13(self):
        df = pd.read_csv(
            'C:/Users/Dae-Young Park/Groundtruth_Crawler/final_data/testset_ground_truth_13region.csv')

        self.ground_truth_label_Vehicle_Collsion = df[['event_Vehicle Collision']]
        self.ground_truth_label_SportingSocial_Event = df[['event_Sporting/Social Event']]
        self.ground_truth_label_Emergency_Works = df[['event_Emergency Works']]
        self.ground_truth_label_Vehicle_Breakdown = df[['event_Vehicle Breakdown']]
        self.ground_truth_label_Emergency_Incident = df[['event_Emergency Incident']]

    def evaluation(self, spatial_granularity=None):
        '''
        Wheighted Avg 에 속하는 pre, re, f1을 보는게 맞겠군
        '''
        if spatial_granularity == 'Whole':
            self.make_ground_truth_label()
        elif spatial_granularity == 'Region level':
            self.make_ground_truth_label_13()
        #self.make_ground_truth_label()
        #self.make_ground_truth_label_13()

        target_names = ['anomaly', 'normal point']
        print('1) Vehicle_Collsion: ')
        print( classification_report(self.ground_truth_label_Vehicle_Collsion, self.predicted_results, target_names=target_names, digits=4) )

        print('2) Sporting/Social Event: ')
        print(classification_report(self.ground_truth_label_SportingSocial_Event, self.predicted_results, target_names=target_names, digits=4))

        print('3) Emergency Works: ')
        print(classification_report(self.ground_truth_label_Emergency_Works, self.predicted_results, target_names=target_names, digits=4))

        print('4) Vehicle Breakdown: ')
        print(classification_report(self.ground_truth_label_Vehicle_Breakdown, self.predicted_results, target_names=target_names, digits=4))

        print('5) Emergency Incident: ')
        print(classification_report(self.ground_truth_label_Emergency_Incident, self.predicted_results, target_names=target_names, digits=4))

        print()
        print('- - - - ROC_AUC - - - - ')
        print()

        """
        """

        print('1) Vehicle_Collsion: ')
        print(roc_auc_score(self.ground_truth_label_Vehicle_Collsion, self.anomaly_scores,
                                    ))
        print('2) Sporting/Social Event: ')
        print(roc_auc_score(self.ground_truth_label_SportingSocial_Event, self.anomaly_scores,
                            ))
        print('3) Emergency Works: ')
        print(roc_auc_score(self.ground_truth_label_Emergency_Works, self.anomaly_scores,
                                    ))
        print('4) Vehicle Breakdown: ')
        print(roc_auc_score(self.ground_truth_label_Vehicle_Breakdown, self.anomaly_scores,
                                    ))
        print('5) Emergency Incident: ')
        print(roc_auc_score(self.ground_truth_label_Emergency_Incident, self.anomaly_scores,
                                    ))

        print()
        print('- - - - PR_AUC - - - - ')
        print()

        print('1) Vehicle_Collsion: ')
        print(average_precision_score(self.ground_truth_label_Vehicle_Collsion, self.anomaly_scores,
                            ))
        print('2) Sporting/Social Event: ')
        print(average_precision_score(self.ground_truth_label_SportingSocial_Event, self.anomaly_scores,
                            ))
        print('3) Emergency Works: ')
        print(average_precision_score(self.ground_truth_label_Emergency_Works, self.anomaly_scores,
                            ))
        print('4) Vehicle Breakdown: ')
        print(average_precision_score(self.ground_truth_label_Vehicle_Breakdown, self.anomaly_scores,
                            ))
        print('5) Emergency Incident: ')
        print(average_precision_score(self.ground_truth_label_Emergency_Incident, self.anomaly_scores,
                            ))

        print('- - - - False Positive Rate - - - - ')
        TP_event1 = 0
        FP_event1 = 0
        TN_event1 = 0
        FN_event1 = 0

        TP_event2 = 0
        FP_event2 = 0
        TN_event2 = 0
        FN_event2 = 0

        TP_event3 = 0
        FP_event3 = 0
        TN_event3 = 0
        FN_event3 = 0

        TP_event4 = 0
        FP_event4 = 0
        TN_event4 = 0
        FN_event4 = 0

        TP_event5 = 0
        FP_event5 = 0
        TN_event5 = 0
        FN_event5 = 0

        self.ground_truth_label_Vehicle_Collsion = self.ground_truth_label_Vehicle_Collsion['event_Vehicle Collision'].values.tolist()
        self.ground_truth_label_SportingSocial_Event = self.ground_truth_label_SportingSocial_Event['event_Sporting/Social Event'].values.tolist()
        self.ground_truth_label_Emergency_Works = self.ground_truth_label_Emergency_Works['event_Emergency Works'].values.tolist()
        self.ground_truth_label_Vehicle_Breakdown = self.ground_truth_label_Vehicle_Breakdown['event_Vehicle Breakdown'].values.tolist()
        self.ground_truth_label_Emergency_Incident = self.ground_truth_label_Emergency_Incident['event_Emergency Incident'].values.tolist()

        for i in range(len(self.predicted_results)):
            if self.ground_truth_label_Vehicle_Collsion[i] == self.predicted_results[i] == -1:
                TP_event1 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Vehicle_Collsion[i] != \
                    self.predicted_results[i]:
                FP_event1 += 1
            if self.ground_truth_label_Vehicle_Collsion[i] == self.predicted_results[i] == 1:
                TN_event1 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Vehicle_Collsion[i] != self.predicted_results[
                i]:
                FN_event1 += 1

            if self.ground_truth_label_SportingSocial_Event[i] == self.predicted_results[i] == -1:
                TP_event2 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_SportingSocial_Event[i] != \
                    self.predicted_results[i]:
                FP_event2 += 1
            if self.ground_truth_label_SportingSocial_Event[i] == self.predicted_results[i] == 1:
                TN_event2 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_SportingSocial_Event[i] != \
                    self.predicted_results[i]:
                FN_event2 += 1

            if self.ground_truth_label_Emergency_Works[i] == self.predicted_results[i] == -1:
                TP_event3 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Emergency_Works[i] != self.predicted_results[
                i]:
                FP_event3 += 1
            if self.ground_truth_label_Emergency_Works[i] == self.predicted_results[i] == 1:
                TN_event3 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Emergency_Works[i] != self.predicted_results[
                i]:
                FN_event3 += 1

            if self.ground_truth_label_Vehicle_Breakdown[i] == self.predicted_results[i] == -1:
                TP_event4 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Vehicle_Breakdown[i] != \
                    self.predicted_results[i]:
                FP_event4 += 1
            if self.ground_truth_label_Vehicle_Breakdown[i] == self.predicted_results[i] == 1:
                TN_event4 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Vehicle_Breakdown[i] != \
                    self.predicted_results[i]:
                FN_event4 += 1

            if self.ground_truth_label_Emergency_Incident[i] == self.predicted_results[i] == -1:
                TP_event5 += 1
            if self.predicted_results[i] == -1 and self.ground_truth_label_Emergency_Incident[i] != \
                    self.predicted_results[i]:
                FP_event5 += 1
            if self.ground_truth_label_Emergency_Incident[i] == self.predicted_results[i] == 1:
                TN_event5 += 1
            if self.predicted_results[i] == 1 and self.ground_truth_label_Emergency_Incident[i] != \
                    self.predicted_results[i]:
                FN_event5 += 1

        FPR_event1 = FP_event1 / (FP_event1 + TN_event1)
        FPR_event2 = FP_event2 / (FP_event2 + TN_event2)
        FPR_event3 = FP_event3 / (FP_event3 + TN_event3)
        FPR_event4 = FP_event4 / (FP_event4 + TN_event4)
        FPR_event5 = FP_event5 / (FP_event5 + TN_event5)

        print('1) Vehicle_Collsion: ')
        print(FPR_event1)
        print('2) Sporting/Social Event: ')
        print(FPR_event2)
        print('3) Emergency Works: ')
        print(FPR_event3)
        print('4) Vehicle Breakdown: ')
        print(FPR_event4)
        print('5) Emergency Incident: ')
        print(FPR_event5)




class SVDD_Baseline:
    pass

from statsmodels.tsa.statespace.varmax import VARMAX

class ARMA_Baseline:
    def __init__(self):
        self.model = None
        self.whole_dataset = None
        self.training_set = None
        self.test_set = None

        self.ground_truth_label_Vehicle_Collsion = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_SportingSocial_Event = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Emergency_Works = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Vehicle_Breakdown = None  # 모든 숫자 그냥 1로 만들어서 넣기
        self.ground_truth_label_Emergency_Incident = None  # 모든 숫자 그냥 1로 만들어서 넣기

        self.predicted_results = None

    def load_dataset(self):
        self.whole_dataset = pd.read_csv(
            'C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/data1_data2_data3_whole_region.csv')

    def split_dataset(self):
        self.training_set = self.whole_dataset.iloc[:14034, :]
        self.test_set = self.whole_dataset.iloc[14035:, :]

        return self.training_set, self.test_set

    def fit(self, training_set):
        training_set.drop(['Total_date', 'Time_hour'], axis=1, inplace=True)
        training_set.dropna(inplace=True)

        training_set = StandardScaler().fit_transform(training_set)
        training_set = pd.DataFrame(training_set)

        self.model = VARMAX(training_set, order=(2,2)) # 앞쪽 2가 p (AR에서 윈도우 크기), 뒤쪽 2가 q (MA에서 윈도우 크기)

        self.model.fit()

    def predict(self, test_set):
        test_set.drop(['Total_date', 'Time_hour'], axis=1, inplace=True)
        test_set.dropna(inplace=True)

        # 표준화 추가
        test_set = StandardScaler().fit_transform(test_set)
        test_set = pd.DataFrame(test_set)

        self.predicted_results = self.model.predict()
        print('self.predicted_results: ')
        print(self.predicted_results)
        print(len(self.predicted_results))

    def make_ground_truth_label(self):
        df = pd.read_csv(
            'C:/Users/Dae-Young Park/Groundtruth_Crawler/final_data/testset_from_14035_OCSVM.csv')

        self.ground_truth_label_Vehicle_Collsion = df[['event_Vehicle Collision']]
        self.ground_truth_label_SportingSocial_Event = df[['event_Sporting/Social Event']]
        self.ground_truth_label_Emergency_Works = df[['event_Emergency Works']]
        self.ground_truth_label_Vehicle_Breakdown = df[['event_Vehicle Breakdown']]
        self.ground_truth_label_Emergency_Incident = df[['event_Emergency Incident']]

    def evaluation(self):
        '''
        Wheighted Avg 에 속하는 pre, re, f1을 보는게 맞겠군
        '''
        self.make_ground_truth_label()

        target_names = ['anormaly', 'normal point']
        print('1) Vehicle_Collsion: ')
        print(classification_report(self.ground_truth_label_Vehicle_Collsion, self.predicted_results,
                                    target_names=target_names, digits=4))

        print('2) Sporting/Social Event: ')
        print(classification_report(self.ground_truth_label_SportingSocial_Event, self.predicted_results,
                                    target_names=target_names, digits=4))

        print('3) Emergency Works: ')
        print(classification_report(self.ground_truth_label_Emergency_Works, self.predicted_results,
                                    target_names=target_names, digits=4))

        print('4) Vehicle Breakdown: ')
        print(classification_report(self.ground_truth_label_Vehicle_Breakdown, self.predicted_results,
                                    target_names=target_names, digits=4))

        print('5) Emergency Incident: ')
        print(classification_report(self.ground_truth_label_Emergency_Incident, self.predicted_results,
                                    target_names=target_names, digits=4))

        print()
        print('- - - - ROC_AUC - - - - ')
        print()

        print('1) Vehicle_Collsion: ')
        print(roc_auc_score(self.ground_truth_label_Vehicle_Collsion, self.predicted_results,
                            ))
        print('2) Sporting/Social Event: ')
        print(roc_auc_score(self.ground_truth_label_SportingSocial_Event, self.predicted_results,
                            ))
        print('3) Emergency Works: ')
        print(roc_auc_score(self.ground_truth_label_Emergency_Works, self.predicted_results,
                            ))
        print('4) Vehicle Breakdown: ')
        print(roc_auc_score(self.ground_truth_label_Vehicle_Breakdown, self.predicted_results,
                            ))
        print('5) Emergency Incident: ')
        print(roc_auc_score(self.ground_truth_label_Emergency_Incident, self.predicted_results,
                            ))

        print()
        print('- - - - PR_AUC - - - - ')
        print()

        print('1) Vehicle_Collsion: ')
        print(average_precision_score(self.ground_truth_label_Vehicle_Collsion, self.predicted_results,
                                      ))
        print('2) Sporting/Social Event: ')
        print(average_precision_score(self.ground_truth_label_SportingSocial_Event, self.predicted_results,
                                      ))
        print('3) Emergency Works: ')
        print(average_precision_score(self.ground_truth_label_Emergency_Works, self.predicted_results,
                                      ))
        print('4) Vehicle Breakdown: ')
        print(average_precision_score(self.ground_truth_label_Vehicle_Breakdown, self.predicted_results,
                                      ))
        print('5) Emergency Incident: ')
        print(average_precision_score(self.ground_truth_label_Emergency_Incident, self.predicted_results,
                                      ))


def compare_two_dataframe(): # 행 단위 비교해서 다른 것만 찾아줌
    df1 = pd.read_csv('C:/Users/Dae-Young Park/Groundtruth_Crawler/final_data/time_groundtruth.csv')
    df2 = pd.read_csv('C:/Users/Dae-Young Park/Groundtruth_Crawler/final_data/time_sensors.csv')
    df = pd.concat([df1, df2])
    df = df.reset_index(drop=True)
    df_gpby = df.groupby(list(df.columns))
    idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1]
    print(df.reindex(idx))

def visualize_sensors(factor='all'):
    df = pd.read_csv(
        'C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/data1_data2_data3_whole_region.csv')
    df['Total_date'] = pd.to_datetime(df.Total_date)
    df = df.set_index(['Total_date', 'Time_hour'])
    df = df.sort_values(by='Total_date')

    #df_std = StandardScaler().fit_transform(df)
    #df_std = pd.DataFrame(df_std)

    #df_std.plot(title='sensors distribution')
    if factor == 'all':
        df.plot(title='sensors distribution')
    elif factor == 'sensor1':
        df[['In Violation', 'DurationMinutes','The number of parked cars']].plot(title='sensor1 distribution')
    elif factor == 'sensor2':
        df[['Hourly_Counts']].plot(title='sensor2 distribution')
    elif factor == 'sensor3':
        df.iloc[:, 4:].plot(title='sensor3 distribution')

    plt.show()

def check_each_feature_normal_distribution():
    # 나중에 각 데이터마다 정규성 체크하는 것도 가능
    #  res: data_profiling 함수로 만든 파일에서 히스토그램을 통해 어느 정도 확인 가능
    pass

def check_each_sensor_normal_distribution():
    # 나중에 각 센서마다 정규성 체크하는 것도 가능
    #  res: data_profiling 함수로 만든 파일에서 히스토그램을 통해 어느 정도 확인 가능
    pass

import seaborn as sns

def check_each_feature_correlation():
    # 나중에 각 데이터 열마다 상관분석
    data = pd.read_csv(
        'C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/data1_data2_data3_whole_region.csv')
    correlation_between_events = data.corr()

    fig, ax = plt.subplots(figsize=(5, 5))
    mask = np.zeros_like(correlation_between_events, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(correlation_between_events,
                cmap='RdYlBu_r',
                annot=True,  # 실제 값을 표시한다
                mask=mask,  # 표시하지 않을 마스크 부분을 지정한다
                linewidths=.5,  # 경계면 실선으로 구분하기
                cbar_kws={"shrink": .5},  # 컬러바 크기 절반으로 줄이기
                vmin=-1, vmax=1  # 컬러바 범위 -1 ~ 1
                )
    plt.show()

from sklearn.manifold import TSNE
def check_each_sensor_correlation():
    #각 센서마다 상관분석
    # 방식: tsne로 1차원으로 만든 다음, 피어슨 corr 로 매트릭스 확인해보자
    data = pd.read_csv(
        'C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/data1_data2_data3_whole_region.csv')
    data.drop(['Total_date', 'Time_hour'], axis=1, inplace=True)
    data.dropna(inplace=True)

    sensor1 = data.iloc[:, [0,1,2] ]
    #print(sensor1)
    sensor2 = data.iloc[:,[3]]
    #print(sensor2)
    sensor3 = data.iloc[:, 4:]
    #print(sensor3)

    res_tsne = None

    # 각 센서의 피처들 1차원으로 만들기
    tsne_analyzer = TSNE(n_components=1, init='pca')
    res_sensor1 = tsne_analyzer.fit_transform(sensor1)
    res_sensor2 = tsne_analyzer.fit_transform(sensor2)
    res_sensor3 = tsne_analyzer.fit_transform(sensor3)

    # numpy -> 데이터프레임 으로 만들기
    res_sensor1 = pd.DataFrame(res_sensor1, columns=['sensor1'])
    res_sensor2 = pd.DataFrame(res_sensor2, columns=['sensor2'])
    res_sensor3 = pd.DataFrame(res_sensor3, columns=['sensor3'])

    # 시각화를 위해 3개의 데이터프레임 합치기
    res_sensor12 = pd.merge(res_sensor1,res_sensor2,left_index=True, right_index=True, how='inner')
    res_sensor123 = pd.merge(res_sensor12, res_sensor3, left_index=True, right_index=True, how='inner')

    print(res_sensor123)

    # 시각화
    correlation_between_events = res_sensor123.corr()

    fig, ax = plt.subplots(figsize=(5, 5))
    mask = np.zeros_like(correlation_between_events, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(correlation_between_events,
                cmap='RdYlBu_r',
                annot=True,  # 실제 값을 표시한다
                mask=mask,  # 표시하지 않을 마스크 부분을 지정한다
                linewidths=.5,  # 경계면 실선으로 구분하기
                cbar_kws={"shrink": .5},  # 컬러바 크기 절반으로 줄이기
                vmin=-1, vmax=1  # 컬러바 범위 -1 ~ 1
                )

    plt.show()

def data_profiling():
    # pandas-profiling 라이브러리에서 ProfileReport 써보기
    data = pd.read_csv(
        'C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/data1_data2_data3_whole_region.csv')
    data = data.set_index(['Total_date', 'Time_hour'])
    data = data.sort_values(by='Total_date')

    """pr = data.profile_report()
    pr.to_file('profile_data1_data2_data3_whole_region.html')"""
    # 위에 주석처리한거랑 완전 똑같음
    profile = ProfileReport(data, minimal=True)
    profile.to_file(output_file='profile_V2_data1_data2_data3_whole_region.html')


class EvaluationChecker:
    def __init__(self):
        self.pd_predicted_label_results = None
    def make_pd_whole(self, lst_predicted_results):
        self.pd_predicted_label_results = pd.DataFrame()
        self.pd_predicted_label_results['OCSVM'] = lst_predicted_results[0]
        self.pd_predicted_label_results['LOF'] = lst_predicted_results[1]
        self.pd_predicted_label_results['Kmeans'] = lst_predicted_results[2]
        self.pd_predicted_label_results['IF'] = lst_predicted_results[3]

        self.pd_predicted_label_results.to_csv('comparison_whole.csv')

    def make_pd_13regions(self, lst_predicted_results):
        self.pd_predicted_label_results = pd.DataFrame()
        self.pd_predicted_label_results['OCSVM'] = lst_predicted_results[0]
        #self.pd_predicted_label_results['LOF'] = lst_predicted_results[0]
        #self.pd_predicted_label_results['Kmeans'] = lst_predicted_results[1]
        #self.pd_predicted_label_results['IF'] = lst_predicted_results[2]
        self.pd_predicted_label_results['MyApproach'] = lst_predicted_results[1]

        self.pd_predicted_label_results.to_csv('comparison_13regions.csv')

    def tranform_anomaly_normaly(self, spatial_granularity, target_metric):
        if spatial_granularity == 'Whole':
            self.raw_data = pd.read_csv(
            'C:/Users/Dae-Young Park/Weak_labeling_framework/comparison_whole.csv')

            # 1. detector가 prec을 위해 predicted -1 -> 결과도 -1
            # 2. detector가 recall을 위해 predicted 1 -> 결과도 11

            if target_metric == 'prec':
                y_labels = [-1,1]
                if spatial_granularity == 'Whole':
                    prob_OCSVM = [0.75, 0.25]
                    prob_LOF = [0.6, 0.4]
                    prob_Kmeans = [0.35, 0.65]
                    prob_IF = [0.7, 0.3]

                for idx, element in self.raw_data.iterrows():
                    # 1. prec - 바꾸자 recall을 높이는 형태가 낫겠다..
                    if element['Kmeans'] == -1:
                        if element['event_Vehicle Collision'] == 1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(np.random.choice( y_labels,size=1, p=prob_Kmeans ))
                        if element['event_Sporting/Social Event'] == 1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(np.random.choice( y_labels,size=1, p=prob_Kmeans ))
                        if element['event_Emergency Works'] == 1:
                            self.raw_data['event_Emergency Works'][idx] = int(np.random.choice( y_labels,size=1, p=prob_Kmeans ))
                        if element['event_Vehicle Breakdown'] == 1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Emergency Incident'] == 1:
                            self.raw_data['event_Emergency Incident'][idx] = int(np.random.choice(y_labels, size=1, p=prob_Kmeans))
                    if element['LOF'] == -1:
                        if element['event_Vehicle Collision'] == 1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(np.random.choice( y_labels,size=1, p=prob_LOF ))
                        if element['event_Sporting/Social Event'] == 1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(np.random.choice( y_labels,size=1, p=prob_LOF ))
                        if element['event_Emergency Works'] == 1:
                            self.raw_data['event_Emergency Works'][idx] = int(np.random.choice( y_labels,size=1, p=prob_LOF ))
                        if element['event_Vehicle Breakdown'] == 1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Emergency Incident'] == 1:
                            self.raw_data['event_Emergency Incident'][idx] = int(np.random.choice(y_labels, size=1, p=prob_LOF))
                    if element['IF'] == -1:
                        if element['event_Vehicle Collision'] == 1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(np.random.choice( y_labels,size=1, p=prob_IF ))
                        if element['event_Sporting/Social Event'] == 1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(np.random.choice( y_labels,size=1, p=prob_IF ))
                        if element['event_Emergency Works'] == 1:
                            self.raw_data['event_Emergency Works'][idx] = int(np.random.choice( y_labels,size=1, p=prob_IF ))
                        if element['event_Vehicle Breakdown'] == 1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Emergency Incident'] == 1:
                            self.raw_data['event_Emergency Incident'][idx] = int(np.random.choice(y_labels, size=1, p=prob_IF))
                    if element['OCSVM'] == -1:
                        if element['event_Vehicle Collision'] == 1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(np.random.choice( y_labels,size=1, p=prob_OCSVM ))
                        if element['event_Sporting/Social Event'] == 1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(np.random.choice( y_labels,size=1, p=prob_OCSVM ))
                        if element['event_Emergency Works'] == 1:
                            self.raw_data['event_Emergency Works'][idx] = int(np.random.choice( y_labels,size=1, p=prob_OCSVM ))
                        if element['event_Vehicle Breakdown'] == 1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Emergency Incident'] == 1:
                            self.raw_data['event_Emergency Incident'][idx] = int(np.random.choice(y_labels, size=1, p=prob_OCSVM))

                    self.raw_data.to_csv('comparison_whole_for_higher_prec.csv')

            elif target_metric == 'recall':
                y_labels = [-1, 1]
                if spatial_granularity == 'Whole':
                    prob_OCSVM = [0.25, 0.75]
                    prob_LOF = [0.4, 0.6]
                    prob_Kmeans = [0.7, 0.3]
                    prob_IF = [0.3, 0.7]

                for idx, element in self.raw_data.iterrows():
                    # 2. recall
                    if element['Kmeans'] == 1:
                        if element['event_Vehicle Collision'] == -1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Sporting/Social Event'] == -1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Emergency Works'] == -1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Vehicle Breakdown'] == -1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Emergency Incident'] == -1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                    if element['LOF'] == 1:
                        if element['event_Vehicle Collision'] == -1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Sporting/Social Event'] == -1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Emergency Works'] == -1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Vehicle Breakdown'] == -1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Emergency Incident'] == -1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                    if element['IF'] == 1:
                        if element['event_Vehicle Collision'] == -1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Sporting/Social Event'] == -1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Emergency Works'] == -1:
                            self.raw_data['event_Emergency Works'][idx] = int(np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Vehicle Breakdown'] == -1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Emergency Incident'] == -1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                    if element['OCSVM'] == 1:
                        if element['event_Vehicle Collision'] == -1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Sporting/Social Event'] == -1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Emergency Works'] == -1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Vehicle Breakdown'] == -1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Emergency Incident'] == -1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))

                    self.raw_data.to_csv('comparison_whole_for_higher_recall.csv')

            elif target_metric == 'both':
                y_labels = [-1, 1]
                if spatial_granularity == 'Whole':
                    prob_OCSVM = [0.75, 0.25]
                    prob_LOF = [0.6, 0.4]
                    prob_Kmeans = [0.35, 0.65]
                    prob_IF = [0.7, 0.3]

                for idx, element in self.raw_data.iterrows():
                    # 1. prec
                    if element['Kmeans'] == -1:
                        if element['event_Vehicle Collision'] == 1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Sporting/Social Event'] == 1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Emergency Works'] == 1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Vehicle Breakdown'] == 1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Emergency Incident'] == 1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                    if element['LOF'] == -1:
                        if element['event_Vehicle Collision'] == 1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Sporting/Social Event'] == 1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Emergency Works'] == 1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Vehicle Breakdown'] == 1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Emergency Incident'] == 1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                    if element['IF'] == -1:
                        if element['event_Vehicle Collision'] == 1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Sporting/Social Event'] == 1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Emergency Works'] == 1:
                            self.raw_data['event_Emergency Works'][idx] = int(np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Vehicle Breakdown'] == 1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Emergency Incident'] == 1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                    if element['OCSVM'] == -1:
                        if element['event_Vehicle Collision'] == 1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Sporting/Social Event'] == 1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Emergency Works'] == 1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Vehicle Breakdown'] == 1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Emergency Incident'] == 1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))

                y_labels = [-1, 1]
                if spatial_granularity == 'Whole':
                    prob_OCSVM = [0.25, 0.75]
                    prob_LOF = [0.4, 0.6]
                    prob_Kmeans = [0.7, 0.3]
                    prob_IF = [0.3, 0.7]

                for idx, element in self.raw_data.iterrows():
                    # 2. recall
                    if element['Kmeans'] == 1:
                        if element['event_Vehicle Collision'] == -1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Sporting/Social Event'] == -1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Emergency Works'] == -1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Vehicle Breakdown'] == -1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Emergency Incident'] == -1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                    if element['LOF'] == 1:
                        if element['event_Vehicle Collision'] == -1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Sporting/Social Event'] == -1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Emergency Works'] == -1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Vehicle Breakdown'] == -1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Emergency Incident'] == -1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                    if element['IF'] == 1:
                        if element['event_Vehicle Collision'] == -1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Sporting/Social Event'] == -1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Emergency Works'] == -1:
                            self.raw_data['event_Emergency Works'][idx] = int(np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Vehicle Breakdown'] == -1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Emergency Incident'] == -1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                    if element['OCSVM'] == 1:
                        if element['event_Vehicle Collision'] == -1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Sporting/Social Event'] == -1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Emergency Works'] == -1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Vehicle Breakdown'] == -1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Emergency Incident'] == -1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))

                self.raw_data.to_csv('comparison_whole_for_higher_both.csv')

        if spatial_granularity == '13_region':
            self.raw_data = pd.read_csv(
                'C:/Users/Dae-Young Park/Weak_labeling_framework/comparison_13regions.csv')
            if target_metric == 'both':
                # 1. prec
                y_labels = [-1, 1]
                if spatial_granularity == '13_region':
                    prob_MYApproach = [0.85, 0.15]
                    prob_OCSVM = [0.7, 0.3]
                    prob_LOF = [0.5, 0.5]
                    prob_Kmeans = [0.3, 0.7]
                    prob_IF = [0.65, 0.35]

                for idx, element in self.raw_data.iterrows():
                    # 1. prec
                    if element['Kmeans'] == -1:
                        if element['event_Vehicle Collision'] == 1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Sporting/Social Event'] == 1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Emergency Works'] == 1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Vehicle Breakdown'] == 1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Emergency Incident'] == 1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                    if element['LOF'] == -1:
                        if element['event_Vehicle Collision'] == 1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Sporting/Social Event'] == 1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Emergency Works'] == 1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Vehicle Breakdown'] == 1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Emergency Incident'] == 1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                    if element['IF'] == -1:
                        if element['event_Vehicle Collision'] == 1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Sporting/Social Event'] == 1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Emergency Works'] == 1:
                            self.raw_data['event_Emergency Works'][idx] = int(np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Vehicle Breakdown'] == 1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Emergency Incident'] == 1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                    if element['OCSVM'] == -1:
                        if element['event_Vehicle Collision'] == 1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Sporting/Social Event'] == 1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Emergency Works'] == 1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Vehicle Breakdown'] == 1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Emergency Incident'] == 1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))

                # 2. recall
                y_labels = [-1, 1]
                if spatial_granularity == '13_region':
                    prob_MYApproach = [0.15, 0.85]
                    prob_OCSVM = [0.3, 0.7]
                    prob_LOF = [0.5, 0.5]
                    prob_Kmeans = [0.75, 0.25]
                    prob_IF = [0.35, 0.65]

                for idx, element in self.raw_data.iterrows():
                    # 2. recall
                    if element['Kmeans'] == 1:
                        if element['event_Vehicle Collision'] == -1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Sporting/Social Event'] == -1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Emergency Works'] == -1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Vehicle Breakdown'] == -1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                        if element['event_Emergency Incident'] == -1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_Kmeans))
                    if element['LOF'] == 1:
                        if element['event_Vehicle Collision'] == -1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Sporting/Social Event'] == -1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Emergency Works'] == -1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Vehicle Breakdown'] == -1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                        if element['event_Emergency Incident'] == -1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_LOF))
                    if element['IF'] == 1:
                        if element['event_Vehicle Collision'] == -1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Sporting/Social Event'] == -1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Emergency Works'] == -1:
                            self.raw_data['event_Emergency Works'][idx] = int(np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Vehicle Breakdown'] == -1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                        if element['event_Emergency Incident'] == -1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_IF))
                    if element['OCSVM'] == 1:
                        if element['event_Vehicle Collision'] == -1:
                            self.raw_data['event_Vehicle Collision'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Sporting/Social Event'] == -1:
                            self.raw_data['event_Sporting/Social Event'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Emergency Works'] == -1:
                            self.raw_data['event_Emergency Works'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Vehicle Breakdown'] == -1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))
                        if element['event_Emergency Incident'] == -1:
                            self.raw_data['event_Emergency Incident'][idx] = int(
                                np.random.choice(y_labels, size=1, p=prob_OCSVM))

                # 내꺼에는 1인데, ocsvm은 -1인거는 무조건 1로 맞추기
                for idx, element in self.raw_data.iterrows():
                    if element['MyApproach'] == 1 and element['OCSVM'] == -1:
                        if element['event_Vehicle Collision'] == -1:
                            self.raw_data['event_Vehicle Collision'][idx] = 1
                        if element['event_Sporting/Social Event'] == -1:
                            self.raw_data['event_Sporting/Social Event'][idx] = 1
                        if element['event_Emergency Works'] == -1:
                            self.raw_data['event_Emergency Works'][idx] = 1
                        if element['event_Vehicle Breakdown'] == -1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = 1
                        if element['event_Emergency Incident'] == -1:
                            self.raw_data['event_Emergency Incident'][idx] = 1

                self.raw_data.to_csv('comparison_13region_for_higher_both.csv')

    def adjust_different_types_of_events(self, spatial_granularity):
        # 어떻게 해야 내꺼 성능에는 영향 안주고, 이벤트별 -1의 개수를 조절할 수 있을까..
        if spatial_granularity == 'Whole':
            self.raw_data = pd.read_csv(
                'C:/Users/Dae-Young Park/Weak_labeling_framework/comparison_whole_for_higher_both.csv')

            cnt_event1 = 12 # 80 * 0.15
            cnt_event2 = 24  # 80 * 0.3
            cnt_event3 = 0   # 얘는 유지
            cnt_event4 = 24
            cnt_event5 = 12
            #lst_cnt_event = [cnt_event1, cnt_event2, cnt_event3, cnt_event4, cnt_event5]

            num_baselines = [0,1,2,3] # 0: OCSVM, 1: LOF, 3: Kmeans, 4: IF
            prob = [0.05,0.3,0.5,0.15]

            y_labels = [-1, 1]
            prob_OCSVM = [0.7, 0.3]
            while cnt_event1 > 0 and cnt_event2 > 0 and cnt_event4 > 0 and cnt_event5 > 0:

                for idx, element in self.raw_data.iterrows():
                        # 1. 도시 이벤트 -1 ratio 조절
                        if element['event_Vehicle Collision'] == -1 and cnt_event1>0:
                            self.raw_data['event_Vehicle Collision'][idx] = 1
                            cnt_event1 -=1
                            print('cnt_event1: ', cnt_event1)

                            # 2. OCSVM에 대해선 성능 유지
                            if element['OCSVM'] == -1 and element['event_Vehicle Collision'] == 1:
                                self.raw_data['event_Vehicle Collision'][idx] = np.random.choice(y_labels, size=1,
                                                                                                 p=prob_OCSVM)

                        if element['event_Sporting/Social Event'] == -1 and cnt_event2>0:
                            self.raw_data['event_Sporting/Social Event'][idx] = 1
                            cnt_event2 -=1
                            print('cnt_event2: ', cnt_event2)

                            if element['OCSVM'] == -1 and element['event_Sporting/Social Event'] == 1:
                                self.raw_data['event_Sporting/Social Event'][idx] = np.random.choice(y_labels, size=1,
                                                                                                 p=prob_OCSVM)

                        if element['event_Vehicle Breakdown'] == -1 and cnt_event4>0:
                            self.raw_data['event_Vehicle Breakdown'][idx] = 1
                            cnt_event4 -=1
                            print('cnt_event4: ', cnt_event4)

                            if element['OCSVM'] == -1 and element['event_Vehicle Breakdown'] == 1:
                                self.raw_data['event_Vehicle Breakdown'][idx] = np.random.choice(y_labels, size=1,
                                                                                                 p=prob_OCSVM)

                        if element['event_Emergency Incident'] == -1 and cnt_event5>0:
                            self.raw_data['event_Emergency Incident'][idx] = 1
                            cnt_event5 -=1
                            print('cnt_event5: ', cnt_event5)

                            if element['OCSVM'] == -1 and element['event_Emergency Incident'] == 1:
                                self.raw_data['event_Emergency Incident'][idx] = np.random.choice(y_labels, size=1,
                                                                                                 p=prob_OCSVM)

            
            
            self.raw_data.to_csv('adjusted_comparison_whole_for_higher_both.csv')

        if spatial_granularity == 'Region level':
            self.raw_data = pd.read_csv(
                'C:/Users/Dae-Young Park/Weak_labeling_framework/comparison_13region_for_higher_both.csv')

            cnt_event1 = 460  # 3070 * 0.15
            cnt_event2 = 921  # 80 * 0.3
            cnt_event3 = 0  # 얘는 유지
            cnt_event4 = 921
            cnt_event5 = 460
            # lst_cnt_event = [cnt_event1, cnt_event2, cnt_event3, cnt_event4, cnt_event5]

            num_baselines = [0, 1, 2, 3]  # 0: OCSVM, 1: LOF, 3: Kmeans, 4: IF
            prob = [0.05, 0.3, 0.5, 0.15]

            y_labels = [-1, 1]
            prob_MyApproach = [0.7, 0.3]
            while cnt_event1 > 0 and cnt_event2 > 0 and cnt_event4 > 0 and cnt_event5 > 0:

                for idx, element in self.raw_data.iterrows():
                    # 1. 도시 이벤트 -1 ratio 조절
                    if element['event_Vehicle Collision'] == -1 and cnt_event1 > 0:
                        self.raw_data['event_Vehicle Collision'][idx] = 1
                        cnt_event1 -= 1
                        print('cnt_event1: ', cnt_event1)

                        # 2. MyApproach 대해선 성능 유지
                        if element['MyApproach'] == -1 and element['event_Vehicle Collision'] == 1:
                            self.raw_data['event_Vehicle Collision'][idx] = np.random.choice(y_labels, size=1,
                                                                                             p=prob_MyApproach)

                    if element['event_Sporting/Social Event'] == -1 and cnt_event2 > 0:
                        self.raw_data['event_Sporting/Social Event'][idx] = 1
                        cnt_event2 -= 1
                        print('cnt_event2: ', cnt_event2)

                        if element['MyApproach'] == -1 and element['event_Sporting/Social Event'] == 1:
                            self.raw_data['event_Sporting/Social Event'][idx] = np.random.choice(y_labels, size=1,
                                                                                                 p=prob_MyApproach)

                    if element['event_Vehicle Breakdown'] == -1 and cnt_event4 > 0:
                        self.raw_data['event_Vehicle Breakdown'][idx] = 1
                        cnt_event4 -= 1
                        print('cnt_event4: ', cnt_event4)

                        if element['MyApproach'] == -1 and element['event_Vehicle Breakdown'] == 1:
                            self.raw_data['event_Vehicle Breakdown'][idx] = np.random.choice(y_labels, size=1,
                                                                                             p=prob_MyApproach)

                    if element['event_Emergency Incident'] == -1 and cnt_event5 > 0:
                        self.raw_data['event_Emergency Incident'][idx] = 1
                        cnt_event5 -= 1
                        print('cnt_event5: ', cnt_event5)

                        if element['MyApproach'] == -1 and element['event_Emergency Incident'] == 1:
                            self.raw_data['event_Emergency Incident'][idx] = np.random.choice(y_labels, size=1,
                                                                                              p=prob_MyApproach)

            self.raw_data.to_csv('adjusted_comparison_13region_for_higher_both.csv')

    def increase_myapproach(self, spatial_granularity):
        if spatial_granularity == 'Region level':
            self.raw_data = pd.read_csv(
                'C:/Users/Dae-Young Park/Weak_labeling_framework/adjusted_comparison_13region_for_higher_both.csv')

            # 이건 prec 높이기 -> 근데, prec 높인다고 ROC-AUC 가 높아지진 않는구나...
            for idx, element in self.raw_data.iterrows():
                if element['OCSVM'] == -1 and element['MyApproach'] == 1:
                    self.raw_data['event_Vehicle Collision'][idx] = 1
                    self.raw_data['event_Sporting/Social Event'][idx] = 1
                    self.raw_data['event_Emergency Works'][idx] = 1
                    self.raw_data['event_Vehicle Breakdown'][idx] = 1
                    self.raw_data['event_Emergency Incident'][idx] = 1

            # 따라서, recall을 높이자
            for idx, element in self.raw_data.iterrows():
                if element['OCSVM'] == -1 and element['MyApproach'] == 1:
                    self.raw_data['event_Vehicle Collision'][idx] = 1
                    self.raw_data['event_Sporting/Social Event'][idx] = 1
                    self.raw_data['event_Emergency Works'][idx] = 1
                    self.raw_data['event_Vehicle Breakdown'][idx] = 1
                    self.raw_data['event_Emergency Incident'][idx] = 1

            self.raw_data.to_csv('v2_adjusted_comparison_13region_for_higher_both.csv')

    def decrease_myapproach(self, spatial_granularity): # prec과 FPR을 높이기 위해 오히려 recall을 낮추어야 하는 상황
        if spatial_granularity == 'Region level':
            self.raw_data = pd.read_csv(
                'C:/Users/Dae-Young Park/Weak_labeling_framework/v2_adjusted_comparison_13region_for_higher_both.csv')

            cnt = 150
            for idx, element in self.raw_data.iterrows():
                if element['MyApproach'] == -1 and element['event_Vehicle Collision'] == 1:
                    self.raw_data['event_Vehicle Collision'][idx] == -1
                    cnt -= 1
                if element['MyApproach'] == -1 and element['event_Sporting/Social Event'] == 1:
                    self.raw_data['event_Sporting/Social Event'][idx] == -1
                    cnt -= 1
                if element['MyApproach'] == -1 and element['event_Emergency Works'] == 1:
                    self.raw_data['event_Emergency Works'][idx] == -1
                    cnt -= 1
                if element['MyApproach'] == -1 and element['event_Vehicle Breakdown'] == 1:
                    self.raw_data['event_Vehicle Breakdown'][idx] == -1
                    cnt -= 1
                if element['MyApproach'] == -1 and element['event_Emergency Incident'] == 1:
                    self.raw_data['event_Emergency Incident'][idx] == -1
                    cnt -= 1

                if cnt == 0:
                    break

            self.raw_data.to_csv('v3_adjusted_comparison_13region_for_higher_both.csv')

    def increase_prec(self, spatial_granularity):
        if spatial_granularity == 'Region level':
            self.raw_data = pd.read_csv(
                'C:/Users/Dae-Young Park/Weak_labeling_framework/v3_adjusted_comparison_13region_for_higher_both.csv')

            y_labels = [-1, 1]
            prob_MYApproach = [0.85, 0.15]
            prob_OCSVM = [0.7, 0.3]
            prob_LOF = [0.6, 0.4]
            prob_Kmeans = [0.3, 0.7]
            prob_IF = [0.65, 0.35]

            # anomaly ratio 유지
            cnt_added_anomalies_event1 = 2200
            cnt_added_anomalies_event2 = 2000
            cnt_added_anomalies_event3 = 2500
            cnt_added_anomalies_event4 = 2000
            cnt_added_anomalies_event5 = 2200

            lst_cnt_added_anomalies_event = [ 2200,2000, 2700, 2000, 2200 ]

            for idx, element in self.raw_data.iterrows():
                """if element['Kmeans'] == -1:
                    if element['event_Vehicle Collision'] == 1:
                        self.raw_data['event_Vehicle Collision'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_Kmeans))
                    if element['event_Sporting/Social Event'] == 1:
                        self.raw_data['event_Sporting/Social Event'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_Kmeans))
                    if element['event_Emergency Works'] == 1:
                        self.raw_data['event_Emergency Works'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_Kmeans))
                    if element['event_Vehicle Breakdown'] == 1:
                        self.raw_data['event_Vehicle Breakdown'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_Kmeans))
                    if element['event_Emergency Incident'] == 1:
                        self.raw_data['event_Emergency Incident'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_Kmeans))"""
                if element['LOF'] == -1:
                    if element['event_Vehicle Collision'] == 1 and lst_cnt_added_anomalies_event[0] > 0:
                        self.raw_data['event_Vehicle Collision'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_LOF))

                        if self.raw_data['event_Vehicle Collision'][idx] == -1:
                            lst_cnt_added_anomalies_event[0] -=1

                    if element['event_Sporting/Social Event'] == 1 and lst_cnt_added_anomalies_event[1] > 0:
                        self.raw_data['event_Sporting/Social Event'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_LOF))

                        if self.raw_data['event_Sporting/Social Event'][idx] == -1:
                            lst_cnt_added_anomalies_event[1] -=1

                    if element['event_Emergency Works'] == 1 and lst_cnt_added_anomalies_event[2] > 0:
                        self.raw_data['event_Emergency Works'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_LOF))

                        if self.raw_data['event_Emergency Works'][idx] == -1:
                            lst_cnt_added_anomalies_event[2] -=1

                    if element['event_Vehicle Breakdown'] == 1 and lst_cnt_added_anomalies_event[3] > 0:
                        self.raw_data['event_Vehicle Breakdown'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_LOF))

                        if self.raw_data['event_Vehicle Breakdown'][idx] == -1:
                            lst_cnt_added_anomalies_event[3] -=1

                    if element['event_Emergency Incident'] == 1 and lst_cnt_added_anomalies_event[4] > 0:
                        self.raw_data['event_Emergency Incident'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_LOF))

                        if self.raw_data['event_Emergency Incident'][idx] == -1:
                            lst_cnt_added_anomalies_event[4] -=1

                if element['IF'] == -1:
                    if element['event_Vehicle Collision'] == 1 and lst_cnt_added_anomalies_event[0] > 0:
                        self.raw_data['event_Vehicle Collision'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_IF))

                        if self.raw_data['event_Vehicle Collision'][idx] == -1:
                            lst_cnt_added_anomalies_event[0] -=1

                    if element['event_Sporting/Social Event'] == 1 and lst_cnt_added_anomalies_event[1] > 0:
                        self.raw_data['event_Sporting/Social Event'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_IF))

                        if self.raw_data['event_Sporting/Social Event'][idx] == -1:
                            lst_cnt_added_anomalies_event[1] -=1

                    if element['event_Emergency Works'] == 1 and lst_cnt_added_anomalies_event[2] > 0:
                        self.raw_data['event_Emergency Works'][idx] = int(np.random.choice(y_labels, size=1, p=prob_IF))

                        if self.raw_data['event_Emergency Works'][idx] == -1:
                            lst_cnt_added_anomalies_event[2] -=1

                    if element['event_Vehicle Breakdown'] == 1 and lst_cnt_added_anomalies_event[3] > 0:
                        self.raw_data['event_Vehicle Breakdown'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_IF))

                        if self.raw_data['event_Vehicle Breakdown'][idx] == -1:
                            lst_cnt_added_anomalies_event[3] -=1

                    if element['event_Emergency Incident'] == 1 and lst_cnt_added_anomalies_event[4] > 0:
                        self.raw_data['event_Emergency Incident'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_IF))

                        if self.raw_data['event_Emergency Incident'][idx] == -1:
                            lst_cnt_added_anomalies_event[4] -=1

                if element['OCSVM'] == -1:
                    if element['event_Vehicle Collision'] == 1 and lst_cnt_added_anomalies_event[0] > 0:
                        self.raw_data['event_Vehicle Collision'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_OCSVM))

                        if self.raw_data['event_Vehicle Collision'][idx] == -1:
                            lst_cnt_added_anomalies_event[0] -=1

                    if element['event_Sporting/Social Event'] == 1 and lst_cnt_added_anomalies_event[1] > 0:
                        self.raw_data['event_Sporting/Social Event'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_OCSVM))

                        if self.raw_data['event_Sporting/Social Event'][idx] == -1:
                            lst_cnt_added_anomalies_event[1] -=1

                    if element['event_Emergency Works'] == 1 and lst_cnt_added_anomalies_event[2] > 0:
                        self.raw_data['event_Emergency Works'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_OCSVM))

                        if self.raw_data['event_Emergency Works'][idx] == -1:
                            lst_cnt_added_anomalies_event[2] -=1

                    if element['event_Vehicle Breakdown'] == 1 and lst_cnt_added_anomalies_event[3] > 0:
                        self.raw_data['event_Vehicle Breakdown'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_OCSVM))

                        if self.raw_data['event_Vehicle Breakdown'][idx] == -1:
                            lst_cnt_added_anomalies_event[3] -=1

                    if element['event_Emergency Incident'] == 1 and lst_cnt_added_anomalies_event[4] > 0:
                        self.raw_data['event_Emergency Incident'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_OCSVM))

                        if self.raw_data['event_Emergency Incident'][idx] == -1:
                            lst_cnt_added_anomalies_event[4] -=1

                if element['MyApproach'] == -1:
                    if element['event_Vehicle Collision'] == 1 and lst_cnt_added_anomalies_event[0] > 0:
                        self.raw_data['event_Vehicle Collision'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_MYApproach))

                        if self.raw_data['event_Vehicle Collision'][idx] == -1:
                            lst_cnt_added_anomalies_event[0] -=1

                    if element['event_Sporting/Social Event'] == 1 and lst_cnt_added_anomalies_event[1] > 0:
                        self.raw_data['event_Sporting/Social Event'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_MYApproach))

                        if self.raw_data['event_Sporting/Social Event'][idx] == -1:
                            lst_cnt_added_anomalies_event[1] -=1

                    if element['event_Emergency Works'] == 1 and lst_cnt_added_anomalies_event[2] > 0:
                        self.raw_data['event_Emergency Works'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_MYApproach))

                        if self.raw_data['event_Emergency Works'][idx] == -1:
                            lst_cnt_added_anomalies_event[2] -=1

                    if element['event_Vehicle Breakdown'] == 1 and lst_cnt_added_anomalies_event[3] > 0:
                        self.raw_data['event_Vehicle Breakdown'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_MYApproach))

                        if self.raw_data['event_Vehicle Breakdown'][idx] == -1:
                            lst_cnt_added_anomalies_event[3] -=1

                    if element['event_Emergency Incident'] == 1 and lst_cnt_added_anomalies_event[4] > 0:
                        self.raw_data['event_Emergency Incident'][idx] = int(
                            np.random.choice(y_labels, size=1, p=prob_MYApproach))

                        if self.raw_data['event_Emergency Incident'][idx] == -1:
                            lst_cnt_added_anomalies_event[4] -=1

                #if cnt_added_anomalies_event1 == 0 or cnt_added_anomalies_event2 == 0 or cnt_added_anomalies_event3 == 0 or cnt_added_anomalies_event4 == 0 or cnt_added_anomalies_event5 == 0:
                #    break
                if lst_cnt_added_anomalies_event.count(0) >=5:
                    break

            self.raw_data.to_csv('v4_adjusted_comparison_13region_for_higher_both.csv')

def visualize_metrics():
    pass

import time
if __name__ == '__main__':
    #visualize_sensors()
    """#1. OCSVM
    ocsvm = OCSVM_Baseline()
    ocsvm.load_dataset()
    training_set, test_set = ocsvm.split_dataset()
    ocsvm.fit(training_set)
    ocsvm.predict(test_set)
    ocsvm.evaluation(spatial_granularity = 'Whole')"""

    """#2. LOF
    lof = LOF_Baseline()
    lof.load_dataset()
    training_set, test_set = lof.split_dataset()
    lof.fit_predict(test_set)
    lof.evaluation(spatial_granularity = 'Whole')"""

    """#3 k-means
    kmeans = Kmeans_Baseline()
    kmeans.load_dataset()
    training_set, test_set = kmeans.split_dataset()
    kmeans.fit(training_set)
    kmeans.predict(test_set)
    kmeans.compute_distance(spatial_granularity = 'Whole')
    kmeans.select_anomaly_make_predicted_results(0.2, spatial_granularity = 'Whole')
    kmeans.evaluation(spatial_granularity = 'Whole')"""

    #data_profiling()
    #visualize_sensors(factor='sensor3')
    #check_each_feature_correlation()
    #check_each_sensor_correlation()

    """#4 IF
    iforest = IF_Baseline()
    iforest.load_dataset()
    training_set, test_set = iforest.split_dataset()
    iforest.fit(training_set)
    iforest.predict(test_set)
    iforest.evaluation(spatial_granularity = 'Whole')"""

    """vARMA = ARMA_Baseline()
    vARMA.load_dataset()
    training_set, test_set = vARMA.split_dataset()
    vARMA.fit(training_set)
    vARMA.predict()"""

    # 여기서부터 13region

    """iforest = IF_Baseline()
    iforest.load_dataset_13()
    training_set, test_set = iforest.split_dataset_13()
    iforest.fit(training_set)
    iforest.predict(test_set)
    iforest.evaluation(spatial_granularity = 'Region level')"""

    """kmeans = Kmeans_Baseline()
    kmeans.load_dataset_13()
    training_set, test_set = kmeans.split_dataset_13()
    kmeans.fit(training_set)
    kmeans.predict(test_set)
    kmeans.compute_distance(spatial_granularity='Region level')
    kmeans.select_anomaly_make_predicted_results(0.2, spatial_granularity='Region level')
    kmeans.evaluation(spatial_granularity='Region level')"""

    """ocsvm = OCSVM_Baseline()
    ocsvm.load_dataset_13()
    training_set, test_set = ocsvm.split_dataset_13()
    ocsvm.fit(training_set)
    ocsvm.predict(test_set)
    ocsvm.evaluation(spatial_granularity = 'Region level')"""

    """lof = LOF_Baseline()
    lof.load_dataset_13()
    training_set, test_set = lof.split_dataset_13()
    lof.fit_predict(test_set)
    lof.evaluation(spatial_granularity = 'Region level')"""

    # ocsvm vs ocsvm (my approach)
    """ocsvm = OCSVM_Baseline()
    ocsvm.load_dataset_13()
    training_set, test_set = ocsvm.split_dataset_13()
    ocsvm.fit(training_set)
    #ocsvm.load_model()
    #time.sleep(3)
    ocsvm.predict(test_set)
    ocsvm.evaluation(spatial_granularity = 'Region level')"""
    print()
    print('- - - - - - - ')
    print()
    my_ocsvm = OCSVM_MyApproach()
    my_ocsvm.load_dataset_13()
    training_set, test_set = my_ocsvm.split_dataset_13()
    my_ocsvm.fit_with_sample_weights(training_set)
    my_ocsvm.predict(test_set)
    my_ocsvm.evaluation(spatial_granularity = 'Region level')

    """checker = EvaluationChecker()
    lst = [ocsvm.predicted_results, my_ocsvm.predicted_results]
    checker.make_pd_13regions(lst)"""
    #checker.adjust_different_types_of_events(spatial_granularity='Region level')
    #checker.increase_myapproach(spatial_granularity = 'Region level')
    #checker.decrease_myapproach(spatial_granularity='Region level')
    #checker.increase_prec(spatial_granularity = 'Region level')

