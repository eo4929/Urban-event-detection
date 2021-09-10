import pandas as pd
import random
import numpy as np
import sklearn.svm as SVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, average_precision_score

import matplotlib.pyplot as plt

from pandas_profiling import ProfileReport

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.neighbors import KDTree

import joblib

from igraph import *

import json

import sys
np.set_printoptions(threshold=sys.maxsize)

class DataPreparer:
    def __init__(self):
        self.raw_data = pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/ordered_final_data1_data2_data3_13region.csv')

        self.data_hour_0 = None; self.data_hour_1 = None; self.data_hour_2 = None
        self.data_hour_3 = None; self.data_hour_4 = None; self.data_hour_5 = None
        self.data_hour_6 = None; self.data_hour_7 = None; self.data_hour_8 = None
        self.data_hour_9 = None; self.data_hour_10 = None; self.data_hour_11 = None
        self.data_hour_12 = None; self.data_hour_13 = None; self.data_hour_14 = None
        self.data_hour_15 = None; self.data_hour_16 = None; self.data_hour_17 = None
        self.data_hour_18 = None; self.data_hour_19 = None; self.data_hour_20 = None
        self.data_hour_21 = None; self.data_hour_22 = None; self.data_hour_23 = None

        # 위에 있는 데이터들 지역단위로 groupby한 데이터들
        self.grouped_data_hour_0 = None;
        self.grouped_data_hour_1 = None;
        self.grouped_data_hour_2 = None
        self.grouped_data_hour_3 = None;
        self.grouped_data_hour_4 = None;
        self.grouped_data_hour_5 = None
        self.grouped_data_hour_6 = None;
        self.grouped_data_hour_7 = None;
        self.grouped_data_hour_8 = None
        self.grouped_data_hour_9 = None;
        self.grouped_data_hour_10 = None;
        self.grouped_data_hour_11 = None
        self.grouped_data_hour_12 = None;
        self.grouped_data_hour_13 = None;
        self.grouped_data_hour_14 = None
        self.grouped_data_hour_15 = None;
        self.grouped_data_hour_16 = None;
        self.grouped_data_hour_17 = None
        self.grouped_data_hour_18 = None;
        self.grouped_data_hour_19 = None;
        self.grouped_data_hour_20 = None
        self.grouped_data_hour_21 = None;
        self.grouped_data_hour_22 = None;
        self.grouped_data_hour_23 = None

        self.region_data_per_hours = dict()

        self.lst_graph = None

    def separate_raw_data(self):
        print('separate_raw_data invoked')
        self.data_hour_0 = self.raw_data.loc[(self.raw_data['Time_hour'] == 0)]
        self.data_hour_1 = self.raw_data.loc[(self.raw_data['Time_hour'] == 1)]
        self.data_hour_2 = self.raw_data.loc[(self.raw_data['Time_hour'] == 2)]
        self.data_hour_3 = self.raw_data.loc[(self.raw_data['Time_hour'] == 3)]
        self.data_hour_4 = self.raw_data.loc[(self.raw_data['Time_hour'] == 4)]
        self.data_hour_5 = self.raw_data.loc[(self.raw_data['Time_hour'] == 5)]
        self.data_hour_6 = self.raw_data.loc[(self.raw_data['Time_hour'] == 6)]
        self.data_hour_7 = self.raw_data.loc[(self.raw_data['Time_hour'] == 7)]
        self.data_hour_8 = self.raw_data.loc[(self.raw_data['Time_hour'] == 8)]
        self.data_hour_9 = self.raw_data.loc[(self.raw_data['Time_hour'] == 9)]
        self.data_hour_10 = self.raw_data.loc[(self.raw_data['Time_hour'] == 10)]
        self.data_hour_11 = self.raw_data.loc[(self.raw_data['Time_hour'] == 11)]
        self.data_hour_12 = self.raw_data.loc[(self.raw_data['Time_hour'] == 12)]
        self.data_hour_13 = self.raw_data.loc[(self.raw_data['Time_hour'] == 13)]
        self.data_hour_14 = self.raw_data.loc[(self.raw_data['Time_hour'] == 14)]
        self.data_hour_15 = self.raw_data.loc[(self.raw_data['Time_hour'] == 15)]
        self.data_hour_16 = self.raw_data.loc[(self.raw_data['Time_hour'] == 16)]
        self.data_hour_17 = self.raw_data.loc[(self.raw_data['Time_hour'] == 17)]
        self.data_hour_18 = self.raw_data.loc[(self.raw_data['Time_hour'] == 18)]
        self.data_hour_19 = self.raw_data.loc[(self.raw_data['Time_hour'] == 19)]
        self.data_hour_20 = self.raw_data.loc[(self.raw_data['Time_hour'] == 20)]
        self.data_hour_21 = self.raw_data.loc[(self.raw_data['Time_hour'] == 21)]
        self.data_hour_22 = self.raw_data.loc[(self.raw_data['Time_hour'] == 22)]
        self.data_hour_23 = self.raw_data.loc[(self.raw_data['Time_hour'] == 23)]

    def create_region_table(self):
        print('create_region_table invoked')
        self.data_hour_0.drop(['Total_date','Day'], axis=1,inplace = True)
        self.data_hour_1.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_2.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_3.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_4.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_5.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_6.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_7.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_8.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_9.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_10.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_11.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_12.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_13.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_14.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_15.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_16.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_17.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_18.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_19.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_20.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_21.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_22.drop(['Total_date','Day'], axis=1, inplace = True)
        self.data_hour_23.drop(['Total_date','Day'], axis=1, inplace = True)

        print('a example for dropped data_hour_0')
        print(self.data_hour_0)

        self.grouped_data_hour_0 = self.data_hour_0.groupby(['Region']).mean()
        self.grouped_data_hour_1 = self.data_hour_1.groupby(['Region']).mean()
        self.grouped_data_hour_2 = self.data_hour_2.groupby(['Region']).mean()
        self.grouped_data_hour_3 = self.data_hour_3.groupby(['Region']).mean()
        self.grouped_data_hour_4 = self.data_hour_4.groupby(['Region']).mean()
        self.grouped_data_hour_5 = self.data_hour_5.groupby(['Region']).mean()
        self.grouped_data_hour_6 = self.data_hour_6.groupby(['Region']).mean()
        self.grouped_data_hour_7 = self.data_hour_7.groupby(['Region']).mean()
        self.grouped_data_hour_8 = self.data_hour_8.groupby(['Region']).mean()
        self.grouped_data_hour_9 = self.data_hour_9.groupby(['Region']).mean();
        self.grouped_data_hour_10 = self.data_hour_10.groupby(['Region']).mean();
        self.grouped_data_hour_11 = self.data_hour_11.groupby(['Region']).mean()
        self.grouped_data_hour_12 = self.data_hour_12.groupby(['Region']).mean();
        self.grouped_data_hour_13 = self.data_hour_13.groupby(['Region']).mean();
        self.grouped_data_hour_14 = self.data_hour_14.groupby(['Region']).mean()
        self.grouped_data_hour_15 = self.data_hour_15.groupby(['Region']).mean();
        self.grouped_data_hour_16 = self.data_hour_16.groupby(['Region']).mean();
        self.grouped_data_hour_17 = self.data_hour_17.groupby(['Region']).mean()
        self.grouped_data_hour_18 = self.data_hour_18.groupby(['Region']).mean();
        self.grouped_data_hour_19 = self.data_hour_19.groupby(['Region']).mean();
        self.grouped_data_hour_20 = self.data_hour_20.groupby(['Region']).mean()
        self.grouped_data_hour_21 = self.data_hour_21.groupby(['Region']).mean();
        self.grouped_data_hour_22 = self.data_hour_22.groupby(['Region']).mean();
        self.grouped_data_hour_23 = self.data_hour_23.groupby(['Region']).mean()

        print('a example for grouped_data_hour_0')
        print(self.grouped_data_hour_0)

    def export_grouped_data_hour(self):
        self.grouped_data_hour_0.to_csv('region_level_data_hour_0_groupbyRegion.csv')
        self.grouped_data_hour_1.to_csv('region_level_data_hour_1_groupbyRegion.csv')
        self.grouped_data_hour_2.to_csv('region_level_data_hour_2_groupbyRegion.csv')
        self.grouped_data_hour_3.to_csv('region_level_data_hour_3_groupbyRegion.csv')
        self.grouped_data_hour_4.to_csv('region_level_data_hour_4_groupbyRegion.csv')
        self.grouped_data_hour_5.to_csv('region_level_data_hour_5_groupbyRegion.csv')
        self.grouped_data_hour_6.to_csv('region_level_data_hour_6_groupbyRegion.csv')
        self.grouped_data_hour_7.to_csv('region_level_data_hour_7_groupbyRegion.csv')
        self.grouped_data_hour_8.to_csv('region_level_data_hour_8_groupbyRegion.csv')
        self.grouped_data_hour_9.to_csv('region_level_data_hour_9_groupbyRegion.csv')
        self.grouped_data_hour_10.to_csv('region_level_data_hour_10_groupbyRegion.csv')
        self.grouped_data_hour_11.to_csv('region_level_data_hour_11_groupbyRegion.csv')
        self.grouped_data_hour_12.to_csv('region_level_data_hour_12_groupbyRegion.csv')
        self.grouped_data_hour_13.to_csv('region_level_data_hour_13_groupbyRegion.csv')
        self.grouped_data_hour_14.to_csv('region_level_data_hour_14_groupbyRegion.csv')
        self.grouped_data_hour_15.to_csv('region_level_data_hour_15_groupbyRegion.csv')
        self.grouped_data_hour_16.to_csv('region_level_data_hour_16_groupbyRegion.csv')
        self.grouped_data_hour_17.to_csv('region_level_data_hour_17_groupbyRegion.csv')
        self.grouped_data_hour_18.to_csv('region_level_data_hour_18_groupbyRegion.csv')
        self.grouped_data_hour_19.to_csv('region_level_data_hour_19_groupbyRegion.csv')
        self.grouped_data_hour_20.to_csv('region_level_data_hour_20_groupbyRegion.csv')
        self.grouped_data_hour_21.to_csv('region_level_data_hour_21_groupbyRegion.csv')
        self.grouped_data_hour_22.to_csv('region_level_data_hour_22_groupbyRegion.csv')
        self.grouped_data_hour_23.to_csv('region_level_data_hour_23_groupbyRegion.csv')

    def import_grouped_data_hour(self):
        print('import_grouped_data_hour invoked')
        self.grouped_data_hour_0 = pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_0_groupbyRegion.csv')
        self.grouped_data_hour_1= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_1_groupbyRegion.csv')
        self.grouped_data_hour_2= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_2_groupbyRegion.csv')
        self.grouped_data_hour_3= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_3_groupbyRegion.csv')
        self.grouped_data_hour_4= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_4_groupbyRegion.csv')
        self.grouped_data_hour_5= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_5_groupbyRegion.csv')
        self.grouped_data_hour_6= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_6_groupbyRegion.csv')
        self.grouped_data_hour_7= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_7_groupbyRegion.csv')
        self.grouped_data_hour_8= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_8_groupbyRegion.csv')
        self.grouped_data_hour_9= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_9_groupbyRegion.csv')
        self.grouped_data_hour_10= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_10_groupbyRegion.csv')
        self.grouped_data_hour_11= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_11_groupbyRegion.csv')
        self.grouped_data_hour_12= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_12_groupbyRegion.csv')
        self.grouped_data_hour_13= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_13_groupbyRegion.csv')
        self.grouped_data_hour_14= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_14_groupbyRegion.csv')
        self.grouped_data_hour_15= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_15_groupbyRegion.csv')
        self.grouped_data_hour_16= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_16_groupbyRegion.csv')
        self.grouped_data_hour_17= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_17_groupbyRegion.csv')
        self.grouped_data_hour_18= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_18_groupbyRegion.csv')
        self.grouped_data_hour_19= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_19_groupbyRegion.csv')
        self.grouped_data_hour_20= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_20_groupbyRegion.csv')
        self.grouped_data_hour_21= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_21_groupbyRegion.csv')
        self.grouped_data_hour_22= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_22_groupbyRegion.csv')
        self.grouped_data_hour_23= pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/region_data_per_hour/region_level_data_hour_23_groupbyRegion.csv')

        self.grouped_data_hour_0.set_index(['Region'], inplace=True)
        self.grouped_data_hour_0.drop( [ 'Season'] , axis=1, inplace=True)
        self.grouped_data_hour_1.set_index(['Region'], inplace=True)
        self.grouped_data_hour_1.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_2.set_index(['Region'], inplace=True)
        self.grouped_data_hour_2.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_3.set_index(['Region'], inplace=True)
        self.grouped_data_hour_3.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_4.set_index(['Region'], inplace=True)
        self.grouped_data_hour_4.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_5.set_index(['Region'], inplace=True)
        self.grouped_data_hour_5.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_6.set_index(['Region'], inplace=True)
        self.grouped_data_hour_6.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_7.set_index(['Region'], inplace=True)
        self.grouped_data_hour_7.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_8.set_index(['Region'], inplace=True)
        self.grouped_data_hour_8.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_9.set_index(['Region'], inplace=True)
        self.grouped_data_hour_9.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_10.set_index(['Region'], inplace=True)
        self.grouped_data_hour_10.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_11.set_index(['Region'], inplace=True)
        self.grouped_data_hour_11.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_12.set_index(['Region'], inplace=True)
        self.grouped_data_hour_12.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_13.set_index(['Region'], inplace=True)
        self.grouped_data_hour_13.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_14.set_index(['Region'], inplace=True)
        self.grouped_data_hour_14.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_15.set_index(['Region'], inplace=True)
        self.grouped_data_hour_15.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_16.set_index(['Region'], inplace=True)
        self.grouped_data_hour_16.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_17.set_index(['Region'], inplace=True)
        self.grouped_data_hour_17.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_18.set_index(['Region'], inplace=True)
        self.grouped_data_hour_18.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_19.set_index(['Region'], inplace=True)
        self.grouped_data_hour_19.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_20.set_index(['Region'], inplace=True)
        self.grouped_data_hour_20.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_21.set_index(['Region'], inplace=True)
        self.grouped_data_hour_21.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_22.set_index(['Region'], inplace=True)
        self.grouped_data_hour_22.drop(['Season'], axis=1, inplace=True)
        self.grouped_data_hour_23.set_index(['Region'], inplace=True)
        self.grouped_data_hour_23.drop(['Season'], axis=1, inplace=True)

        #print( self.grouped_data_hour_23 )
    def make_sensor_values_per_region(self):
        print('make_sensor_values_per_region invoked')

        sensor_values_hour_0 = []
        for idx, element in self.grouped_data_hour_0.iterrows():
            element_to_np = element.to_list()
            #print('element_to_np: ', element_to_np)
            sensor_values_hour_0.append( element_to_np )
        self.region_data_per_hours['{}'.format(0)] = sensor_values_hour_0

        sensor_values_hour_1 = []
        for idx, element in self.grouped_data_hour_1.iterrows():
            element_to_np = element.to_list()
            #print('element_to_np: ', element_to_np)
            sensor_values_hour_1.append(element_to_np)
        self.region_data_per_hours['{}'.format(1)] = sensor_values_hour_1

        sensor_values_hour_2 = []
        for idx, element in self.grouped_data_hour_2.iterrows():
            element_to_np = element.to_list()
            #print('element_to_np: ', element_to_np)
            sensor_values_hour_2.append(element_to_np)
        self.region_data_per_hours['{}'.format(2)] = sensor_values_hour_2

        sensor_values_hour_3 = []
        for idx, element in self.grouped_data_hour_3.iterrows():
            element_to_np = element.to_list()
            #print('element_to_np: ', element_to_np)
            sensor_values_hour_3.append(element_to_np)
        self.region_data_per_hours['{}'.format(3)] = sensor_values_hour_3

        sensor_values_hour_4 = []
        for idx, element in self.grouped_data_hour_4.iterrows():
            element_to_np = element.to_list()
            #print('element_to_np: ', element_to_np)
            sensor_values_hour_4.append(element_to_np)
        self.region_data_per_hours['{}'.format(4)] = sensor_values_hour_4

        sensor_values_hour_5 = []
        for idx, element in self.grouped_data_hour_5.iterrows():
            element_to_np = element.to_list()
            #print('element_to_np: ', element_to_np)
            sensor_values_hour_5.append(element_to_np)
        self.region_data_per_hours['{}'.format(5)] = sensor_values_hour_5

        sensor_values_hour_6 = []
        for idx, element in self.grouped_data_hour_6.iterrows():
            element_to_np = element.to_list()
            #print('element_to_np: ', element_to_np)
            sensor_values_hour_6.append(element_to_np)
        self.region_data_per_hours['{}'.format(6)] = sensor_values_hour_6

        sensor_values_hour_7 = []
        for idx, element in self.grouped_data_hour_7.iterrows():
            element_to_np = element.to_list()
            #print('element_to_np: ', element_to_np)
            sensor_values_hour_7.append(element_to_np)
        self.region_data_per_hours['{}'.format(7)] = sensor_values_hour_7

        sensor_values_hour_8 = []
        for idx, element in self.grouped_data_hour_8.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_8.append(element_to_np)
        self.region_data_per_hours['{}'.format(8)] = sensor_values_hour_8

        sensor_values_hour_9 = []
        for idx, element in self.grouped_data_hour_9.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_9.append(element_to_np)
        self.region_data_per_hours['{}'.format(9)] = sensor_values_hour_9

        sensor_values_hour_10 = []
        for idx, element in self.grouped_data_hour_10.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_10.append(element_to_np)
        self.region_data_per_hours['{}'.format(10)] = sensor_values_hour_10

        sensor_values_hour_11 = []
        for idx, element in self.grouped_data_hour_11.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_11.append(element_to_np)
        self.region_data_per_hours['{}'.format(11)] = sensor_values_hour_11

        sensor_values_hour_12 = []
        for idx, element in self.grouped_data_hour_12.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_12.append(element_to_np)
        self.region_data_per_hours['{}'.format(12)] = sensor_values_hour_12

        sensor_values_hour_13 = []
        for idx, element in self.grouped_data_hour_13.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_13.append(element_to_np)
        self.region_data_per_hours['{}'.format(13)] = sensor_values_hour_13

        sensor_values_hour_14 = []
        for idx, element in self.grouped_data_hour_14.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_14.append(element_to_np)
        self.region_data_per_hours['{}'.format(14)] = sensor_values_hour_14

        sensor_values_hour_15 = []
        for idx, element in self.grouped_data_hour_15.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_15.append(element_to_np)
        self.region_data_per_hours['{}'.format(15)] = sensor_values_hour_15

        sensor_values_hour_16 = []
        for idx, element in self.grouped_data_hour_16.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_16.append(element_to_np)
        self.region_data_per_hours['{}'.format(16)] = sensor_values_hour_16

        sensor_values_hour_17 = []
        for idx, element in self.grouped_data_hour_17.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_17.append(element_to_np)
        self.region_data_per_hours['{}'.format(17)] = sensor_values_hour_17

        sensor_values_hour_18 = []
        for idx, element in self.grouped_data_hour_18.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_18.append(element_to_np)
        self.region_data_per_hours['{}'.format(18)] = sensor_values_hour_18

        sensor_values_hour_19 = []
        for idx, element in self.grouped_data_hour_19.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_19.append(element_to_np)
        self.region_data_per_hours['{}'.format(19)] = sensor_values_hour_19

        sensor_values_hour_20 = []
        for idx, element in self.grouped_data_hour_20.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_20.append(element_to_np)
        self.region_data_per_hours['{}'.format(20)] = sensor_values_hour_20

        sensor_values_hour_21 = []
        for idx, element in self.grouped_data_hour_21.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_21.append(element_to_np)
        self.region_data_per_hours['{}'.format(21)] = sensor_values_hour_21

        sensor_values_hour_22 = []
        for idx, element in self.grouped_data_hour_22.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_22.append(element_to_np)
        self.region_data_per_hours['{}'.format(22)] = sensor_values_hour_22

        sensor_values_hour_23 = []
        for idx, element in self.grouped_data_hour_23.iterrows():
            element_to_np = element.to_list()
            # print('element_to_np: ', element_to_np)
            sensor_values_hour_23.append(element_to_np)
        self.region_data_per_hours['{}'.format(23)] = sensor_values_hour_23

        print(self.region_data_per_hours)
        print()

        with open('C:/Users/Dae-Young Park/Weak_labeling_framework/region_data_per_hours.json', 'w') as output:
            json.dump(self.region_data_per_hours, output)

    def load_region_data_per_hours(self):
        with open('C:/Users/Dae-Young Park/Weak_labeling_framework/region_data_per_hours.json') as data:
            self.region_data_per_hours = json.load(data)
        # here 이거 구현했으니 기거 기반으로 vertex['sensor values'] 채워넣자

    def construct_region_graph(self):
        def compute_k():
            pass

        lst_graph = [ GraphGenerator() for _ in range(24) ]

        # 1. vertex 구성하기
        for i in range(24):
            lst_graph[i].hour_information = i
            lst_graph[i].make_vertex_regionlevel( self.region_data_per_hours['{}'.format(i)] )
            lst_graph[i].make_edge_regionlevel()
            lst_graph[i].compute_centeredness_center_closeness()
            lst_graph[i].make_Spatial_Outlierness_scores()

        self.lst_graph = lst_graph

    def make_data_with_spatial_outlierness_score(self):
        weighter = Weighter()
        weighter.impute_Spatial_Outlierness(self.lst_graph)


class Weighter():
    def __init__(self):
        self.empty_data = pd.read_csv('C:/Users/Dae-Young Park/Weak_labeling_framework/final_sensor_data/Spatial_Outliernesss_final_data1_data2_data3_13region.csv')
        #self.empty_data.set_index(['Total_date'], inplace=True)

    def impute_Spatial_Outlierness(self, lst_graph):

        for idx, element in self.empty_data.iterrows():
            hour_id = int(element['Time_hour'])
            region_id =  int(element['Region'])
            lst_Spatial_Outlierness = lst_graph[hour_id].get_Spatial_Outlierness_scores()
            self.empty_data['Total_Spatial_Outlierness'][idx] = lst_Spatial_Outlierness[region_id]
            print('outlierness score for idx {}: '.format(idx), self.empty_data['Total_Spatial_Outlierness'][idx])

        self.empty_data.to_csv('k12_filled_Spatial_Outliernesss_final_data1_data2_data3_13region.csv')

class GraphGenerator():
    def __init__(self):
        self.graph = Graph(directed = True)
        self.hour_information = None

        self.kdtree = None
        self.region_data = None

        self.thres_final_outlierness = None # 1~13 range 이어야 할 것임
        self.lst_Spatial_Outlierness = None

        print('initial graph: ', self.graph)
        print('hour information: ', self.hour_information)
    def make_vertex_regionlevel(self, region_data):
        print('make_vertex_regionlevel invoked')

        # 거리 계싼 디폴트는 minkowski 라고 함 -> 다른 후보: ["euclidean", "minkowski", "manhattan"]
        self.kdtree = KDTree(region_data, metric="minkowski")
        self.region_data = region_data

        for i in range(13):
            self.graph.add_vertex()
            vertex = self.graph.vs[ self.graph.vcount() - 1 ]
            vertex['id'] = vertex.index
            vertex['sensor values'] = region_data[i]

        print('graph: ', self.graph)
        print()

    def search_k(self):
        pass # edge 구성을 위한 적절한 k 구하기 -> 이거 해야 함

    def make_edge_regionlevel(self, K=12): # 지금은 k가 3으로 되어있구만
        print('make_edge invoked')

        # 1. 노드별 knn 인 vertex에 대한 인덱스 구하기
        for i in range(13):

            dist, idx_knn = self.kdtree.query( [self.region_data[i]], k=K )
            #print(idx_knn)
            #print(dist)

            knn = idx_knn[0]
            dist = dist[0]

            #2. graph에 엣지 넣어주기
            for j in range(1, len(knn) ):
                self.graph.add_edges( [(knn[0] , knn[j])] )
                edge = self.graph.es[ self.graph.ecount() - 1 ]
                edge['id'] = edge.index
                edge['weight'] = 1 / dist[j] # weight 이 similarity 니까

        print('self.graph: ')
        print(self.graph)
        #print(self.graph.get_edgelist()  )
        #print( self.graph.es['weight'] )
        print()

    def compute_centeredness_center_closeness(self):
        centeredness_score = self.graph.authority_score(
            weights = self.graph.es['weight'],
            scale = True
        )

        center_closeness_score = self.graph.hub_score(
            weights=self.graph.es['weight'],
            scale=True
        )

        for i in range( len(self.graph.vs) ):
            vertex = self.graph.vs[i]
            vertex['centeredness_score'] = centeredness_score[i]
            vertex['center_closeness_score'] = center_closeness_score[i]

        """print('centeredness_score: ')
        print(self.graph.vs['centeredness_score'])
        print('center_closeness_score: ')
        print(self.graph.vs['center_closeness_score'])"""

    def make_Spatial_Outlierness_scores(self):

        sorted_centeredness_score = sorted(self.graph.vs['centeredness_score'] )
        sorted_center_closeness_score = sorted( self.graph.vs['center_closeness_score'] )

        self.thres_final_outlierness = 3 # 일단, 3개 지역만 아웃라이어로 고려해보자

        lst_idx_lower_centeredness_score = []
        lst_index_check = [0] * 13 # 이미 체크된 놈은 그냥 넘어가도록 하자
        for each_ele in sorted_centeredness_score[:self.thres_final_outlierness]:
            for i in range(len( self.graph.vs['centeredness_score'] )):
                if each_ele == self.graph.vs['centeredness_score'][i] and lst_index_check[i] == 0:
                    lst_idx_lower_centeredness_score.append( i )
                    lst_index_check[i] = 1
                    break
        #print('lst_idx_lower_centeredness_score: ',lst_idx_lower_centeredness_score)

        lst_idx_lower_center_closeness_score = []
        lst_index_check = [0] * 13  # 이미 체크된 놈은 그냥 넘어가도록 하자
        for each_ele in sorted_center_closeness_score[:self.thres_final_outlierness]:
            for i in range(len( self.graph.vs['center_closeness_score'] )):
                if each_ele == self.graph.vs['center_closeness_score'][i] and lst_index_check[i] == 0:
                    lst_idx_lower_center_closeness_score.append( i )
                    lst_index_check[i] = 1
                    break
        #print('lst_idx_lower_center_closeness_score: ', lst_idx_lower_center_closeness_score)

        lst_idx_intersection = [ x for x in lst_idx_lower_centeredness_score if x in lst_idx_lower_center_closeness_score ]
        #print('lst_idx_intersection: ', lst_idx_intersection)

        lst_Spatial_Outlierness = [0] * 13

        for i in range(len(lst_idx_intersection)):
            lst_Spatial_Outlierness[ lst_idx_intersection[i] ] = 1 # 1-> check 의미 실수 이게 weight vector니까 작게 줘야겟네 7넣으니 똑같다.. 100 넣어보자 -> 100도 얼마 차이 안남.. -> 500 넣자

        print('lst_Spatial_Outlierness: ', lst_Spatial_Outlierness)

        self.lst_Spatial_Outlierness = lst_Spatial_Outlierness

    def get_Spatial_Outlierness_scores(self):
        return self.lst_Spatial_Outlierness


if __name__ == '__main__':
    separator = DataPreparer()
    #separator.separate_raw_data()
    #separator.create_region_table()
    #separator.export_grouped_data_hour()
    #separator.import_grouped_data_hour()
    #separator.make_sensor_values_per_region()
    separator.load_region_data_per_hours()
    separator.construct_region_graph()
    separator.make_data_with_spatial_outlierness_score()