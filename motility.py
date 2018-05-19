#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:35:53 2018

@author: paluchlab
"""
from __future__ import division
import skimage.io as io
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import  mannwhitneyu, mstats
import matplotlib
import os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# Choose features from heteromotility table to be used for PCA and t-SNE
# Add them to the col-List below
col = ['cell_id', 'total_distance', 'net_distance',  'progressivity', 
      'max_speed', 'min_speed', 'avg_speed', 'MSD_slope', 
       'disp_var', 'disp_skew', 'rw_netdist', 
       'avg_moving_speed01', 'time_moving01', 
 'hurst_RS', 'nongauss', 'autocorr_1', 'rw_kurtosis01',]
  
# autocorr_2',]
# 'autocorr_3','autocorr_4','autocorr_5','autocorr_6','autocorr_7','autocorr_8',]
# 'rw_kurtosis02',
# 'rw_kurtosis03','rw_kurtosis04','rw_kurtosis05','rw_kurtosis10',

# Choose an average speed threshold speed_thresh in px/frame, cells moving 
# slower than this threshold wont be considered for analysis
speed_thresh = 1.0
# create a directory with your heteromotility stats .csv 
directory = './190218/heteromotility/'
# All files in this directory ending with .csv will be considered for analysis
files = os.listdir(directory)
files = [f for f in files if '.csv' in f]
# Reads in the csv in pandas, adds a label column = filename, drops all cells
# slower than speed_thresh
samples, labels =[],[]
for filename in files:
    df = pd.read_csv(directory + filename , usecols = col)
    df['label'] = filename.replace('.csv','')
    labels.append(filename.replace('.csv',''))
    samples.append(df[df.avg_speed>speed_thresh])



#plt.hist(speed_6[speed_6 > 1], bins = 10, alpha = 0.5)
#plt.hist(speed_10[speed_10 > 1], bins = 10, alpha = 0.5)
#print mstats.normaltest(speed_10[speed_10 > 1])
#mean_6 = np.mean(speed_6[speed_6 > 1])
#mean_10 = np.mean(speed_10[speed_10 > 1])
#print mannwhitneyu(speed_6[speed_6 > 1], speed_10[speed_10 > 1])
# All samples will be put together in df
df = pd.concat(samples, ignore_index = True)
# For analysis we drop cell_id and split off the label column
label_df = df.label
df = df.drop(['label'], axis = 1).drop(['cell_id'], axis = 1)
# Data is normalized/scaled to ensure equal contribution from all features
normalized_df=(df-df.min())/(df.max()-df.min())
# Create a PCA object from sklearn.decomposition
pca = PCA()
# Perform PCA on the normalized data and return transformed principal components
transformed = pca.fit_transform(normalized_df.values)
components = pca.components_
normed_comp = abs(components)/np.sum(abs(components),axis = 0)

# Calculates variance contribution of each principal component
expl_var_ratio = pca.explained_variance_ratio_
# Creates a scatter plot of the first two principal components
w, h = plt.figaspect(1.)
pca_fig, pca_ax =plt.subplots(figsize=(w,h))
for i in labels:
    pca_ax.scatter(transformed[:,0][label_df == i], transformed[:,1][label_df == i], 
                label = str(i) )
pca_ax.legend()
pca_ax.set_xlabel('PC1 (variance ' + str(int(expl_var_ratio[0]*100))+ ' %)')
pca_ax.set_ylabel('PC2 (variance ' + str(int(expl_var_ratio[1]*100))+ ' %)')
# Saves pca in directory
pca_fig.savefig(directory + 'pca.pdf',  bbox_inches='tight')

from sklearn.manifold import TSNE
# Creates t-SNE plot without axis
tsne = TSNE(n_components = 2, init = 'pca', random_state= 0 )
tsne_points = tsne.fit_transform(normalized_df.values)
fig, ax = plt.subplots(figsize=(w,h))
ax.axis('off')
for i in labels:
    ax.scatter(tsne_points[:,0][label_df == i], tsne_points[:,1][label_df == i], 
           label = str(i))
ax.legend(loc=2)
# Saves t-SNE plot in directory
fig.savefig(directory + 'tsne.pdf',  bbox_inches='tight')
