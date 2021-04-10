"""
This module handles the clustering of the data
"""
import math
import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import config as cnf
import utils as utils
import viz as viz

# set up formatting for plotting
sns.set(font_scale=1.1, rc={'axes.facecolor': 'lightgrey'})

def cluster(features: np.ndarray) -> None:
	max_cluster = cnf.MAX_CLUSTER+1 # plus 1 to get desired cluster number
	inertia = []
	silhouette_scores = []

	if os.path.exists(cnf.MODELS_PATH):
		print("#### Reading stored kMeans models")
		all_files = os.listdir(cnf.MODELS_PATH)
		all_files = filter(lambda x: x[-4:] == '.pkl', all_files)
		kmeans_per_k = [pickle.load(open(cnf.MODELS_PATH+'/'+file_name, 'rb')) for file_name in all_files]
	else:
		print("#### Generate kMeans models ####")
		kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(features)
		            for k in tqdm(range(2, max_cluster))]

		if not os.path.exists('models'):
		    os.mkdir('models')

		# save models locally
		[pickle.dump(model[1], open(cnf.MODELS_PATH+'/kmeans_{}.pkl'.format(model[0]+2), 'wb')) for model in enumerate(kmeans_per_k)]
		print("####kMeans models generated and saved locally ####")

	# computing scores
	print("####kMeans - computing inertia ####")
	inertias = [model.inertia_ for model in tqdm(kmeans_per_k)]
	print("####kMeans - computing silhouette scores ####")
	silhouette_scores = [silhouette_score(features, model.labels_)
	                     for model in tqdm(kmeans_per_k)]
	# get index position of top N values
	top_sil_score_indexes = utils.get_top_n_indexes(silhouette_scores, 4)

	# create image - kMeans on word vector
	fig, ax = plt.subplots(figsize=(8, 5))
	ax = sns.lineplot(x=range(1, len(inertias) + 1), y=inertias)
	ax.set_title("kMeans - k for M4 time series selected features")
	ax.set_xlabel("# of k")
	ax.set_ylabel("Inertia")
	viz.save_fig('kmeans_daily_series')

	# silhouette_scores viz
	fig, ax = plt.subplots(figsize=(8, 5))
	ax = sns.lineplot(x=range(1, len(silhouette_scores) + 1),
	                  y=silhouette_scores)
	ax.set_title('Silhouette Scores')
	ax.set_xlabel('k')
	ax.set_ylabel('Silhouette Score')
	viz.save_fig('kmeans_sil_score_daily_series')

    # Silhouette Diagrams
	viz.create_sil_diagram(kmeans_per_k, features, top_sil_score_indexes,
						"kmeans_sil_dia_daily_series", silhouette_scores)
