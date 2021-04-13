"""
This module handles the clustering of the data
"""
import math
import os
import pickle
from typing import List

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
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
	inertias = []
	silhouette_scores = []

	if os.path.exists(cnf.MODELS_PATH) and\
		os.path.exists(cnf.SCORES_FOLDER_PATH):
		print("#### Reading stored kMeans models ###")
		all_files = os.listdir(cnf.MODELS_PATH)
		all_files = filter(lambda x: x[-4:] == '.pkl', all_files)
		kmeans_per_k = [pickle.load(open(cnf.MODELS_PATH+'/'+file_name, 'rb')) for file_name in all_files]

		# reading scores
		print("#### Reading stored kMeans inertia scores ####")
		with open(cnf.SCORES_FOLDER_PATH+'/'+cnf.INERTIA_FILE_PATH, 'r') as filehandle:
			for line in filehandle:
				inertia = line[:-1] # remove linebreak
				inertias.append(inertia)

		print("#### Reading stored kMeans silhouette_scores ####")
		with open(cnf.SCORES_FOLDER_PATH+'/'+cnf.SIL_SCORES_FILE_PATH, 'r') as filehandle:
			for line in filehandle:
				sil_score = line[:-1] #remove linebreak
				silhouette_scores.append(sil_score)

	else:
		print("#### Generate kMeans models ####")
		kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(features)
		            for k in tqdm(range(2, max_cluster))]

		if not os.path.exists(cnf.MODELS_PATH):
		    os.mkdir(cnf.MODELS_PATH)

		# save models locally
		[pickle.dump(model[1], open(cnf.MODELS_PATH+'/kmeans_{}.pkl'.format(model[0]+2), 'wb')) for model in enumerate(kmeans_per_k)]
		print("####kMeans models generated and saved locally ####")

		if not os.path.exists(cnf.SCORES_FOLDER_PATH):
			os.mkdir(cnf.SCORES_FOLDER_PATH)

		# computing scores
		print("####kMeans - gathering inertia per k ####")
		inertias = [model.inertia_ for model in tqdm(kmeans_per_k)]
		print("####kMeans - computing silhouette scores ####")
		silhouette_scores = [silhouette_score(features, model.labels_)
		                     for model in tqdm(kmeans_per_k)]

		with open(cnf.SCORES_FOLDER_PATH+'/'+cnf.INERTIA_FILE_PATH, 'w') as filehandle:
			for listitem in inertias:
				filehandle.write('{}\n'.format(listitem))

		with open(cnf.SCORES_FOLDER_PATH+'/'+cnf.SIL_SCORES_FILE_PATH, 'w') as filehandle:
			for listitem in silhouette_scores:
				filehandle.write('{}\n'.format(listitem))


	# get index position of top N values
	print("####kMeans compute top silhouette scores ####")
	top_sil_score_indexes = utils.get_top_n_indexes(silhouette_scores, cnf.TOP_MODELS)
	top_n_models = [kmeans_per_k[idx] for idx in top_sil_score_indexes]
	# top_n_models = []
	# for idx in top_sil_score_indexes:
	# 	top_n_models.append(kmeans_per_k[idx])



	# create image - kMeans
	# fig, ax = plt.subplots(figsize=(8, 5))
	# ax = sns.lineplot(x=range(1, len(inertias) + 1), y=inertias)
	# ax.set_title("kMeans - k for M4 time series selected features")
	# ax.set_xlabel("# of k")
	# ax.set_ylabel("Inertia")
	# viz.save_fig('kmeans_daily_series')

	# # silhouette_scores viz
	# fig, ax = plt.subplots(figsize=(8, 5))
	# ax = sns.lineplot(x=range(1, len(silhouette_scores) + 1),
	#                   y=silhouette_scores)
	# ax.set_title('Silhouette Scores')
	# ax.set_xlabel('k')
	# ax.set_ylabel('Silhouette Score')
	# viz.save_fig('kmeans_sil_score_daily_series')

    # Silhouette Diagrams
	# viz.create_sil_diagram(kmeans_per_k, features, top_sil_score_indexes,
	# 					"kmeans_sil_dia_daily_series", silhouette_scores)
	return top_n_models


def get_class_label(df: pd.DataFrame, model: KMeans,
					ts_index_l: List[str],
					k: int ) -> pd.DataFrame:
	"""
	retrieve the class label from the current classification model
	assumes column V1 contains the time series id
	Params:
	-------
	df : dataframe from the split operation in groupby
	model : model in which look up the label class
	ts_index_l : mapping table between time series id  and class label
	k : integer indicating k of kMeans model
	Returns:
	--------
	df : enhanced df
	"""
	ts_id = df.iloc[0]['V1']
	class_label_id = ts_index_l.index(ts_id)
	class_label = model.labels_[class_label_id]
	df['k'] = k
	df['class'] = class_label

	return df
