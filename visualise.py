'''
Author: Anjali Sebastian Karimpil
'''
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config



def show_pedestrian_path(observed_path, actual_path, predicted_path, filename):
	'''
	Given numpy arrays of shape 1, 2, INPUT_SEQ_LENGTH 
	'''
	df = format_data(observed_path, actual_path, predicted_path)
	plot_data(df, filename)

def plot_data(df, filename):
	'''
	Given a Pandas dataframe in the format of [x_pos, y_pos, type] plots each point 
	with a different colour based on type. type = ['observed', 'actual', 'predicted']
	'''
	ax = sns.scatterplot('x_pos', 'y_pos', data=df, hue='type')
	for i, txt in enumerate(df['label']):
	    ax.annotate(df['label'].iloc[i], (df['x_pos'].iloc[i], df['y_pos'].iloc[i]))
	ax.set(xlabel='', ylabel='', xticklabels='', yticklabels='')
	plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False)
	plt.savefig('graphs/'+ filename + '.png', bbox_inches='tight')
	plt.show()

def format_data(observed_path, actual_path, predicted_path):
	observed_path = observed_path.reshape(config.NUM_DIMENSIONS, config.INPUT_SEQ_LENGTH).transpose()
	actual_path = actual_path.reshape(config.NUM_DIMENSIONS, config.OUTPUT_SEQ_LENGTH).transpose()
	predicted_path = predicted_path.reshape(config.NUM_DIMENSIONS, config.OUTPUT_SEQ_LENGTH).transpose()
	df_obs = pd.DataFrame(dict(x_pos=observed_path[:, 0], y_pos=observed_path[:, 1], 
		type='observed', label=list(range(1, len(observed_path)+1))))
	df_act = pd.DataFrame(dict(x_pos=actual_path[:, 0], y_pos=actual_path[:, 1], type='actual',
		label=list(range(1, len(actual_path)+1))))
	df_pred = pd.DataFrame(dict(x_pos=predicted_path[:, 0], y_pos=predicted_path[:, 1],
		type='predicted', label=list(range(1, len(predicted_path)+1))))
	return pd.concat([df_obs, df_act, df_pred])
