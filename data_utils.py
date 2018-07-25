import numpy as np
import pandas as pd
import os
import shutil
import tarfile
import matplotlib
from sklearn.model_selection import train_test_split
import platform

if platform.system() == 'Darwin':
    PROJECT_PATH = '/Users/anjalikarimpil/Google Drive/Dissertation'
else:
    PROJECT_PATH = '/users/mscdsa2018/ask2/Projects'
# PROJECT_PATH = '/Users/anjalikarimpil/Google Drive/Dissertation'
# PROJECT_PATH = '/users/mscdsa2018/ask2/Projects'
INPUT_SEQ_LENGTH = 5

def get_data_folders():
	'''
	Returns list of folders where the data resides after extracting the 
	compressed files in the Data/Social LSTM folder. In case files
	are already extracted, returns the list of folders skipping extraction.

	Returns -
		data_folders - a list of folders containing the data.
	'''
	data_files_path = os.path.join(PROJECT_PATH, 'Data', 'Social LSTM')
	data_file_names = [file_name for file_name in os.listdir(data_files_path)
					   if file_name.startswith('al_position') and file_name.endswith('tar.gz')]
	
	data_folders = []
	for data_file_name in data_file_names:
		file_path = os.path.join(data_files_path, data_file_name)
		extracted_folder_path = file_path.split('.', 1)[0]
		if not os.path.exists(extracted_folder_path):
			with tarfile.open(file_path, 'r:gz') as tar_file:
				tar_file.extractall(extracted_folder_path)
		data_folders.append(extracted_folder_path)
	
	return data_folders


def read_files(file_count=1):
	'''
	Reads files from Data/Social LSTM folder and returns a list of dataframes.
	If no file_count is passed, only 1 file is read.
	Pass any number >= number of files to be read into this function.

	Returns -
		df_list - A list of pandas DataFrames, each DataFrame containing the
			data in one file.
	'''
	data_folders = get_data_folders()
	df_list = []
	problem_files = []
	for index, data_folder in enumerate(data_folders):
		print(index)
		if index >= file_count:
			break
		data_file_name = os.listdir(data_folder)[0]
		data_file_path = os.path.join(data_folder, data_file_name)
		try:
			df_list.append(pd.read_csv(data_file_path, sep=';', header=None))
		except Exception:
			print(data_file_name)
			problem_files.append(data_file_name)
	return df_list, problem_files

def process_files(df_list):
	'''
	Given a list of data frames, parsees it by splitting datetime field to 
	date and time,
	'''
	processed_df_list = []
	THRESHOLD = pd.to_timedelta('00:20:00.00000')
	for df in df_list:
		df['date'], df['time'] = df[0].str.split('T', 1).str
		df[0] = pd.to_datetime(df[0], format="%Y-%m-%dT%H:%M:%S:%f")
		df.columns = ['datetime', 'place', 'x_pos', 'y_pos', 'person_id', 'date', 'time']

		df.sort_values(['person_id','datetime'], inplace=True, ascending=True)
		df.reset_index()
		df['time_lag'] = df.groupby(['person_id', 'date'])['datetime'].shift(1)
		df['person_lag'] = df['person_id'].shift(1)
		time_threshold = pd.to_timedelta('00:00:02.00000')
		# flag 1 
		df['fl_1'] = np.where((abs(df['time_lag'] - df['datetime']) > time_threshold) |\
			(df['person_lag'] != df['person_id']), 1, 0)
		df['traj_id'] = df['fl_1'].cumsum()
		position_threshold = 500
		df['x_lag'] = df.groupby(['traj_id'])['x_pos'].shift(1)
		df['y_lag'] = df.groupby(['traj_id'])['y_pos'].shift(1)

		df['x_diff'] = abs(df['x_pos'] - df['x_lag'])
		df['y_diff'] = abs(df['y_pos'] - df['y_lag'])
		df['fl_2'] = np.where((df['x_diff'] > position_threshold) | \
			(df['y_diff'] > position_threshold), 1, 0)
		df['fl_3'] = np.where((df['fl_1'] | df['fl_2']), 1, 0)
		df['traj_id'] = df['fl_3'].cumsum()

		data = df[['traj_id','x_pos','y_pos']]

		for i in range(1, INPUT_SEQ_LENGTH + 1):
			data['x_'+str(i)] = data.groupby(['traj_id'])['x_pos'].shift(-i)
			data['y_'+str(i)] = data.groupby(['traj_id'])['x_pos'].shift(-i)
			# Remove NAs 
			data = data.dropna()
		processed_df_list.append(data)
	return pd.concat(processed_df_list, ignore_index=True)


def next_batch(batch, batch_size, filt_X, filt_Y):
	x_batch = []
	y_batch = []
	for i in range(batch_size):

		x_batch.append(filt_X[batch*batch_size+i])
		y_batch.append(filt_Y[batch*batch_size+i])

	return x_batch, y_batch

def split_data():
	df_list, problem_files = read_files()
	data = process_files(df_list)
	# data = pd.read_pickle('processed_file')
	train = 0.8
	test = 0.1
	dev = 0.1
	total_length = len(data)
	total_trajectories = np.ma.count(data['traj_id'].unique())
	train_ix = train * total_trajectories
	test_ix = test * total_trajectories
	X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 3:], data[['x_pos','y_pos']], 
														train_size = 0.8, test_size = 0.2, 
														random_state = 1)
	X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, 
														train_size = 0.5, test_size = 0.5, 
														random_state = 1)
	# return X_train, y_train, X_test, y_test, X_dev, y_dev

	return convert_and_reshape(X_train, 'x'), convert_and_reshape(y_train, 'y'),\
	convert_and_reshape(X_test, 'x'), convert_and_reshape(y_test, 'y'),\
	convert_and_reshape(X_dev, 'x'), convert_and_reshape(y_dev, 'y')

def convert_and_reshape(df, type):
	if type == 'x':
		return np.array(df).reshape(-1, 2, INPUT_SEQ_LENGTH)
	else:
		return np.array(df)


