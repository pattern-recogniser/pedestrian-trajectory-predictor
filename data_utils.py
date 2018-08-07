'''
Author: Anjali Sebastian Karimpil
'''
import matplotlib
import numpy as np
import os
import pandas as pd
import pickle
import platform
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tarfile

import config

if platform.system() == 'Darwin':
	PROJECT_PATH = '/Users/anjalikarimpil/Google Drive/Dissertation'
else:
	PROJECT_PATH = '/users/mscdsa2018/ask2/Projects'
# PROJECT_PATH = '/Users/anjalikarimpil/Google Drive/Dissertation'
# PROJECT_PATH = '/users/mscdsa2018/ask2/Projects'


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
		if index >= file_count:
			break
		data_file_name = os.listdir(data_folder)[0]
		data_file_path = os.path.join(data_folder, data_file_name)
		try:
			df_list.append(pd.read_csv(data_file_path, sep=';', header=None, error_bad_lines=False))
		except Exception:
			print(data_file_name)
			problem_files.append(data_file_name)
	return df_list, problem_files

def process_files(df_list):
	'''
	Given a list of data frames, parsees it by splitting datetime field to 
	date and time. Assigns traj_id to person-position combination if the records are 
	nearby

	Returns - 
	a dataframe of all the processed data with trajectory_id, x_pos, y_pos, and
	x_pos and y_pos lead for config.INPUT_SEQ_LENGTH window size. Now we have all data
	for training in one row
	'''
	processed_df_list = []
	TIME_THRESHOLD = pd.to_timedelta('00:00:02.00000')
	POSITION_THRESHOLD = 500
	for df in df_list:
		df.columns = ['datetime', 'place', 'x_pos', 'y_pos', 'person_id']
		df['date'], df['time'] = df['datetime'].str.split('T', 1).str
		df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%dT%H:%M:%S:%f")

		df.sort_values(['person_id', 'datetime'], inplace=True, ascending=True)
		df.reset_index()
		df['time_lag'] = df.groupby(['person_id', 'date'])['datetime'].shift(1)
		df['person_lag'] = df['person_id'].shift(1)

		# flag 1 
		df['flag_1'] = (abs(df['time_lag'] - df['datetime']) > TIME_THRESHOLD) | \
						(df['person_lag'] != df['person_id'])
		df['traj_id'] = df['flag_1'].cumsum()
		df['x_lag'] = df.groupby(['traj_id'])['x_pos'].shift(1)
		df['y_lag'] = df.groupby(['traj_id'])['y_pos'].shift(1)

		df['x_diff'] = abs(df['x_pos'] - df['x_lag'])
		df['y_diff'] = abs(df['y_pos'] - df['y_lag'])
		df['flag_2'] = (df['x_diff'] > POSITION_THRESHOLD) | (df['y_diff'] > POSITION_THRESHOLD)
		df['flag_3'] = (df['flag_1'] | df['flag_2'])
		df['traj_id'] = df['flag_3'].cumsum()

		position_df = df[['traj_id', 'x_pos', 'y_pos']].copy()

		for i in range(1, config.INPUT_SEQ_LENGTH + config.OUTPUT_SEQ_LENGTH):
			position_df['x_' + str(i)] = position_df.groupby(['traj_id'])['x_pos'].shift(-i)
			position_df['y_' + str(i)] = position_df.groupby(['traj_id'])['y_pos'].shift(-i)
		# Remove NAs 
		position_df = position_df.dropna()
		processed_df_list.append(position_df)
	return processed_df_list


def next_batch(batch, batch_size, filt_X, filt_Y):
	'''
	Returns data in batches X-data and y_data
	'''
	x_batch = []
	y_batch = []
	for i in range(batch_size):
		x_batch.append(filt_X[batch * batch_size + i])
		y_batch.append(filt_Y[batch * batch_size + i])

	return x_batch, y_batch

def get_data(force_preprocess=False):
	'''
	Reads data from files, processes it and splits it into test, training and dev sets.
	Reads from pre-existing file from disc if it is present.
	Returns -
		6 numpy arrays of training, test, and dev data for input and labels.
		Input data is of the shape num_rows x config.NUM_DIMENSIONS x config.INPUT_SEQ_LENGTH and 
		Labels are of the shape num_rows x config.NUM_DIMENSIONS x config.OUTPUT_SEQ_LENGTH
	'''
	if force_preprocess or not os.path.exists('processed_file'):
		df_list, problem_files = read_files()
		data = process_files(df_list)
		data.to_pickle('processed_file')
	else:
		data = pd.read_pickle('processed_file')
	X_train, y_train, X_test, y_test, X_dev, y_dev = split_data(data)
	return X_train, y_train, X_test, y_test, X_dev, y_dev


def split_data(data, train=0.7, test=0.2, dev=0.1):
	'''
	Given data as a Pd dataframe, splits it into 3 numpy arrays (train, test and dev) 
	for input and labels using train_test_split from sklearn.model_selection
	Returns -
		3 sets of numpy arrays 1 each for train, test and dev, 
		with input/X data of shape num_rows x config.NUM_DIMENSIONS x config.INPUT_SEQ_LENGTH 
		and lables/Y data as numpy array of shape num_rows x config.NUM_DIMENSIONS 
	'''	
	X = np.array(data.iloc[:, 1:((config.INPUT_SEQ_LENGTH * config.NUM_DIMENSIONS) + 1)])
	Y = np.array(data.iloc[:, -(config.OUTPUT_SEQ_LENGTH * config.NUM_DIMENSIONS):])
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=(1 - train), 
		random_state = 1)
	X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=(dev / (test + dev)), 
		random_state = 1)
	return data_reshape(X_train, 'x'), data_reshape(y_train, 'y'), data_reshape(X_test, 'x'),\
	data_reshape(y_test, 'y'), data_reshape(X_dev, 'x'), data_reshape(y_dev, 'y')


def data_reshape(data, type):
	'''
	Takes in a numpy array and reshapes according to current requirements.
	For input
	Returns -
		numpy array of shape num_rows x config.NUM_DIMENSIONS x config.INPUT_SEQ_LENGTH 
		for input, or,
		numpy array of shape num_rows x config.NUM_DIMENSIONS for output
	'''
	if type == 'x':
		return data.reshape(-1, config.NUM_DIMENSIONS, config.INPUT_SEQ_LENGTH)
	else:
		return data.reshape(-1, config.NUM_DIMENSIONS)


def get_pedestrian_data():
	pickled_object = 'pedestrian_data.pickle'
	if os.path.exists(pickled_object):
		with open(pickled_object, 'rb') as handle:
			pedestrian_data = pickle.load(handle)
	else:
		pedestrian_data = PedestrianData()
		with open(pickled_object, 'wb') as handle:
			pickle.dump(pedestrian_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
	return pedestrian_data

class PedestrianData(object):
	'''
	'''
	def __init__(self):
		self.train_df_list, _ = read_files(file_count=1)
		self.train_df_list = process_files(self.train_df_list)
		self.row_counts = [len(df) for df in self.train_df_list]
		self.total_row_count = sum(self.row_counts)
		self.split_data()
		print ("***************** Split")

	def split_data(self):
		'''
		Splits the train_df_list into train_df_list, test_df and dev_df.
		Train - test - dev split is 98 - 1 - 1.

		Instance variables test_df and dev_df are set here.
		'''
		df = self.train_df_list[-1]
		test_sample_size = int(0.01 * self.total_row_count)
		dev_sample_size = int(0.01 * self.total_row_count)
		train_df, test_df = train_test_split(df, test_size=test_sample_size)
		train_df, dev_df = train_test_split(train_df, test_size=dev_sample_size)
		self.train_df_list[-1] = train_df
		self.test_df = test_df
		self.dev_df = dev_df
		self.row_counts[-1] = len(train_df)
		self.total_row_count = sum(self.row_counts)

	def data_normalise(self, data, type):
		if type == 'x':
			self.x_scaler = MinMaxScaler()
			x_train_scaled = self.x_scaler.fit_transform(data)
			return x_train_scaled
		else:
			self.y_scaler = MinMaxScaler()
			y_train_scaled = self.y_scaler.fit_transform(data)
			return y_train_scaled

	def data_denormalise(self, data, type):
		if type == 'x':
			return self.x_scaler.inverse_transform(data)
		else:
			return self.y_scaler.inverse_transform(data)


	def next_batch(self, batch_num, batch_size, mode='train'):
		'''
		'''
		# print ("{} batch {}".format(mode, batch_num))
		def _get_df_index_and_row(index):
			df_index = 0
			previous_cumulative_row_count = 0
			row = index
			for ix, cumulative_row_count in enumerate(cumulative_row_counts):
				if current_index < cumulative_row_count:
					df_index = ix
					if ix > 0:
						previous_cumulative_row_count = cumulative_row_counts[ix - 1]
						row = index % previous_cumulative_row_count
					break
			return df_index, row

		current_index = batch_num * batch_size
		next_index = (batch_num + 1) * batch_size
		if mode == 'train':
			current_index = current_index % self.total_row_count
			next_index = next_index % self.total_row_count
			cumulative_row_counts = np.array(self.row_counts).cumsum()
			current_df_index, current_index_df_row = _get_df_index_and_row(current_index)
			current_index_df = self.train_df_list[current_df_index]
			next_df_index, next_index_df_row = _get_df_index_and_row(next_index)
			next_index_df = self.train_df_list[next_df_index]
			if current_df_index == next_df_index:
				next_batch = current_index_df.iloc[current_index_df_row:next_index_df_row].copy()
			else:
				next_batch = pd.concat([current_index_df.iloc[current_index_df_row:],
										next_index_df.iloc[:next_index_df_row]])
		else:
			mode_df_dict = {
				'test': self.test_df,
				'dev': self.dev_df
			}
			df = mode_df_dict[mode]
			next_batch = df[current_index: next_index]
		X = np.array(next_batch.iloc[:, 1:((config.INPUT_SEQ_LENGTH * config.NUM_DIMENSIONS) + 1)])
		Y = np.array(next_batch.iloc[:, - (config.OUTPUT_SEQ_LENGTH * config.NUM_DIMENSIONS):])

		X = X.reshape(-1, config.INPUT_SEQ_LENGTH, config.NUM_DIMENSIONS)
		X = X.transpose([0, 2, 1])
		X = self.data_normalise(X.reshape((-1, config.INPUT_SEQ_LENGTH * config.NUM_DIMENSIONS)), 'x')
		X = X.reshape(-1, config.NUM_DIMENSIONS, config.INPUT_SEQ_LENGTH)

		Y = Y.reshape(-1, config.OUTPUT_SEQ_LENGTH, config.NUM_DIMENSIONS)
		Y = Y.transpose([0, 2, 1])
		Y = self.data_normalise(Y.reshape((-1, config.OUTPUT_SEQ_LENGTH * config.NUM_DIMENSIONS)), 'y')
		Y = Y.reshape(-1, config.NUM_DIMENSIONS, config.OUTPUT_SEQ_LENGTH)
		return X, Y
