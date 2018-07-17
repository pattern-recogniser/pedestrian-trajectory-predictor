import numpy as np
import pandas as pd
import os
import shutil
import tarfile
import matplotlib


# PROJECT_PATH = '/Users/anjalikarimpil/Google Drive/Dissertation'
PROJECT_PATH = '/users/mscdsa2018/ask2/Projects'

# # TODO: Create folder structure in College System as the one here 
# datafile_path = '/Users/anjalikarimpil/Google Drive/Dissertation/Data/Social LSTM/'
# # datafile_path = '/users/mscdsa2018/ask2/Projects/Datasets/Social LSTM'


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
	THRESHOLD = pd.to_timedelta('00:20:00.00000')
	INPUT_SEQ_LENGTH = 20
	for df in df_list:
		df['date'], df['time'] = df[0].str.split('T', 1).str
		df[0] = pd.to_datetime(df[0], format="%Y-%m-%dT%H:%M:%S:%f") 
		df.columns = ['datetime', 'place', 'x_pos', 'y_pos', 'person_id', 'date', 'time']

		df.sort_values(['person_id','datetime'], inplace=True, ascending=True)
		df['time_lead'] = df.groupby(['person_id', 'date'])['datetime'].shift(-1)
		df['target_x'] = df.groupby(['person_id', 'date'])['x_pos'].shift(-1)
		df['target_y'] = df.groupby(['person_id', 'date'])['y_pos'].shift(-1)
		df['fl'] = np.where(abs(df['time_lead'] - df['datetime']) > THRESHOLD, 1, 0)
		df['traj_id'] = df['fl'].cumsum()

		data = df[['traj_id','x_pos','y_pos']]
		for i in range(1, INPUT_SEQ_LENGTH + 1):
		    data['x_'+str(i)] = data.groupby(['traj_id'])['x_pos'].shift(-i)
		    data['y_'+str(i)] = data.groupby(['traj_id'])['x_pos'].shift(-i)
		# Remove NAs 
		data = data.dropna()
