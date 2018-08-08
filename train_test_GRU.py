'''
This file contains two functions. The first one builds an LSTM RNN model, and the second one used the model to train the parameters and
tests in test set. Before the functions, some variables are defined. These variables can be changed during model evaluation process. This
file can be run directly on terminal line:

python train_test_LSTM.py


Author: Mingchen Li
'''
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

import config
import data_utils


batch_size = 3
rnn_size = 400
num_layers = 3
output_size = 1
learning_rate = 0.05

logs_path = 'logs/' 

inputs = tf.placeholder('float', [None, config.NUM_DIMENSIONS, config.INPUT_SEQ_LENGTH], name = 'inputs')
targets = tf.placeholder('float', [None, config.NUM_DIMENSIONS, config.OUTPUT_SEQ_LENGTH], name = 'targets')

weight = tf.Variable(tf.constant(0.0025, shape=[rnn_size, config.NUM_DIMENSIONS * config.OUTPUT_SEQ_LENGTH]), name = 'weight')
bias = tf.Variable(tf.constant(0.1, shape=[config.NUM_DIMENSIONS * config.OUTPUT_SEQ_LENGTH]),name = 'bias')

# training_X, training_Y, dev_X, dev_Y, testing_X, testing_Y = get_data()                   
pedestrian_data = data_utils.get_pedestrian_data()
# pickled_object = '/Users/anjalikarimpil/Google Drive/Dissertation/Project code/' + \
#                     'pedestrian-trajectory-predictor/ped_data.pickle'
# with open(pickled_object, 'wb') as handle:
#     pickle.dump(pedestrian_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # with open(pickled_object, 'rb') as handle:
# #     pedestrian_data = pickle.load(handle)
'''
This function defines a RNN. It is an LSTM RNN for now, but if want to change to GRU, just change the
cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
into
cell = tf.nn.rnn_cell.GRUCell(rnn_size)
'''
def recurrent_neural_network(inputs, w, b):

    # cells = []
    # for _ in range(num_layers):
    #   cell = tf.contrib.rnn.GRUCell(rnn_size)  # Or LSTMCell(num_units)
    #   cells.append(cell)
    # cell = tf.contrib.rnn.MultiRNNCell(cells)
    cell = tf.nn.rnn_cell.GRUCell(rnn_size)
    initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

    outputs, last_State = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, 
    	dtype=tf.float32, scope="dynamic_rnn")
    print("Output shape is ", outputs.shape)
    outputs = tf.transpose(outputs, [1, 0, 2])
    # He has transposed here to facilitate gathering. Refer this
    # https://stackoverflow.com/questions/36764791/in-tensorflow-how-to-use-tf-gather-for-the-last-dimension
    # This is beacuse tensorflow doesnt let you to do just do array[,:-1]
    last_output = tf.gather(outputs, 1, name="last_output")
    print("Last Output shape is ", last_output.shape)
    prediction = tf.matmul(last_output, w) + b
    print("Prediction shape is ", prediction.shape)
    import ipdb; ipdb.set_trace()
    prediction = tf.reshape(prediction, (-1, config.NUM_DIMENSIONS, config.OUTPUT_SEQ_LENGTH))

    return prediction, outputs

'''
This function trains the model and tests its performance. After each iteration of the training, it prints out the number of iteration
and the loss of that iteration. When the training is done, prints out the trainingned parameters. After the testing, it prints out the test
loss and saves the predicted values and the ground truth values into a new .csv file so that it is each to compare the results and
evaluate the model performance. The file has two rows, with the first row being predicted values and second row being real values.
'''
def train_neural_network(inputs):
    
    prediction, pred_1 = recurrent_neural_network(inputs, weight, bias)
    print('prediction', prediction.shape)
    print('target shape', targets.shape)
    print('shape of result of tf.reduce_sum(prediction - targets, 0)', tf.reduce_sum(prediction - targets, 0).shape)

    cost = tf.reduce_sum(tf.square(tf.norm(prediction - targets, ord='euclidean', axis=1)))
    # The cost looks like it's the mean squared error, ie. sum of squared errors
    #cost = tf.square(tf.norm(tf.reduce_sum(prediction - targets, 0)))      # prediction: (len,2)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_epoch_loss = 1.0
        prev_train_loss = 0.0
        iteration = 0
        train_cost_list = []
        dev_cost_list = []
        while (abs(train_epoch_loss - prev_train_loss) > 1e-5):
            iteration += 1
            prev_train_loss = train_epoch_loss

            dev_epoch_loss = 0
            for batch in range(int(len(pedestrian_data.dev_df)/batch_size)):
                # x_batch, y_batch = data_utils.next_batch(batch, batch_size, dev_X, dev_Y)
                x_batch, y_batch = pedestrian_data.next_batch(mode='dev', batch_num=batch,
                                                              batch_size=batch_size)
                # print('x_batch is:', x_batch)
                # print('y_batch is', y_batch)
                data_feed = {inputs: x_batch, targets: y_batch}
                
                c, dev_predict, ouputs = sess.run([cost, prediction, pred_1], data_feed)
                print('================Dev predict')
                print('X_batch:', x_batch, 'y_batch', y_batch)
                print('predict final', dev_predict)
                print('dev: ', c)
                dev_epoch_loss += c/batch_size

            dev_epoch_loss = dev_epoch_loss / (int(len(pedestrian_data.dev_df) / batch_size))
            # training cost
            train_epoch_loss = 0
            for batch in range(int(pedestrian_data.total_row_count/batch_size)):
                # x_batch, y_batch = data_utils.next_batch(batch, batch_size, training_X, training_Y)
                x_batch, y_batch = pedestrian_data.next_batch(mode='train', batch_num=batch,
                                                              batch_size=batch_size)
                data_feed = {inputs: x_batch, targets: y_batch}
                _, c = sess.run([optimizer, cost], data_feed)
                
                #print('dev: ', c)
                train_epoch_loss += c / batch_size

            train_epoch_loss = train_epoch_loss / (int(pedestrian_data.total_row_count / batch_size))
            # denominator int(len(training_X)/batch_size) is number of batches
            # train_epoch_loss outside the loop gives per instance loss across all batches

            # dev cost
            
            '''
            data_feed = {inputs: dev_X, targets: dev_Y}
            _, dev_c = sess.run([prediction, cost], data_feed)
            dev_epoch_loss = dev_c/len(dev_X)
            '''



            train_cost_list.append(train_epoch_loss)
            dev_cost_list.append(dev_epoch_loss)
            print('Train iteration', iteration,'train loss:', train_epoch_loss)
            print('Train iteration', iteration,'dev loss:', dev_epoch_loss)
            test_epoch_loss = 0
            test_prediction = np.empty([len(pedestrian_data.test_df), 2, config.OUTPUT_SEQ_LENGTH])


            if iteration == 3:
                break
        iter_list = range(1, iteration + 1)
        plt.figure(1)
        plt.plot(iter_list, train_cost_list)
        plt.plot(iter_list, dev_cost_list)
        plt.title('iteration vs. epoch cost'    )
        # plt.show()

        # After the training, print out the trained parameters
        trained_w = sess.run(weight)
        trained_b = sess.run(bias)
        # print('trained_w: ', trained_w, 'trained_b: ', trained_b, 'trained_w shape: ', trained_w.shape)

        # Begin testing
        test_epoch_loss = 0
        test_prediction = np.empty([len(pedestrian_data.test_df), 2, config.OUTPUT_SEQ_LENGTH])
        '''
        data_feed = {inputs: testing_X, targets: testing_Y}
        pre, test_c = sess.run([prediction, cost], data_feed)
        test_prediction = pre
        test_epoch_loss = test_c/int(len(testing_X))
        '''
        test_prediction = np.empty([int(len(pedestrian_data.test_df) / batch_size) * batch_size, 2, config.OUTPUT_SEQ_LENGTH])
        y_batch_list = []
        x_batch_list = []
        for batch in range(int(len(pedestrian_data.test_df) / batch_size)):
            # x_batch, y_batch = data_utils.next_batch(batch, batch_size, testing_X, testing_Y)
            x_batch, y_batch = pedestrian_data.next_batch(mode='test', batch_num=batch,
                                                          batch_size=batch_size)
            y_batch_list.append(y_batch)
            x_batch_list.append(x_batch)
            data_feed = {inputs: x_batch, targets: y_batch}
            pre, c = sess.run([prediction, cost], data_feed)
            pre = np.array(pre)
            test_epoch_loss += c
            test_prediction[batch*batch_size : (batch+1)*batch_size, :] = pre
        testing_Y = np.concatenate(y_batch_list, axis=0)
        testing_X = np.concatenate(x_batch_list, axis=0)
        test_epoch_loss = test_epoch_loss/(int(len(pedestrian_data.test_df)/batch_size)*batch_size)
        
        print('Test loss:', test_epoch_loss)

        # Save predicted data and ground truth data into a .csv file.
        # import ipdb; ipdb.set_trace()
        testing_X = pedestrian_data.data_denormalise(
        	testing_X.reshape(-1, config.INPUT_SEQ_LENGTH * config.NUM_DIMENSIONS), 'x')
        testing_Y = pedestrian_data.data_denormalise(
        	testing_Y.reshape(-1, config.OUTPUT_SEQ_LENGTH * config.NUM_DIMENSIONS), 'y')
        test_prediction = pedestrian_data.data_denormalise(
        	test_prediction.reshape(-1, config.OUTPUT_SEQ_LENGTH * config.NUM_DIMENSIONS), 'y')
        test_all_data = np.hstack((testing_X, testing_Y, test_prediction))
        test_accuracy = mean_squared_error(testing_Y, test_prediction)
        print('Test accuracy', test_accuracy)
        # test_prediction = np.transpose(test_prediction)                                                             # The first row of file: prediction
        # testing_Y_array = np.transpose(np.array(testing_Y)[0 : int(len(testing_X)/batch_size)*batch_size, :])       # The second row of file: ground truth
        # test_prediction_and_real = np.vstack((test_prediction, testing_Y_array))
        # test_prediction_and_real = test_prediction_and_real.reshape(
        #             (test_prediction_and_real.shape[0] * test_prediction_and_real.shape[1], -1))
        np.savetxt("GRU_test_prediction_and_real.csv", test_all_data, delimiter = ",")


train_neural_network(inputs)


