# train LSTM model using the derived time series data
#from pymongo import MongoClient
import json
import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from keras.constraints import max_norm

def create_dataset(file_path):
	f = open(file_path, 'r')
	dataset_dict = json.load(f)
	return dataset_dict

def extract_pid_list(dataset_dict):
	pid_list = []
	for k in dataset_dict.keys():
		if(dataset_dict[k]['pid'] not in pid_list):
			pid_list.append(dataset_dict[k]['pid'])
	return pid_list

class RocCallback(tf.keras.callbacks.Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
    	if(epoch%10==0):
	        y_pred_train = self.model.predict(self.x)
	        roc_train = roc_auc_score(self.y, y_pred_train[:,1])
	        y_pred_val = self.model.predict(self.x_val)
	        roc_val = roc_auc_score(self.y_val, y_pred_val[:,1])
	        print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
	        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def train_model(data_pid_dict, pid_list, random_state):
	#idset = list(data_pid_dict.keys())
	idset = pid_list
	train_idset, test_idset = train_test_split(np.array(idset), test_size=0.3, random_state=random_state)

	train_dataset = []
	train_labelset = []
	test_dataset = []
	test_labelset = []

	train_ordered_idset = []
	train_original_dataset = {}

	train_num = 0

	"""
	f = open('inclusion_ids_icd.json', 'r')
	legal_id_dict = json.load(f)
	f.close()
	"""
	
	for num_id in data_pid_dict.keys():
		pid = data_pid_dict[num_id]['pid']
		if(pid in train_idset):
			train_dataset.append(data_pid_dict[num_id]['data'])
			train_labelset.append(data_pid_dict[num_id]['label'])
			train_ordered_idset.append([train_num])
			train_num += 1
		else:
			test_dataset.append(data_pid_dict[num_id]['data'])
			test_labelset.append(data_pid_dict[num_id]['label'])

	"""
	for num_id in data_pid_dict.keys():
		if(num_id in train_idset):
			train_dataset.append(data_pid_dict[num_id]['data'])
			train_labelset.append(data_pid_dict[num_id]['label'])
			train_ordered_idset.append([train_num])
			train_num += 1
		else:
			test_dataset.append(data_pid_dict[num_id]['data'])
			test_labelset.append(data_pid_dict[num_id]['label'])
	print('train_num:', train_num)
	"""

	#X_resampled, y_resampled = RandomUnderSampler().fit_sample(train_ordered_idset, train_labelset)
	X_resampled, y_resampled = SMOTE().fit_sample(train_ordered_idset, train_labelset)

	train_ordered_idset = X_resampled
	train_labelset = y_resampled

	new_train_dataset = []

	for num_id in train_ordered_idset:
		new_train_dataset.append(train_dataset[num_id[0]])

	train_dataset = np.array(new_train_dataset)
	train_labelset = np.array(train_labelset)
	test_dataset = np.array(test_dataset)
	test_labelset = np.array(test_labelset)


	new_train_labelset = []
	new_test_labelset = []

	roc_callback = RocCallback(training_data=(train_dataset, train_labelset), validation_data=(test_dataset, test_labelset))

	"""
	lstm_layer_1 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(64),input_shape=(None,12))
	lstm_layer_2 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(32))
	model = tf.keras.models.Sequential([lstm_layer_1, lstm_layer_2, tf.keras.layers.BatchNormalization(), tf.keras.layers.Dense(10,activation='relu'), tf.keras.layers.Dense(2,activation='softmax')])
	"""

	model = tf.keras.Sequential()
	#model.add(tf.keras.layers.Embedding(12, 20, input_length=10))
	model.add(tf.keras.layers.LSTM(100, activation = 'relu', unroll=True, return_sequences=True, input_shape=(10,11)))
	model.add(tf.keras.layers.Dropout(0.5))
	#model.add(tf.keras.layers.LSTM(20,return_sequences=True))
	model.add(tf.keras.layers.LSTM(10, activation = 'relu', unroll=True))
	model.add(tf.keras.layers.Dropout(0.5))
	model.add(tf.keras.layers.Dense(2,activation='softmax'))
	model.summary()
	model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01, decay=0.9), metrics=['accuracy'])
	model.fit(train_dataset, train_labelset, validation_data=(test_dataset, test_labelset), batch_size=32, epochs=1000, verbose=0, callbacks=[roc_callback])
	y_test_pred = model.predict(test_dataset)[:,1]

	#print y_test_pred

	fpr, tpr, thres = roc_curve(test_labelset, y_test_pred)
	auc_value = auc(fpr, tpr)
	prec, rec, _ = precision_recall_curve(test_labelset, y_test_pred)
	auc_pr = auc(rec, prec)

	t = np.arange(0.0, 1.0, 0.01)
	diff = 1.0
	best_t = 0.5
	selected_metrics = []

	for i in range(t.shape[0]):
		dt = t[i] - 0.5
		sens, spec, PPV, acc = evaluation_per_class(test_labelset, np.round(y_test_pred-dt))
		if(abs(spec-0.95)<diff):
			best_t = t[i]
			selected_metrics = [sens, spec, PPV, acc]
			diff = abs(spec-0.95)

	print("best t:", best_t, "auc:", auc_value, "aupr:", auc_pr, "metrics:", selected_metrics)

	#return auc_value, auc_pr, selected_metrics


def evaluation_per_class(y, pred_y):
	correct_label_list = [0, 0]
	total_label_list = [0, 0]
	for i in range(len(y)):
		if(y[i]==pred_y[i]):
			correct_label_list[y[i]] += 1
		total_label_list[y[i]] += 1

	TP = correct_label_list[1]
	TN = correct_label_list[0]
	FP = total_label_list[0] - correct_label_list[0]
	FN = total_label_list[1] - correct_label_list[1]
	#print "TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN
	sens = float(TP)/float(TP+FN) if float(TP+FN) != 0.0 else 0.0
	spec = float(TN)/float(TN+FP) if float(TN+FP) != 0.0 else 0.0
	PPV = float(TP)/float(TP+FP) if float(TP+FP) != 0.0 else 0.0
	acc = float(TP+TN) / float(total_label_list[0]+total_label_list[1])
	return sens, spec, PPV, acc


def main():
	#tf.enable_eager_execution()
	#print("TensorFlow version: {}".format(tf.VERSION))
	print("Eager execution: {}".format(tf.executing_eagerly()))
	print("GPU Avaliable: ", tf.test.is_gpu_available())

	file_path = 'normalized_dataset_lstm_01_13.json'
	data_pid_dict = create_dataset(file_path)
	pid_list = extract_pid_list(data_pid_dict)
	train_model(data_pid_dict, pid_list, 0)

if __name__ == '__main__':
	main()
