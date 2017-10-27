import tarfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Hide messy TensorFlow warnings
import numpy as np 
from os import listdir
from scipy.io.wavfile import read
import tensorflow as tf
rawdata_path='./rawdata/'
filelist=listdir(rawdata_path)
print len(filelist)
lstm_size=4
batch_size=10
num_batches=3000
num_features=10
sample_len=num_features*num_batches
valid_sample_size=100

file_index = 0
num_10samples=100
def labeling(filename):
	tar = tarfile.open(rawdata_path+filename, "r:gz")
	wave_array=np.empty([0,sample_len])
	for member in tar.getmembers():
		if member.name.endswith('README'):
			readmefile = tar.extractfile(member)
			content=readmefile.readlines()[4]
			if content=='Gender: Male\n':
				label=np.array([[1,0]],np.float32)
			else:
				label=np.array([[0,1]],np.float32)
		if member.name.endswith('.wav'):
			wavfile=tar.extractfile(member)
			wave_single=read(wavfile)
			try:
				wave_array=np.append(wave_array,np.reshape(wave_single[1][0:sample_len],[1,sample_len]),axis=0)
			except Exception: pass
	return wave_array,np.repeat(label,batch_size,axis=0)

def generate_batch():
	global file_index
	filename =filelist[file_index]
	wave_array,label=labeling(filename)
	file_index=file_index+1
	return wave_array,label
def gen_validation_batch():
	global file_index_valid
	filename_valid =filelist[file_index_valid]
	wave_array,label=labeling(filename_valid)
	file_index_valid=file_index_valid+1
	while wave_array.shape!=(10,sample_len):
		filename_valid =filelist[file_index_valid]
		wave_array,label=labeling(filename_valid)
		file_index_valid=file_index_valid+1
	return wave_array,label



graph = tf.Graph()
with graph.as_default():
	dataset = tf.placeholder(tf.float32, [num_batches, batch_size, num_features])
	lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
	# Initial state of the LSTM memory.
	hidden_state = tf.zeros([batch_size, lstm.state_size[0]])
	current_state = tf.zeros([batch_size, lstm.state_size[1]])
	state = hidden_state, current_state
	loss = 0.0

	for i in range(num_batches):
		current_batch = dataset[i]
		# The value of state is updated after processing each batch of words.
		output, state = lstm(current_batch, state)
		if i==0:output_all=tf.expand_dims(output,2)
		else:
			output_all=tf.concat([output_all,tf.expand_dims(output,2)],axis=2)
		if i%1000==0:
			print 'Graph built for %s cells' % i


	target = tf.placeholder(tf.float32, [batch_size, 2]) 
	output_all=tf.expand_dims(output_all, 3)
	average_pool=tf.nn.avg_pool(output_all,ksize=[1,10,10,1])
	flat=tf.layers.flatten(average_pool)
	W = tf.Variable(tf.zeros([flat.shape[1], 2]))
	b = tf.Variable(tf.zeros([2]))

	print 'building logistic regression layer...'
	pred = tf.nn.softmax(tf.matmul(output_all, W) + b) # Softmax
	print 'building cross entropy layer... '
	# cross entropy
	cost = tf.reduce_mean(-tf.reduce_sum(target*tf.log(pred), reduction_indices=1))

	global_step = tf.Variable(0, trainable=False)
	print 'building optimizer...'
	optimizer_ADAM=tf.train.AdamOptimizer(1E-4).minimize(cost,global_step=global_step)
	init = tf.global_variables_initializer()

	print 'building validation function...'
	#validation
	correct_class = tf.equal(tf.argmax(pred,1), tf.argmax(target,1))
	accuracy = tf.reduce_mean(tf.cast(correct_class, tf.float32))

with tf.Session(graph=graph) as session:
	print('Initializing weights..')
	init.run()
	print('Initialized')

	for step in xrange(num_10samples):

		wave_array,label=generate_batch()
		print wave_array.shape
		try:
			wave_array_devi=np.reshape(wave_array,[batch_size,	num_batches,	num_features])
			dataset_feed=np.swapaxes(wave_array_devi,0,1)
			print 'batch number %s is generated successfully on tgz file: %s' % (step,filelist[file_index-1])
			feed_train = {dataset: dataset_feed, target: label}
			_,cost_val = session.run([optimizer_ADAM,cost], feed_dict=feed_train)
			print cost_val
		except Exception: pass

		if step%10==0:
			file_index_valid=1601
			accuracy_all=np.empty([1,valid_sample_size])
			for j in range(valid_sample_size):
				wave_valid,label_valid=gen_validation_batch()
				wave_valid_devi=np.reshape(wave_valid,[batch_size,	num_batches,	num_features])
				dataset_feed_valid=np.swapaxes(wave_valid_devi,0,1)
				feed_valid = {dataset: dataset_feed_valid, target: label_valid}
				accuracy_all[0,j] = session.run(accuracy, feed_dict=feed_valid)
			accuracy_average=np.mean(accuracy_all)
			print 'prediction accuracy after %s steps training: %s' % (step+1,	accuracy_average)	
		
	