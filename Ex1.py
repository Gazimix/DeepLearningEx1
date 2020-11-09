import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model, backend
import numpy as np
from sklearn.model_selection import train_test_split

EPOCHS = 500


class MyModel(Model):
	def __init__(self):
		super(MyModel, self).__init__()
		self.conv1 = Conv2D(32, (1, 1), activation='relu')
		self.flatten = Flatten()
		self.d1 = Dense(256, activation='relu')
		self.d2 = Dense(64)
		self.d3 = Dense(2)
	
	def call(self, x, **kwargs):
		x = self.conv1(x)
		x = self.flatten(x)
		x = self.d1(x)
		x = self.d2(x)
		return self.d3(x)


model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


@tf.function
def train_step(sequences, labels):
	with tf.GradientTape() as tape:
		# training=True is only needed if there are layers with different
		# behavior during training versus inference (e.g. Dropout).
		predictions = model(sequences, training=True)
		loss = loss_object(labels, predictions)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	
	train_loss(loss)
	train_accuracy(labels, predictions)


@tf.function
def test_step(sequences, labels):
	# training=False is only needed if there are layers with different
	# behavior during training versus inference (e.g. Dropout).
	predictions = model(sequences, training=False)
	t_loss = loss_object(labels, predictions)
	test_loss(t_loss)
	test_accuracy(labels, predictions)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-05, amsgrad=True,
									name='Adam')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


def train_and_appraise(train_ds, test_ds):
	"""
	Trains the data with the GD method and outputs the loss, accuracy, test loss and test accuracy achieved
	during the process.
	:param train_ds:
	:param test_ds:
	"""
	for epoch in range(EPOCHS):
		# Reset the metrics at the start of the next epoch
		train_loss.reset_states()
		train_accuracy.reset_states()
		test_loss.reset_states()
		test_accuracy.reset_states()
		
		for sequences, labels in train_ds:
			train_step(sequences, labels)
		
		for test_sequences, test_labels in test_ds:
			test_step(test_sequences, test_labels)
		
		print(f'Epoch {epoch + 1}, '
			  f'Loss: {train_loss.result()}, '
			  f'Accuracy: {train_accuracy.result() * 100}, '
			  f'Test Loss: {test_loss.result()}, '
			  f'Test Accuracy: {test_accuracy.result() * 100}')


POSITIVE = r"positive"
NEGATIVE = r"negative"


def data_as_tensor():
	lst = [POSITIVE, NEGATIVE]
	X = []
	y = []
	A = ord("A")
	for label in lst:
		with open(label) as file:
			for sample in file:
				sample = sample.strip()
				chrs = []
				for c in sample:
					chrs.append(ord(c) - A)
				if str(label) == NEGATIVE:
					X.append(chrs)
					y.append(0)
				if str(label) == POSITIVE:
					X.append(chrs)
					y.append(1)
	X = np.array(X)
	y = np.array(y)
	X = X.reshape(X.shape + (1,)).astype('float32')
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
	ret = (tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train)), (
		tf.convert_to_tensor(X_test), tf.convert_to_tensor(y_test))
	return ret


def get_data():
	"""
	Reads the data from the files, zips it with the correct label (positive or negative), shuffles it
	and outputs the data as a tuple.
	:return: tuple of the train dataset and the test data set, each given as an array of tuples of
	the sample and the label
	"""
	(x_train, y_train), (x_test, y_test) = data_as_tensor()
	# Add a channels dimension
	x_train = x_train[..., tf.newaxis]
	x_test = x_test[..., tf.newaxis]
	train_ds = tf.data.Dataset.from_tensor_slices(
		(x_train, y_train)).shuffle(10000).batch(32)
	test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
	return train_ds, test_ds


if __name__ == '__main__':
	train_ds, test_ds = get_data()
	train_and_appraise(train_ds, test_ds)
