from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Activation

LEARNING_RATE = 0.004
EPOCHS = 10
AMINOS = 'GALMFWKQESPVICYHRNDT'
char_to_int = {c: i for i, c in enumerate(AMINOS)}
int_to_char = {i: c for i, c in enumerate(AMINOS)}

train_losses = []
test_losses = []
test_accuracies = []
train_accuracies = []

POSITIVE = r"positive"
NEGATIVE = r"negative"


class MyModel(Model):
	def __init__(self):
		super(MyModel, self).__init__()
		self.flatten = Flatten()
		# self.d1 = Dense(256, activation='relu')
		# self.d2 = Dense(32, activation='relu')
		# self.d3 = Dense(2, activation='relu')
		self.d3 = Dense(512, activation='relu')
		self.d4 = Dense(256, activation='relu')
		self.d5 = Dense(128, activation='relu')
		self.d6 = Dense(64, activation='relu')
		self.d7 = Dense(32, activation='relu')
		self.d8 = Dense(2, activation='relu')
		self.a = Activation(activation='softmax')
	
	def call(self, x, **kwargs):
		x = self.flatten(x)
		# x = self.d1(x)
		# x = self.d2(x)
		x = self.d3(x)
		x = self.d4(x)
		x = self.d5(x)
		x = self.d6(x)
		x = self.d7(x)
		x = self.a(x)
		return x


model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


@tf.function
def train_step(model: MyModel, sequences, labels):
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
def test_step(model: MyModel, sequences, labels):
	# training=False is only needed if there are layers with different
	# behavior during training versus inference (e.g. Dropout).
	predictions = model(sequences, training=True)
	t_loss = loss_object(labels, predictions)
	test_loss(t_loss)
	test_accuracy(labels, predictions)


optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


def train_and_appraise(model: MyModel, train_ds, test_ds):
	"""
	Trains the data with the GD method and outputs the loss, accuracy, test loss and test accuracy achieved
	during the process.
	:param train_ds:
	:param test_ds:
	"""
	for epoch in range(EPOCHS):
		# Reset the metrics at the start of the next epoch
		reset_loss()
		
		for sequences, labels in train_ds:
			train_step(model, sequences, labels)
		
		for test_sequences, test_labels in test_ds:
			test_step(model, test_sequences, test_labels)
		
		print(f'Epoch {epoch + 1}, '
			  f'Loss: {train_loss.result()}, '
			  f'Accuracy: {train_accuracy.result() * 100}, '
			  f'Test Loss: {test_loss.result()}, '
			  f'Test Accuracy: {test_accuracy.result() * 100}')
		
		log_loss()


def reset_loss():
	train_loss.reset_states()
	train_accuracy.reset_states()
	test_loss.reset_states()
	test_accuracy.reset_states()


def log_loss():
	train_losses.append(train_loss.result())
	test_losses.append(test_loss.result())
	test_accuracies.append(test_accuracy.result())
	train_accuracies.append(train_accuracy.result())


def data_as_tensor():
	lst = [POSITIVE, NEGATIVE]
	X = []
	y = []
	for label in lst:
		with open(label) as file:
			for sample in file:
				amino_acid = get_one_hot(sample)
				if str(label) == POSITIVE:
					X.append(amino_acid)
					y.append(1)
				if str(label) == NEGATIVE:
					X.append(amino_acid)
					y.append(0)
	X = np.array(X).astype('float32')
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
	ret = (X_train, y_train), (X_test, y_test)
	return ret


def get_one_hot(sample):
	sample = sample.strip().upper()
	integer_encoded = np.array([char_to_int[char] for char in sample])
	amino_acid = []
	for value in integer_encoded:
		letter = np.zeros(len(AMINOS))
		letter[value] = 1
		amino_acid.append(letter)
	return np.array(amino_acid)


def get_data():
	"""
	Reads the data from the files, zips it with the correct label (positive or negative),
	 shuffles it and outputs the data as a tuple.
	:return: tuple of the train dataset and the test data set, each given as an array of tuples of
	the sample and the label
	"""
	(x_train, y_train), (x_test, y_test) = data_as_tensor()
	train_ds = tf.data.Dataset.from_tensor_slices(
		(x_train, y_train)).shuffle(10000).batch(32)
	test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
	return train_ds, test_ds


def get_9_mers():
	with open(r"predict") as pred:
		lines = pred.read().replace("\n", "").strip()
	encoded_k, k_mers = [], []
	
	for i in range(len(lines.strip()) - 8):
		partial = lines[i:i + 9]
		k_mers.append(partial)
		encoded_k.append(get_one_hot(partial))
	
	encoded_k = np.array(encoded_k)
	return encoded_k, k_mers


def result_stats(predicted):
	predictions = [x[0] < x[1] for x in predicted]
	predicted_true = [x for i, x in enumerate(nine_mers) if predictions[i]]
	predicted_false = [x for i, x in enumerate(nine_mers) if not predictions[i]]
	results = [(nine_mers[i], predictions[i],
				max(predicted[i])) for i in range(len(predicted))]
	# for r in results:
	#     if not r[1]:
	#         print(r)
	results.sort(key=lambda tup: tup[2], reverse=True)
	
	print(f"epoch count ={EPOCHS}")
	# print(f"positive rate={sum(1 if _ else 0 for _ in predictions) / len(predictions)}")
	count_ones = sum(1 if t[2] == 1 else 0 for t in results)
	print(f"certain 1.000 count = {count_ones} of {len(results)} ("
		  f"{100 * count_ones // len(results)}%)")
	
	certains = [r for r in results if r[2] == 1]
	certain_true = [r[0] for r in certains if r[1]]
	certain_false = [r[0] for r in certains if not r[1]]
	print(f"certain positives = {certain_true}")
	# print(f"certain negatives = {certain_false}")
	print()
	
	positives = ([r for r in results if r[1]])
	negatives = [r for r in results if not r[1]]
	print(f"positives = {positives}")
	
	# pd.DataFrame.from_records(results).to_excel("covid7.xlsx")
	
	print()
	print(f"{len(positives)} positives, {len(negatives)} negatives")
	print(f"{len(positives) * 100 / len(results)} % positives")


def plot():
	plt.xkcd()
	plt.figure(figsize=(8, 6))
	plt.subplot(221)
	plt.title("Test loss")
	plt.xlabel("Epochs")
	plt.plot(test_losses)
	plt.subplot(222)
	plt.xlabel("Epochs")
	plt.title("Train loss")
	plt.plot(train_losses)
	plt.subplot(223)
	plt.xlabel("Epochs")
	plt.title("Accuracy of test")
	plt.plot(test_accuracies)
	plt.subplot(224)
	plt.xlabel("Epochs")
	plt.title("Accuracy of train")
	plt.plot(train_accuracies)
	plt.suptitle(f"{EPOCHS} Epochs\n"
				 f"Learning rate = {LEARNING_RATE}")
	plt.show()
	plt.savefig(f"{EPOCHS}_{LEARNING_RATE}.png")


if __name__ == '__main__':
	train_ds, test_ds = get_data()
	
	train_and_appraise(model, train_ds, test_ds)
	encoded, nine_mers = get_9_mers()
	predict = model.predict(encoded)
	
	result_stats(predict)
	plot()


def circle_los(difference):
	"""
	:param difference: predicted subtracted from true label
	:return: circle-compatible loss value
	"""
	return 360 - abs(difference) if difference < 0 else abs(difference)
