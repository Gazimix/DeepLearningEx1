import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Activation

EPOCHS = 3
AMINOS = 'GALMFWKQESPVICYHRNDT'
char_to_int = {c: i for i, c in enumerate(AMINOS)}
int_to_char = {i: c for i, c in enumerate(AMINOS)}


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(256, activation='relu')
        self.d2 = Dense(32, activation='relu')
        self.d3 = Dense(2, activation='relu')
        self.a = Activation(activation='softmax')

    def call(self, x, **kwargs):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.a(x)
        return x

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y, sample_weight = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight,
                                      regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)

        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def train_and_appraise(self, train_ds, test_ds, weights):
        """
        Trains the data with the GD method and outputs the loss, accuracy, test loss and test accuracy achieved
        during the process.
        :param train_ds:
        :param test_ds:
        """
        for epoch in range(EPOCHS):
            for sequences, labels in train_ds:
                self.train_step((sequences, labels, weights))

            for test_sequences, test_labels in test_ds:
                self.test_step((test_sequences, test_labels))

            print(self.losses)


POSITIVE = r"positive"
NEGATIVE = r"negative"


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
    Reads the data from the files, zips it with the correct label (positive or negative), shuffles it
    and outputs the data as a tuple.
    :return: tuple of the train dataset and the test data set, each given as an array of tuples of
    the sample and the label
    """
    (x_train, y_train), (x_test, y_test) = data_as_tensor()
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    weights = np.array([1 if y else 3 / 27 for y in y_train])
    weights = np.expand_dims(weights,axis=1)
    return train_ds, test_ds, weights


def get_9_mers():
    string = ""
    with open(r"predict") as pred:
        lines = pred.read().replace("\n", "").strip()
    encoded_k, nine_mers = [], []

    for i in range(len(lines.strip()) - 8):
        partial = lines[i:i + 9]
        nine_mers.append(partial)
        encoded_k.append(get_one_hot(partial))

    encoded_k = np.array(encoded_k)
    return encoded_k, nine_mers


if __name__ == '__main__':
    aggregated = {}
    model = MyModel()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='mae')
    model.train_and_appraise(*get_data())

    encoded, nine_mers = get_9_mers()
    predict = model.predict(encoded)

    predictions = [x[0] > x[1] for x in predict]

    predicted_true = [x for i, x in enumerate(nine_mers) if predictions[i]]
    predicted_false = [x for i, x in enumerate(nine_mers) if not predictions[i]]
    results = [(nine_mers[i], predictions[i], max(predict[i])) for i in
               range(len(predict))]

    results.sort(key=lambda tup: tup[2], reverse=True)
    print(f"epoch count ={EPOCHS}")
    # print(f"positive rate={sum(1 if _ else 0 for _ in predictions) / len(predictions)}")
    count_ones = sum(1 if t[2] == 1 else 0 for t in results)
    print(f"certain 1.000 count = {count_ones} of {len(results)} ("
          f"{100 * count_ones // len(results)}%)")

    for t in results:
        name, sign, p = t
        if not sign:
            p *= -1
        if name in aggregated:
            aggregated[name].append(p)
        else:
            aggregated[name] = [p]
    certains = [r for r in results if r[2] == 1]
    certain_true = [r for r in certains if r[2]]
    certain_false = [r for r in certains if not r[2]]
    print(f"certain positives = {certain_true}")
    print(f"certain negavies = {certain_false}")
    print()


def circle_loss(difference):
    """
    :param difference: predicted subtracted from true label
    :return: circle-compatible loss value
    """
    return 360 - abs(difference) if difference < 0 else abs(difference)
