import tensorflow as tf


class MyKerasModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(tf.random.normal([1]))
        self.b = tf.Variable(tf.random.normal([1]))

    def __call__(self, xx, **kwargs):
        return self.w * xx + self.b


if __name__ == '__main__':
    # generate some data
    TRUE_W = 3.0
    TRUE_B = 2.0
    NUM_EXAMPLES = 1000
    x = tf.random.normal(shape=[NUM_EXAMPLES], seed=42)  # a vector of random x values
    noise = tf.random.normal(shape=[NUM_EXAMPLES], seed=42)  # generate some noise
    y = x * TRUE_W + TRUE_B + noise  # calculate y

    # run the model for above data
    # instantiate a model
    keras_model = MyKerasModel()
    print(type(x))

    # set the parameters of the model
    keras_model.compile(run_eagerly=False,
                        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                        loss=tf.keras.losses.mean_squared_error)

    # train the model for one batch containing the entire dataset
    keras_model.fit(x, y, epochs=25, batch_size=x.shape[0])


    # keras_model.save_weights("my_checkpoint")
