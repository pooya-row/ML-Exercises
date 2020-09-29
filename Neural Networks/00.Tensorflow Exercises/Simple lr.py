import tensorflow as tf
import matplotlib.pyplot as plt

'''
The typical linear regression problem is modelled here using tf module.
No high level API is used here. 
'''


# defining the model class
class MyModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # randomly initialize the weight and bias
        self.w = tf.Variable(tf.random.normal([1]))
        self.b = tf.Variable(tf.random.normal([1]))

    def __call__(self, xx):
        return self.w * xx + self.b


# define a loss function
def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))


# define  training function given a callable model, inputs, outputs, and a learning rate
def train(model, xx, yy, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(yy, model(xx))

    # calculate the gradients with respect to W and b
    dw, db = t.gradient(current_loss, [model.w, model.b])

    # update w & b using gradient scaled by the learning rate
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)


# collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(int(input('Enter the number of epochs: ')))


# define training loop
def training_loop(model, xx, yy):
    # global Ws, bs, epochs
    for epoch in epochs:
        # update the model with the single giant batch
        train(model, xx, yy, learning_rate=0.1)

        # track this before I update
        Ws.append(model.w.numpy())
        bs.append(model.b.numpy())
        current_loss = loss(yy, model(xx))

        print("Epoch %3d:\t\tW= %1.3f, b= %1.3f, loss= %2.5f" %
              (epoch + 1, Ws[-1], bs[-1], current_loss))


if __name__ == '__main__':
    # generate some data
    TRUE_W = 3.0
    TRUE_B = 2.0
    NUM_EXAMPLES = 1000
    # A vector of random x values
    x = tf.random.normal(shape=[NUM_EXAMPLES], seed=42)
    # Generate some noise
    noise = tf.random.normal(shape=[NUM_EXAMPLES], seed=42)
    # Calculate y
    y = x * TRUE_W + TRUE_B + noise

    # run the model for above data
    # instantiate a model
    lr_model = MyModel()

    print("Random init to:\tW= %1.3f, b= %1.3f, loss= %2.5f" %
          (lr_model.w.numpy(), lr_model.b.numpy(), loss(y, lr_model(x))))

    # execute training loop
    training_loop(lr_model, x, y)

    # plot it
    plt.plot(epochs, Ws, "r", epochs, bs, "b")
    plt.plot([TRUE_W] * len(epochs), "r--", [TRUE_B] * len(epochs), "b--")
    plt.xlabel('epoch')
    plt.ylabel('Model Parameters')
    plt.legend(["W", "b", "True W", "True b"])
    plt.show()
