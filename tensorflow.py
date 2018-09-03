import tensorflow as tf

tf.enable_eager_execution()
tfe = tf.contrib.eager


class Model(object):
    def __init__(self):
        self.W = tfe.Variable(5.)
        self.b = tfe.Variable(0.)

    def __call__(self, x):
        return self.W * x + self.b


model = Model()
assert model(3.).numpy() == 15.


def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))


TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000
inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noise = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

import matplotlib.pyplot as plt

plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.show()

print('Current loss: ')
print(loss(model(inputs), outputs).numpy())


def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)

    train(model, inputs, outputs, learning_rate=0.1)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
          (epoch, Ws[-1], bs[-1], current_loss))

# Let's plot it all
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()

'''class mlayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(mlayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel", shape=[input_shape[-1].value, self.num_outputs])

    def call(self, inputs):
        return tf.matmul(input, self.kernel)


layer = mlayer(10)
print(layer(tf.zeros([10, 5])))
print(layer.variables)'''

"""# 用tensorflow的function创建数据集
ds_tensors=tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])
print(ds_tensors)

import tempfile
_,filename=tempfile.mkstemp()
print(filename)

with open(filename,'w') as f:
    f.write(Line1
    Line2
    Line3)

ds_file=tf.data.TextLineDataset(filename)

for x in ds_tensors:
    print(x)
print()
for x in ds_file:
    print(x)
print()
ds_tensors=ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file=ds_file.batch(2)

for x in ds_tensors:
    print(x)
print()
for x in ds_file:
    print(x)
"""

# 指定运算设备
"""# Force execution on CPU
print("On CPU:")
times = time.time()
with tf.device("CPU:0"):
    x = tf.random_uniform([1000, 1000])
    x.device.endswith("CPU:0")
    timee = time.time()
    print(timee - times)

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
    times=time.time()
    with tf.device("GPU:0"):  # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
        x = tf.random_uniform([1000, 1000])
        x.device.endswith("GPU:0")
        timee=time.time()
        print(timee-times)
"""
