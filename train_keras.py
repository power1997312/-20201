import reader
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

# prepare data
from visible import plot_image, plot_value_array

train_img, train_label = reader.load_mnist('datasets', kind='train')  # (60000,784) ,(60000, )
test_img, test_label = reader.load_mnist('datasets', kind='t10k')  # (10000,784) , (10000, )


"""show img[0] in colors
plt.figure()
plt.imshow(train_img[0].reshape((28,28)))
plt.colorbar()
plt.grid(False)
plt.show()"""

#transfrom into greyscale
train_img = train_img / 255.0
test_img = test_img / 255.0

"""show img [0..24] in grey-level pic
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_img[i].reshape((28,28)), cmap=plt.cm.binary)
    plt.xlabel(class_names[train_label[i]])
plt.show()"""

# classes of clothing
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#setup the model
model=keras.Sequential()
model.add(keras.layers.Dense(128,activation=tf.nn.relu,input_shape=(784,)))
model.add(keras.layers.Dense(10,activation=tf.nn.softmax))

"""
model=keras.Sequential(
    [keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
"""

#compile the model
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#feed and train
model.fit(train_img,train_label,batch_size=10,epochs=5)

#test model
test_loss,test_acc=model.evaluate(test_img,test_label)
print('Test_loss:',test_loss,'Test_acc:',test_acc)

#predict
predictions=model.predict(test_img)

# Plot several images with their predictions.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_label, test_img)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_label)
plt.show()