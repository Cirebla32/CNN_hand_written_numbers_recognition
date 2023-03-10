# CNN_hand_written_numbers_recognition by ADJOVI Albéric | CHITOU Kader | IGABOUY CHOBLI Hermine | SOTOHOU Aristide

Command for running the project: <python3 blabla>

# Requirements

- Python 3
- Pip 3
- Tensorflow
- Keras
- Matplotlib
- Pillow
- Numpy

# Dataset 
Dataset is downloadable at https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
Then, you obtain a .npz file containing 4 files:
  - x_test.npy
  - x_train.npy
  - y_test.npy
  - y_train.npy
  
 x_train.npy contains 60,000 images (28x28) gathered in the same file. 
 To save the dataset as a real set of pictures, you need to proceed like the following:   
\>\>\> import matplotlib.pyplot as plt    
\>\>\> from tensorflow import keras   
\>\>\>    
\>\>\> #If you've download mnist.npz at https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz do this   
\>\>\> (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data('path/to/minis.npz')   
\>\>\>    
\>\>\> #If you've not download mnist.npz, it will be automatically downloaded by doing this   
\>\>\> (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()   
\>\>\>    
\>\>\> i = 1   
\>\>\> for im in X_train:   
... &nbsp; &nbsp; &nbsp; plt.imsave('Path/to/save/dataset/' + str(i) + '.jpg', im, format = 'jpg')    
... &nbsp; &nbsp; &nbsp; i += 1   
\>\>\>    
\>\>\> #Now, you can go to 'Path/to/save/dataset/' to see the 60,000 pictures of hand written numbers   
  
