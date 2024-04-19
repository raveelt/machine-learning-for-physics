# Training Models

Now that we have converted our ROOT ntuples to a flattened event format and into the HDF5 format in the [ROOT to HDF5 chapter](ROOT-to-HDF5.md), we're ready to train a machine learning tool with that data. 

In this particular example, we'll try to build a binary classifier to distinguish a "signal" class from a "background" class, where for the signal we use simulations of the $ttZ$ process in the 3-lepton channel. We don't use a fully realistic picture for the background contributions to that process, but instead focus on one single process: $WZ$ production. 

We'll demonstrate how such a classifier can be built using tensorflow's Keras API. Many of the machine-learning model parameters we show here are far from being optimal! Feel free to use this example and try to build a better classifier than ours!

On this page we'll go through the following points step by step:

- [How to load HDF5 files in python](#loading-data),
- [How to select features for the training](#feature-selection),
- [How to convert structured to unstructured data](#unstructuring-data),
- [How to perform train-validation-test splits](#train-validation-test-splits),
- [How to pre-process data before training](#data-pre-processing),
- [How to train models with tensorflow's Keras API](#model-training),
- [How to evaluate the performance of models](#performance-evaluation),

followed by a [code snippet](#complete-example) that provides a full working example.


## Python setup and libraries

For details you should check out the [Prerequisites chapter](Prerequisites.md). For all of the following code snippets, we're using the `tensorflow.keras` API. For loading our data, we'll need the `h5py` library. For some of the operations, we'll also need `numpy`. For the evaluation of the model, we'll need the `matplotlib.pyplot` module. For some individual functions, we'll also need to import from the `sklearn` package, but we'll import those on the spot when we need them. To import the globally used package, we should with something like:

```python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as K
```

We can check which devices are available for tensorflow to perform the training on with the following command. For this example, we ran on a computing node that had a GPU available:

=== "code"
    ```python
    from tensorflow.config import list_physical_devices
    list_physical_devices()
    ```
=== "output"
    ```
    [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
     PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    ```

Whatever device you run on, we strongly recommend to set up a python environment using Docker or Singularity, for example the `training` image that we already provide. Check out the [Prerequisites chapter](Prerequisites.md) for more information on that.


## Loading data

The very first step is to load the flattened event data we stored in the [ROOT to HDF5 chapter](ROOT-to-HDF5.md). Using the `h5py.File()` method, we can open HDF5 files and retrieve datasets from them (here: dataset named "events"). Wrapping the function into a `with` context ensures that it is properly closed as soon as we're done with the file handling: 

```python
with h5py.File("output_signal.h5", "r") as file:
    signal_data = file["events"][:]

with h5py.File("output_bg.h5", "r") as file:
    bg_data = file["events"][:]
```

The extra `[:]` operation ensures that the HDF5 dataset is directly converted to a numpy array. As a cross-check, we can look at the number of events we have just loaded:

=== "code"
    ```python
    len(signal_data), len(bg_data)
    ```
=== "output"
    ```
    (265526, 232503)
    ```


## Feature selection

Maybe we have created flattened event data with a rather large number of features, but then studied those features and decided to only use a subset of them for the model training (also check out the [More on Plotting chapter](Plotting.md) for creating plots of feature distributions). We can simply define a list of input features that we would like to reduce our data to:

```python
input_list = [
    "H_T",
    "jet_1_pt",
    "jet_2_pt",
    "lep_1_pt",
    "lep_2_pt",
    "n_bjets",
    "jet_1_twb",
    "jet_2_twb",
    "bjet_1_pt",
]
```

We can then apply this selection to the signal and background datasets with the following two lines: 

```python
signal_data = signal_data[input_list]
bg_data = bg_data[input_list]
```

We can check whether that worked by calling one of the two datasets:
=== "code"
    ```python
    signal_data
    ```
=== "output"
    ```
    array([(904.3012 , 199.8472  , 191.25633 ,  81.33561 , 44.6725  , 2, 1, 2, 191.25633 ),
           (403.37997,  45.659966,  43.86036 ,  85.853264, 11.965437, 1, 2, 1,  45.659966),
           (754.2562 , 157.29442 , 152.16808 , 121.10224 , 84.75848 , 3, 2, 5, 157.29442 ),
           ...,
           (644.6824 , 145.2465  ,  49.250496,  96.618   , 63.049805, 2, 5, 1, 145.2465  ),
           (469.57916, 103.52897 ,  76.34347 , 104.29997 , 96.93379 , 2, 3, 1, 103.52897 ),
           (562.9407 , 163.88626 ,  52.15347 ,  46.87375 , 34.996433, 2, 3, 1, 163.88626 )],
          dtype={'names': ['H_T', 'jet_1_pt', 'jet_2_pt', 'lep_1_pt', 'lep_2_pt', 'n_bjets', 'jet_1_twb', 'jet_2_twb', 'bjet_1_pt'], 'formats': ['<f4', '<f4', '<f4', '<f4', '<f4', 'u1', 'u1', 'u1', '<f4'], 'offsets': [50, 0, 4, 31, 35, 44, 24, 25, 27], 'itemsize': 54})
    ```

Great! We can see that the dataset now contains the 9 selected features defined in `input_list`. When looking at the complete example in the [ROOT to HDF chapter](ROOT-to-HDF5.md#complete-example), which was used to create these datasets, the original number of features was 19.


## 'Unstructuring' data

Structured data is great for storage, because we encode information about the different data columns directly into the dataset. We can also access individual features directly using the field names, e.g. we could get the pT values of the first jet by typing `data["jet_1_pt"]`. 

However, what we need for a machine-learning tool is to convert this to unstructured data. The `numpy` library provides a function for that which we apply to the two datasets:

```python
from numpy.lib.recfunctions import structured_to_unstructured
signal_data = structured_to_unstructured(signal_data)
bg_data = structured_to_unstructured(bg_data)
```

We can check whether that worked by calling one of the two datasets:

=== "code"
    ```python
    signal
    ```
=== "output"
    ```
    array([[904.3012  , 199.8472  , 191.25633 , ...,   1.      ,   2.      ,
            191.25633 ],
           [403.37997 ,  45.659966,  43.86036 , ...,   2.      ,   1.      ,
             45.659966],
           [754.2562  , 157.29442 , 152.16808 , ...,   2.      ,   5.      ,
            157.29442 ],
           ...,
           [644.6824  , 145.2465  ,  49.250496, ...,   5.      ,   1.      ,
            145.2465  ],
           [469.57916 , 103.52897 ,  76.34347 , ...,   3.      ,   1.      ,
            103.52897 ],
           [562.9407  , 163.88626 ,  52.15347 , ...,   3.      ,   1.      ,
            163.88626 ]], dtype=float32)
    ```

Now we have a 2-dimensional, unstructured numpy array with entries of type `float32`.


## Train-validation-test splits

When preparing data for a supervised classification task (e.g. binary classification), we need to provide the model with the true class labels of the data. However, we also don't want to introduce a bias to the training by feeding one class after the other into the model. So what we need is to concatenate both signal and background datasets and then to shuffle the two arrays that contain the data and the labels _simultaneously_.

In addition, to assess the performance of the model and the training quality, we usually don't train on the full dataset, but only on a fraction of it. The rest we reserve for validation (i.e. evaluating the model performance during training) and for testing (i.e. evaluating the model with a completely unseen dataset _after_ it has been trained). In this example, let's assume we want an 80-10-10 split for training/validation/testing.

Let's first start by concatenating the signal and background datasets:

```python
X = np.concatenate([signal_data, bg_data])
```

In a similar fashion, we can then create an array of class labels using the `numpy.zeros()` and the `numpy.ones()` methods:


```python
y = np.concatenate(
    [
        np.ones(signal_data.shape[0], dtype=int),
        np.zeros(bg_data.shape[0], dtype=int)
    ]
)
```

We can check the created structure by calling the array of class labels, which we expect to contain a bunch of zeros in the beginning and then a bunch of ones at the end:

=== "code"
    ```python
    y
    ```
=== "output"
    ```
    array([1, 1, 1, ..., 0, 0, 0])
    ```
    
Now we perform the 80-10-10 split of the combined dataset. The `sklearn` library provides a function for that, which _automatically shuffles_ the datset before applying the random split. The function expects a data and a label array, and then returns 4 objects: the data and the label arrays both split according to a given fraction value. To achieve the 80-10-10 split, we first separate 80% of the dataset for training. Then the remainder is split 50-50 for validation and testing:

```python
from sklearn.model_selection import train_test_split
x_train, x_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)
x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, train_size=0.5)
```

To validate that this worked as expected, we can check the shapes of the resulting datasets:

=== "code"
    ```python
    x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape
    ```
=== "output"
    ```
    ((398423, 9), (49803, 9), (49803, 9), (398423,), (49803,), (49803,))
    ```

> _Shuffling data can also be done by hand using all sorts of different algorithms, for example with the `random.shuffle()` method. For supervised tasks, one must make sure that both data and labels are shuffled simultaneously. The recommended procedure for such a case is thus to create an array of indices, shuffle these indices, and then use them to remap data and labels:_
> 
> ```python
> from random import shuffle
> shuffled_indices = list(range(len(x_train)))
> shuffle(shuffled_indices)
> x_train = x_train[shuffled_indices]
> y_train = y_train[shuffled_indices]
> ```


## Data pre-processing

To fully leverage the potential of machine-learning models, we should pre-process the data before feeding it into the model. For example, typical activation functions we use for layers of neural networks operate most efficiently if their input data is normalised. The scale of input data for individual layers is directly correlated with the scale of the input features, so it makes sense to apply scaling to the input features, too.

In the past this has caused some headache, because one would have to load external functions, e.g. from the `sklearn` library, to perform feature scaling. Popular examples include the `RobustScaler()` class of sklearn (see [documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)). The parameters of these scaling operations need to be stored in order to replicate and apply the model to any sort of input data. The 'headache' part of this was that one would have to store these parameters separately as they were not part of the actual machine-learning model. By now, tensorflow has introduced the `Normalization()` layer, which can be included into a tensorflow model directly (see [documentation here](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization)).

Let's leverage this new class that performs linear scaling of the input features. We first instantiate the pre-processing layer, then we can let it 'learn' its scaling parameters from the training dataset with the `layer.adapt()` method:

```python
preprocessing_layer = K.layers.Normalization()
preprocessing_layer.adapt(x_train)
```

We can check what the pre-processing layer 'learnt' by calling the `layer.get_weights()` method on it. We expect to see pairs of mean and standard-deviation values, one for each of the 9 features, and one overall bias term:

=== "code"
    ```python
    preprocessing_layer.get_weights()
    ```
=== "output"
    ```
    [array([677.0721   , 168.9923   ,  93.25703  , 131.72237  ,  67.663864 ,
          1.455125 ,   2.269965 ,   2.1896112, 112.771164 ], dtype=float32),
    array([1.2393858e+05, 1.6024768e+04, 4.7870273e+03, 1.2773097e+04,
            1.7167546e+03, 3.8355231e-01, 2.7525799e+00, 2.6958177e+00,
            1.0613261e+04], dtype=float32),
    398423]
    ```


## Model training

We are now ready to start the model training. In this particular example, we would like to perform binary classification to distinguish a class we call "signal" from another class we call "background" (here: $ttZ$ as the signal vs. $WZ$).

There's lots of optimisation one could do for such a binary classification setup. For this example, we'll use a feed-forward neural network with 3 hidden layers, with 50, 25 and 10 nodes, respectively. We'll design the network in such a way that it has one output node with a sigmoid activation function to represent class probabilities. The instantiation of the model is very straightforward with the [Keras Sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential):

```python
model = K.Sequential(
    [
        preprocessing_layer,
        K.layers.Dense(50, activation="relu", name="hidden1"),
        K.layers.Dense(25, activation="relu", name="hidden2"),
        K.layers.Dense(10, activation="relu", name="hidden3"),
        K.layers.Dense(1, activation="sigmoid", name="output"),
    ]
)
```

Noticed how we have included the pre-processing layer as the first layer of the network? This makes it very easy to ensure all our input data -- may it be during training or evaluation -- is pre-processed in the same way. One other thing to note: the model does _not_ contain an input layer -- it thus doesn't know about the dimension of the input when it is instantiated. The input dimensions will be determined with the first batch of data the model receives during training. Then, and only then, the model graph is actually built. 

So far we have only specified the model structure, but not its objective. For that, we use the `model.compile()` method and specify what it's actually meant to do: we'd like to perform binary classification, thus we choose binary cross-entropy as the loss function (implemented in the Keras API [here](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy)). In addition to the loss, we can define other performance metrics to monitor -- in this example we choose the classification accuracy (implemented in the Keras API [here](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryAccuracy)). Last but not least, we specify the optimiser for this objective (here: Adam, implemented in the Keras API [here](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)). The entire compile method then becomes:

```python
model.compile(
    optimizer=K.optimizers.Adam(learning_rate=0.0002),
    loss=K.losses.BinaryCrossentropy(),
    metrics=[K.metrics.BinaryAccuracy()],
)
```
> _Note that we have manually set the learning rate of the optimiser in this example. We chose that value as the default lead to a more unstable loss evolution. Learning rates and other optimisation steps of the model should **always** be evaluated on a case-by-case basis -- there is no universal recipe._

To decide when the model reaches its 'optimal' performance, let's look at the value of the loss function on the validation dataset. To avoid doing that manually, we can use early stopping (implemented in the Keras API [here](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)). We specify that the early stopping should only kick in if there have not been improvements for 10 epochs, and it should check for improvements $\Delta > 0.002$:

```python
early_stopping_callback = K.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    min_delta=0.002, 
    restore_best_weights=True,
    verbose=1,
)
```

Now we are ready to fit the model to our training data. The method expects the actual training data, the early-stopping callback, the validation data, as well as the maximum number of epochs and the batch size as arguments. We can follow the training progress through the output:

=== "code"
    ```python
    fit_history = model.fit(
        x_train,
        y_train,
        batch_size=512,
        epochs=100,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping_callback],
    )
    ```
=== "output"
    ```
    Epoch 1/100
    779/779 [==============================] - 11s 2ms/step - loss: 0.5121 - binary_accuracy: 0.7595 - val_loss: 0.4412 - val_binary_accuracy: 0.8030
    Epoch 2/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4363 - binary_accuracy: 0.8030 - val_loss: 0.4295 - val_binary_accuracy: 0.8063
    Epoch 3/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4301 - binary_accuracy: 0.8057 - val_loss: 0.4266 - val_binary_accuracy: 0.8080
    Epoch 4/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4278 - binary_accuracy: 0.8068 - val_loss: 0.4250 - val_binary_accuracy: 0.8083
    Epoch 5/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4262 - binary_accuracy: 0.8079 - val_loss: 0.4237 - val_binary_accuracy: 0.8096
    Epoch 6/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4250 - binary_accuracy: 0.8089 - val_loss: 0.4229 - val_binary_accuracy: 0.8097
    Epoch 7/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4241 - binary_accuracy: 0.8092 - val_loss: 0.4221 - val_binary_accuracy: 0.8102
    Epoch 8/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4233 - binary_accuracy: 0.8097 - val_loss: 0.4218 - val_binary_accuracy: 0.8104
    Epoch 9/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4225 - binary_accuracy: 0.8100 - val_loss: 0.4212 - val_binary_accuracy: 0.8105
    Epoch 10/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4221 - binary_accuracy: 0.8103 - val_loss: 0.4213 - val_binary_accuracy: 0.8100
    Epoch 11/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4216 - binary_accuracy: 0.8105 - val_loss: 0.4211 - val_binary_accuracy: 0.8105
    Epoch 12/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4212 - binary_accuracy: 0.8109 - val_loss: 0.4201 - val_binary_accuracy: 0.8107
    Epoch 13/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4209 - binary_accuracy: 0.8109 - val_loss: 0.4201 - val_binary_accuracy: 0.8110
    Epoch 14/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4205 - binary_accuracy: 0.8112 - val_loss: 0.4194 - val_binary_accuracy: 0.8110
    Epoch 15/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4202 - binary_accuracy: 0.8114 - val_loss: 0.4193 - val_binary_accuracy: 0.8118
    Epoch 16/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4200 - binary_accuracy: 0.8114 - val_loss: 0.4192 - val_binary_accuracy: 0.8117
    Epoch 17/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4197 - binary_accuracy: 0.8116 - val_loss: 0.4194 - val_binary_accuracy: 0.8118
    Epoch 18/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4195 - binary_accuracy: 0.8116 - val_loss: 0.4185 - val_binary_accuracy: 0.8124
    Epoch 19/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4193 - binary_accuracy: 0.8118 - val_loss: 0.4184 - val_binary_accuracy: 0.8126
    Epoch 20/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4191 - binary_accuracy: 0.8116 - val_loss: 0.4192 - val_binary_accuracy: 0.8117
    Epoch 21/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4188 - binary_accuracy: 0.8119 - val_loss: 0.4183 - val_binary_accuracy: 0.8123
    Epoch 22/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4187 - binary_accuracy: 0.8117 - val_loss: 0.4186 - val_binary_accuracy: 0.8121
    Epoch 23/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4186 - binary_accuracy: 0.8120 - val_loss: 0.4182 - val_binary_accuracy: 0.8123
    Epoch 24/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4184 - binary_accuracy: 0.8119 - val_loss: 0.4182 - val_binary_accuracy: 0.8124
    Epoch 25/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4184 - binary_accuracy: 0.8119 - val_loss: 0.4177 - val_binary_accuracy: 0.8125
    Epoch 26/100
    779/779 [==============================] - 2s 2ms/step - loss: 0.4182 - binary_accuracy: 0.8118 - val_loss: 0.4180 - val_binary_accuracy: 0.8117
    Epoch 26: early stopping
    Restoring model weights from the end of the best epoch: 16.
    ```
    
Great, so the model stopped early and restored its weights from the best epoch! Now that the model has seen data, we can print a quick summary using the `model.summary()` method:

=== "code"
    ```python
    model.summary()
    ```
=== "output"
    ```
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     norm (Normalization)         (None, 9)                19        
                                                                     
     hidden1 (Dense)             (None, 50)                500       
                                                                     
     hidden2 (Dense)             (None, 25)                1275      
                                                                     
     hidden3 (Dense)             (None, 10)                260       
                                                                     
     output (Dense)              (None, 1)                 11        
                                                                     
    =================================================================
    Total params: 2,065
    Trainable params: 2,046
    Non-trainable params: 19
    _________________________________________________________________
    ```


The only thing that's left now is to save the model. The following function stores the model in a tensorflow-internal format. This format can also easily be loaded again with the `K.models.load_model()` method:

=== "code"
    ```python
    model.save("my_model")
    ```
=== "output"
    ```
    INFO:tensorflow:Assets written to: my_model/assets
    ```

However, for injecting a trained model back into our ntuples, we need a more universal format then that. Over the last few years, ONNX (_Open Neural Network Exchance Format_) has become the most popular choice, and a c++ interface to ONNX called ONNX Runtime is available in the ATLAS analysis software releases (AthAnalysis/AnalysisBase/AnalysisTop).

To convert the tensorflow-internal format to ONNX format, we can use the [tf2onnx package](https://github.com/onnx/tensorflow-onnx) command line utility. This utility is also included in the `training` Docker/Singularity image. The command to convert the model is very simple:

```bash
python -m tf2onnx.convert --saved-model my_model --output model.onnx
```

The [Injection to Ntuples chapter](ONNX_AT.md) explains how a model stored in the ONNX format can be loaded into c++ code and thus injected into analysis ntuples.


## Performance evaluation

The following code snippets provide some very simple evaluation plots of the training. More on plotting, especially on comparing the performance of different models, can also be found in the [More on Plotting chapter](Plotting.md).

The following code produces a plot of the loss evolution for training and validation:

```python
plt.plot(fit_history.history["loss"], label="training")
plt.plot(fit_history.history["val_loss"], label="validation")
plt.xlabel("Number of epochs")
plt.ylabel("Loss value")
plt.legend()
plt.tight_layout()
plt.show()
```

![png](plots_training/loss.png)

The following code produces a plot of the accuracy evolution per epoch for training and validation:

```python
plt.plot(fit_history.history["binary_accuracy"], label="training")
plt.plot(fit_history.history["val_binary_accuracy"], label="validation")
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy value")
plt.legend()
plt.tight_layout()
plt.show()
```
 
![png](plots_training/accuracy.png)

The following code plots the distribution of the neural-network output node for both training and test data to check for possible differences between the two. The datasets are sliced according to their truth labels (i.e. the two classes).

```python
_, bins, _ = plt.hist(model.predict(x_test[y_test.astype(bool)]), bins=20, alpha=0.3, density=True, label="test signal")
plt.hist(model.predict(x_test[~y_test.astype(bool)]), bins=bins, alpha=0.3, density=True, label="test bg")
plt.hist(model.predict(x_train[y_train.astype(bool)]), bins=bins, density=True, histtype="step", label="train signal")
plt.hist(model.predict(x_train[~y_train.astype(bool)]), bins=bins, density=True, histtype="step", label="train bg")
plt.xlabel("NN output")
plt.legend()
plt.tight_layout()
plt.show()
```
    
![png](plots_training/nn_output.png)
    
