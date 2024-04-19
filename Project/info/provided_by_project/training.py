#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as K


from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.model_selection import train_test_split


# Adjust some of the settings for matplotlib plotting.
plt.rcParams.update(
    {
        "figure.figsize": [8.0, 4.0],
        "xtick.major.width": 1.5,
        "xtick.major.size": 10.0,
        "xtick.minor.size": 5.0,
        "ytick.major.width": 1.5,
        "ytick.major.size": 10.0,
        "ytick.minor.size": 5.0,
        "font.size": 12,
        "lines.linewidth": 2.0,
    }
)


def create_loss_plot(fit_history, output_file: str = "loss.png"):
    """Creates a loss plot for training and validation data."""
    plt.figure()
    plt.plot(fit_history.history["loss"], label="training")
    plt.plot(fit_history.history["val_loss"], label="validation")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)


def create_accuracy_plot(fit_history, output_file: str = "accuracy.png"):
    """Creates an accuracy plot for training and validation data."""
    plt.figure()
    plt.plot(fit_history.history["binary_accuracy"], label="training")
    plt.plot(fit_history.history["val_binary_accuracy"], label="validation")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)


def create_nn_output_plot(
    model, x_train, x_test, y_train, y_test, output_file: str = "nn_output.png"
):
    """Creates a plot of the NN output for training and test data.

    The function slices both training and test data according to the truth
    labels, so that the displayed spectra can be split into "background" and
    "signal" events (i.e. the two classes the model was trained on).

    """
    plt.figure()
    _, bins, _ = plt.hist(
        model.predict(x_test[y_test.astype(bool)]),
        bins=20,
        alpha=0.3,
        density=True,
        label="test signal",
    )
    plt.hist(
        model.predict(x_test[~y_test.astype(bool)]),
        bins=bins,
        alpha=0.3,
        density=True,
        label="test bg",
    )
    plt.hist(
        model.predict(x_train[y_train.astype(bool)]),
        bins=bins,
        density=True,
        histtype="step",
        label="train signal",
    )
    plt.hist(
        model.predict(x_train[~y_train.astype(bool)]),
        bins=bins,
        density=True,
        histtype="step",
        label="train bg",
    )
    plt.xlabel("NN output")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)


def main():
    """Trains a feed-forward DNN to perform binary classification."""
    with h5py.File("./data/output_signal.h5", "r") as file:
        signal_data = file["events"][:]

    with h5py.File("./data/output_bg.h5", "r") as file:
        bg_data = file["events"][:]

    print(
        f"Loaded signal and background input files with {len(signal_data):,} "
        f"and {len(bg_data):,} events, respectively."
    )

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

    signal = signal_data[input_list]
    bg = bg_data[input_list]

    print(
        "Restructured datasets to contain the following list of "
        f"observables: {signal.dtype.fields.keys()}."
    )

    signal = structured_to_unstructured(signal)
    bg = structured_to_unstructured(bg)

    print("Converted structured to unstructured datasets.")

    X = np.concatenate([signal, bg])
    y = np.concatenate(
        [
            np.ones(signal.shape[0], dtype=int),
            np.zeros(bg.shape[0], dtype=int),
        ]
    )

    x_train, x_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)
    x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, train_size=0.5)

    print("Performed the following train:validation:test split: 80:10:10.")

    preprocessing_layer = K.layers.Normalization()
    preprocessing_layer.adapt(x_train)

    model = K.Sequential(
        [
            preprocessing_layer,
            K.layers.Dense(50, activation="relu", name="hidden1"),
            K.layers.Dense(25, activation="relu", name="hidden2"),
            K.layers.Dense(10, activation="relu", name="hidden3"),
            K.layers.Dense(1, activation="sigmoid", name="output"),
        ]
    )

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.0002),
        loss=K.losses.BinaryCrossentropy(),
        metrics=[K.metrics.BinaryAccuracy()],
    )

    print("Initialised DNN and will start training now ...")

    early_stopping_callback = K.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.002,
        restore_best_weights=True,
        verbose=1,
    )

    fit_history = model.fit(
        x_train,
        y_train,
        batch_size=512,
        epochs=100,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping_callback],
    )

    print("Printing summary of the trained model:")
    print(model.summary())

    save_name = "my_model"
    print(
        f'Storing model with name "{save_name}" now. You can convert '
        'this to ONNX format with the "tf2onnx" command-line utility.'
    )
    model.save(save_name)

    create_loss_plot(fit_history, "loss.png")
    create_accuracy_plot(fit_history, "accuracy.png")
    create_nn_output_plot(model, x_train, x_test, y_train, y_test, "nn_output.png")

    print("Created plots of loss, accuracy, and NN output.")

    return 0


if __name__ == "__main__":
    main()
