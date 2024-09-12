# TODO: Constrain variables related to loan + Increase sparsity
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from alibi.explainers import CounterfactualRLTabular
from alibi.explainers.backends.cfrl_tabular import get_he_preprocessor
from alibi.models.tensorflow import ADULTEncoder, ADULTDecoder

from sklearn.neural_network import MLPClassifier

from utils import load_train_test, PARENT_DIR, TARGET, CURRENT_ID

### Constants
# General
HEAE_PATH = PARENT_DIR / 'meta' / 'Autoencoder_counterfactual'
RANDOM_SEED = 1234
# Constants - Autoencoder
FIT_AUTOENCODER = False
EPOCHS = 50              # epochs to train the autoencoder
HIDDEN_DIM = 128         # hidden dimension of the autoencoder
LATENT_DIM = 15          # define latent dimension
# Constants - Counterfactual
COEFF_SPARSITY = 0.5               # sparsity coefficient
COEFF_CONSISTENCY = 0.5            # consisteny coefficient
# TODO: Increase
TRAIN_STEPS = 500                # number of training steps -> consider increasing the number of steps
BATCH_SIZE = 100                   # batch size

### Load data, split into X, y
train, X_train, y_train, test, X_test, y_test = load_train_test()
# Define output dimensions.
output_dims = [len(X_train.columns)]
feature_names = X_train.columns.tolist()

### Fit NN
clf = MLPClassifier(
    learning_rate="invscaling", learning_rate_init=0.001, max_iter=100, 
    early_stopping=True, random_state=RANDOM_SEED
    )
clf.fit(X_train.values, y_train)

predictor = lambda x: clf.predict_proba(x)

### Obtain encoder/decoder using Tensorflow
heae_preprocessor, heae_inv_preprocessor = get_he_preprocessor(X=X_train, feature_names=feature_names, category_map={}, feature_types={})

if FIT_AUTOENCODER:
    # Get trainset for encoders
    # Define data preprocessor and inverse preprocessor. The invers preprocessor include datatype conversions.

    # Define trainset
    trainset_input = heae_preprocessor(X_train).astype(np.float32)
    trainset_outputs = {
        "output_1": trainset_input
    }

    trainset = tf.data.Dataset.from_tensor_slices((trainset_input, trainset_outputs))
    trainset = trainset.shuffle(1024).batch(128, drop_remainder=True)

    class HeAE(keras.Model):
        def __init__(self, encoder: keras.Model, decoder: keras.Model, **kwargs) -> None:
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder

        def call(self, x: tf.Tensor, **kwargs):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat

    # Define the heterogeneous auto-encoder.
    heae = HeAE(encoder=ADULTEncoder(hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM),
                decoder=ADULTDecoder(hidden_dim=HIDDEN_DIM, output_dims=output_dims))

    # Define loss functions.
    he_loss = [keras.losses.MeanSquaredError()]
    he_loss_weights = [1.]

    # Compile and fit model.
    heae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss=he_loss,
                loss_weights=he_loss_weights)
    heae.fit(trainset, epochs=EPOCHS)

    heae.save(HEAE_PATH, save_format="tf")
else:
    heae = keras.models.load_model(HEAE_PATH, compile=False)

explainer = CounterfactualRLTabular(predictor=predictor,
                                    encoder=heae.encoder,
                                    decoder=heae.decoder,
                                    latent_dim=LATENT_DIM,
                                    encoder_preprocessor=heae_preprocessor,
                                    decoder_inv_preprocessor=heae_inv_preprocessor,
                                    coeff_sparsity=COEFF_SPARSITY,
                                    coeff_consistency=COEFF_CONSISTENCY,
                                    category_map={},
                                    feature_names=feature_names,
                                    # ranges=ranges,
                                    immutable_features= ['CODE_GENDER'],
                                    train_steps=TRAIN_STEPS,
                                    batch_size=BATCH_SIZE,
                                    backend="tensorflow")
explainer = explainer.fit(X=X_train)

# Generate counterfactual instances

# Get first two predicted defaults in test dataset
test["pred_proba"] = clf.predict_proba(X_test)[:, 1]
first_default = test[test["pred_proba"] > 0.5].iloc[:2]
first_default.drop(columns="pred_proba", inplace=True)
X_first_default = first_default.drop(columns=[TARGET, CURRENT_ID])

# no constraints set
explanation = explainer.explain(X_first_default.values, np.array([0]), [])

# original vs counterfactuals
orig = np.concatenate([explanation.data['orig']['X'], explanation.data['orig']['class']],axis=1)
orig_pd = pd.DataFrame(orig, columns=feature_names+["Label"])

cf = np.concatenate([explanation.data['cf']['X'], explanation.data['cf']['class']], axis=1)
cf_pd = pd.DataFrame(cf, columns=feature_names + ["Label"])

# Differences - first default, using rule that probability higher than 10%
# Negative differences are again a sign of overfitting
diff_orig_cf = (cf_pd.iloc[0] - orig_pd.iloc[0]).sort_values(ascending=False)
abs_diff_orig_cf = np.abs(orig_pd.iloc[0] -cf_pd.iloc[0]).sort_values(ascending=False)
breakpoint()