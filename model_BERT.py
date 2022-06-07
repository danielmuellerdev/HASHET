from typing import Any
from pathlib import Path

import tensorflow as tf
import constants as c
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from tensorflow.keras.optimizers import Adam
import keras.layers as layers
from transformers import BertTokenizer, TFBertModel
from keras.layers.core import Dense

import word_embedding_model
from word_embedding_model import WordEmbeddingModel

SCALE_1 = 2 / 3


def _cosine_distance(y_true, y_pred):
    return tf.compat.v1.losses.cosine_distance(tf.nn.l2_normalize(y_pred, 0), tf.nn.l2_normalize(y_true, 0), dim=0)


def _create_mlp(n_hidden: int, encoder_output: Any, out_channels: int) -> Any:
    h1 = Dense(int(n_hidden), activation="relu")(encoder_output)
    h2 = Dense(int(n_hidden * SCALE_1), activation="relu")(h1)
    h3 = Dense(int(n_hidden * SCALE_1 * SCALE_1), activation="relu")(h2)

    output = Dense(out_channels, activation='linear')(h3)

    return output

def _create_sentence_encoder(sequence_len: int):
    input1 = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='input_ids')
    input2 = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='token_type_ids')
    input3 = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='attention_mask')

    encoder_inputs = [input1, input2, input3]

    encoder_layer = TFBertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    for layer in encoder_layer.layers[:]:
        layer.trainable = False

    encoder_outputs = encoder_layer(encoder_inputs)
    encoder_output = encoder_outputs[1] # token_type_ids

    return encoder_inputs, encoder_output

def transfer_and_fine_tune(
    sentences_train, sentences_test, targets_train, targets_test,
    model_verbosity_level: int = c.ONE_LINE_PER_EPOCH,
    early_stopping_verbosity_level: int = c.LOG_LEVEL,
    patience: int = c.PATIENCE,
    max_epochs: int = c.MAX_EPOCHS,
    batch_size: int = c.BATCH_SIZE,
    transfer_learning_model_weights_file_path: Path = c.TRANSFER_LEARNING_MODEL_WEIGHTS_FILE_PATH,
    fine_tuning_model_weights_file_path: int = c.FINE_TUNING_MODEL_WEIGHTS_FILE_PATH,
    sequence_len: int = 100,
    bert_model_name: str = 'bert-base-uncased'):
    """
        Create and train multi level perceptron with Keras API and save it.
    """
    print("BERT version")
    print("TRANSFER LEARNING STEP:")

    tokenizer = BertTokenizer.from_pretrained(bert_model_name) 
    train_encodings = tokenizer(sentences_train.tolist(), truncation=True, padding='max_length',
                                max_length=sequence_len)
    test_encodings = tokenizer(sentences_test.tolist(), truncation=True, padding='max_length', max_length=sequence_len)

    encoder_inputs, encoder_output = _create_sentence_encoder(sequence_len)

    mlp_output = _create_mlp(n_hidden=encoder_output.shape[1], encoder_output=encoder_output, out_channels=len(targets_train[0]))
    
    model = keras.Model(inputs=encoder_inputs, outputs=mlp_output)
    model.summary()

    model.compile(loss=_cosine_distance, optimizer='adam', metrics=['cosine_proximity'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=early_stopping_verbosity_level, patience=patience)

    best_weights_file = transfer_learning_model_weights_file_path
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=model_verbosity_level,
                         save_best_only=True, save_weights_only=True)

    x = [
        np.array(train_encodings["input_ids"]), 
        np.array(train_encodings["token_type_ids"]), 
        np.array(train_encodings["attention_mask"])
    ]
    val_data = (
        [
            np.array(test_encodings["input_ids"]), 
            np.array(test_encodings["token_type_ids"]),
            np.array(test_encodings["attention_mask"])
        ],
        targets_test
    )

    model.fit(x, y=targets_train, validation_data=val_data, batch_size=batch_size, epochs=max_epochs, 
        verbose=model_verbosity_level, callbacks=[es, mc])

    print("FINE TUNING STEP:")
    model.load_weights(transfer_learning_model_weights_file_path)

    model.trainable = True
    model.compile(loss=_cosine_distance, optimizer=Adam(3e-5), metrics=['cosine_proximity'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=early_stopping_verbosity_level, patience=patience)

    best_weights_file = fine_tuning_model_weights_file_path
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=model_verbosity_level,
                         save_best_only=True, save_weights_only=True)
    
    model.fit(x, y=targets_train, validation_data=val_data, batch_size=batch_size, epochs=max_epochs, 
        verbose=model_verbosity_level, callbacks=[es, mc])


def predict_top_k_hashtags(word_emb_model: WordEmbeddingModel, sentences, k):
    """
        Predict hashtags for input sentence embeddings (embeddings_list)

        :param1 sentences: sentences.
        :param2 k: number of hashtags to predict for each sentence.
        :returns results: list of list of (hashtag, likelihood):
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sequence_len = 100
    sentences_encodings = tokenizer(sentences.tolist(), truncation=True, padding='max_length',
                                    max_length=sequence_len)
    # Model reconstruction
    input1 = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='input_ids')
    input2 = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='token_type_ids')
    input3 = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='attention_mask')

    input = [input1, input2, input3]  # {"input_word_ids": input1, "input_mask": input2, "input_types_ids": input3}
    
    encoderlayer = TFBertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    for l in encoderlayer.layers[:]:
        l.trainable = False
    encoderoutputs = encoderlayer(input)

    encoderoutput = encoderoutputs[1]
    
    n_hidden = encoderoutput.shape[1]
    h1 = Dense(int(n_hidden), activation="relu")(encoderoutput)
    h2 = Dense(int(n_hidden * SCALE_1), activation="relu")(h1)
    h3 = Dense(int(n_hidden * SCALE_1 * SCALE_1), activation="relu")(h2)

    out = Dense(c.LATENT_SPACE_DIM, activation='linear')(h3)

    model = keras.Model(inputs=input, outputs=out)
    model.summary()

    # compile model
    model.compile(loss=_cosine_distance, optimizer=Adam(3e-5),
                  metrics=['cosine_proximity'])  # Load weights into the new model
    model.load_weights(c.MODEL_WEIGHTS_FILE_NAME)

    # make probability predictions with the model
    h_list = model.predict([np.array(sentences_encodings["input_ids"]), np.array(sentences_encodings["token_type_ids"]),
                            np.array(sentences_encodings["attention_mask"])])

    h_list = [np.reshape(h_vect, (len(h_vect),)) for h_vect in h_list]

    emb_model = word_emb_model.w2v_model
    top_n_words = 1000
    result = [word_embedding_model.retain_hashtags(emb_model.wv.similar_by_vector(h_vect, topn=top_n_words))[:k] for h_vect
              in h_list]

    return result
