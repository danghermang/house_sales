import os
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from keras import backend as K
from keras import models, layers

K.clear_session()
EPOCHS = 1000
PROJECT_PATH = '.'

# In lipsa unui model antrenat se va realiza unul nou
MODEL_TRAINED = True


# Functia rmse folosita ca si loss function
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def read_dataframes(directory):
    train_data = pd.read_csv(os.path.join(directory, 'train.csv'))
    test_data = pd.read_csv(os.path.join(directory, 'test.csv'))
    return train_data, test_data


# Transforma datele din dataframe astfel incat sa poata fi incadrate
# in intervale mai usor de folosit pentru regresie
def generate_values(train_data, test_data):
    # Coloane ce contin valori continue, care pot fi scalate pe un interval [0,1]
    continous_columns = ['bedrooms', 'bathrooms', 'sqft_living',
                         'sqft_lot', 'floors', 'sqft_above', 'sqft_basement',
                         'yr_built', 'yr_renovated', 'lat',
                         'long', 'sqft_living15', 'sqft_lot15']
    cont_scaler = MinMaxScaler()
    cont_scaler.fit(train_data[continous_columns])
    cont_scaler.fit(test_data[continous_columns])

    train_continuous = cont_scaler.transform(train_data[continous_columns])
    test_continuous = cont_scaler.transform(test_data[continous_columns])

    # Coloanele ce contin valori care sunt grade-uri sau label-uri
    # care pot fi transformate intr-un vector binary one-hot
    binary_columns = ['waterfront', 'view', 'condition', 'grade', 'zipcode']
    train_values = train_continuous
    test_values = test_continuous
    for column in binary_columns:
        label_binarizer = LabelBinarizer()
        label_binarizer.fit(train_data[column])
        label_binarizer.fit(test_data[column])

        train_binary = label_binarizer.transform(train_data[column])
        test_binary = label_binarizer.transform(test_data[column])

        train_values = np.hstack([train_values, train_binary])
        test_values = np.hstack([test_values, test_binary])

    return train_values, test_values


# Folosesc un model de retea neuronala feedforward cu mai multe straturi
# pentru a favoriza abstractia datelor in defavorizarea memorarii lor. De asemenea
# folosesc initializarea lecun a weight-urilor deoarece am observat ca aduce o convergenta mai rapida
# Dropout-ul ajuta pentru a defavoriza creearea unui numar mic
# de legaturi neuronale care sa defineasca reteaua, practic impartind gradul de invatare.

def get_model(input_size):
    model = models.Sequential()

    model.add(layers.Dense(1024, activation='relu', input_shape=(input_size,), kernel_initializer="lecun_normal"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(256, activation='relu', kernel_initializer="lecun_normal"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64, activation='relu', kernel_initializer="lecun_normal"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(16, activation='relu', kernel_initializer="lecun_normal"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(8, activation='relu', kernel_initializer="lecun_normal"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(1, activation='linear', kernel_initializer="lecun_normal"))
    optimizer = Adam(0.001, decay=0.001 / EPOCHS)
    model.compile(optimizer=optimizer, loss=root_mean_squared_error,
                  metrics=[root_mean_squared_error, 'mae'])
    return model


if __name__ == "__main__":
    train_dataframe, test_dataframe = read_dataframes(PROJECT_PATH)
    train_values, test_values = generate_values(train_dataframe, test_dataframe)
    max_price = train_dataframe['price'].max()
    if MODEL_TRAINED and os.path.exists(os.path.join(PROJECT_PATH, 'model.h5')):
        nn_model = models.load_model("model.h5",
                                     custom_objects={'root_mean_squared_error': root_mean_squared_error})

    else:
        # Scalez coloana de preturi in functie de pretul maxim
        train_labels = train_dataframe['price'] / max_price
        # model_checkpoint = ModelCheckpoint(filepath=os.path.join(PROJECT_PATH, 'best_model.h5'),
        #                                    monitor='val_loss', verbose=True, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=200,
                                       restore_best_weights=True, verbose=True)

        callbacks_array = [early_stopping]
        nn_model = get_model(train_values.shape[1])
        nn_model.fit(train_values, train_labels, batch_size=32,
                     epochs=EPOCHS, validation_split=0.2,
                     callbacks=callbacks_array)
        nn_model.save('model.h5')
    preds = nn_model.predict(test_values) * max_price
    preds = preds.reshape(-1).astype(int)
    result_df = pd.DataFrame(data=(zip(test_dataframe.id, preds)),
                             columns=['id', 'price'])
    result_df.to_csv('sample_result.csv', sep=',', encoding='utf-8', index=False)
