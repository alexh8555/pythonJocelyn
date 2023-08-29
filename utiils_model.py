from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, Bidirectional
# from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

def buildManyToOneModel(shape):
	model = Sequential()
	model.add(LSTM(256, input_shape=(shape[1], shape[2])))
	# model.add(Dropout(0.1))
	# model.add(LSTM(256))
	# model.add(Dropout(0.3))
	# output shape: (1, 1)
	model.add(Dense(1))
	model.compile(loss="mse", optimizer="adam")
	model.summary()
	return model

def start_training(model, X_train, Y_train, X_validate, Y_validate, modelName):
	# train model
    model = buildManyToOneModel(X_train.shape)
    callback = EarlyStopping(monitor="loss", patience=100, verbose=1, mode="auto")
    history = model.fit(X_train, Y_train, epochs=5000, batch_size=256, validation_data=(X_validate, Y_validate),
			            callbacks=[callback])

    model.summary()
    model.save(modelName)

    return model