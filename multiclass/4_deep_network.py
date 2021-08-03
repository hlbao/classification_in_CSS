#run deep learning
#should be attached to 2_representations.py
#I separate them for clear illustration reasons.

X_train, X_val, y_train, y_val = train_test_split(train_df[features], train_df[label_col], test_size=0.2, random_state=2021)
X_train = tf_idf_vect.transform(X_train['comment_text'])
X_val = tf_idf_vect.transform(X_val['comment_text'])
X_test = tf_idf_vect.transform(X_test['comment_text'])
feature_names = tf_idf_vect.get_feature_names()

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=train_df[features].shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())
epochs = 5
batch_size = 64
lstm_model = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
accr = model.evaluate(X_test,y_test)
