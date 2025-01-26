import tensorflow as tf 

### load model
model = tf.keras.models.load_model('artifacts/Model.keras')

### convert model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('converted_model.tflite', 'wb') as f:
    f.write(tflite_model)