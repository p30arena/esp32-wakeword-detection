from yamnet_commons import plt, tf
from yamnet_commons import model_path, model, callback, train_ds, val_ds, test_ds

if model_path.exists():
    model = tf.keras.models.load_model(model_path)

history = model.fit(train_ds,
                    epochs=20,
                    validation_data=val_ds,
                    callbacks=callback)

model.save(model_path)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()
