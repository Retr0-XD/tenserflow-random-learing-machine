checkpoint_dir = ‘./checkpoints’+ datetime.datetime.now().strftime(“_%Y.%m.%d-%H:%M:%S”)
checkpoint_prefix = os.path.join(checkpoint_dir, “ckpt_{epoch}”)
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
history = model.fit(tr_dataset, epochs=epochs, callbacks=[checkpoint_callback, early_stop] , validation_data=val_dataset)
print (“Training stopped as there was no improvement after {} epochs”.format(patience))



optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=loss)
patience = 10
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
