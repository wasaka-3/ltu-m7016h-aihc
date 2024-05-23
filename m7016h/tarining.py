from m7016h import default_parameters as dp


def start_training(model, train_generator, val_generator, callbacks=None, epochs=dp.epochs):
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
        validation_data=val_generator,
        validation_steps=max(1, val_generator.samples // val_generator.batch_size),
        epochs=epochs,
        callbacks=callbacks if callbacks is not None else []
    )

    return history
