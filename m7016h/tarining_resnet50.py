from m7016h import default_parameters as dp


def start_training_resnet50_extract_features(model,
                                             train_generator,
                                             val_generator,
                                             callbacks=None,
                                             epochs=dp.epochs):
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
        validation_data=val_generator,
        validation_steps=max(1, val_generator.samples // val_generator.batch_size),
        epochs=epochs,
        callbacks=callbacks if callbacks is not None else []
    )

    return history, model


def start_training_resnet50_phase_2(model,
                                    train_generator,
                                    val_generator,
                                    callbacks=None,
                                    epochs=dp.epochs):
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=val_generator.samples // val_generator.batch_size,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )

    model.load_weights('best_model.keras')

    return history, model
