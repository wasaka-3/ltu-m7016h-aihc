import numpy as np

def evalutate(model, test_generator, tensorboard_callback):
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(
        test_generator,
        steps=max(1, test_generator.samples // test_generator.batch_size),
        callbacks=[tensorboard_callback]
    )
    print(f'Test Accuracy: {test_acc}')

    # Predict the labels for test set
    Y_pred = model.predict(test_generator, steps=(test_generator.samples // test_generator.batch_size) + 1)
    y_pred = np.round(Y_pred).astype(int)

    # Ensuring y_pred matches the number of test samples
    y_pred = y_pred[:test_generator.samples]

    y_true = test_generator.classes

    # Calculate F-score
    from sklearn.metrics import f1_score, precision_score, recall_score

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f'F1 Score: {f1}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')