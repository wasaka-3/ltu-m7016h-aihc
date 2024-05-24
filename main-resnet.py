import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from m7016h import default_parameters as dp
from m7016h.dataset_generator import initialize_generators
from m7016h.download_dataset import prepare_dataset
from m7016h.model_resnet50 import create_model_phase_1, create_model_phase_2
from m7016h.prepare_training import prepare
from m7016h.tarining_resnet50 import start_training_resnet50_extract_features, start_training_resnet50_phase_2

dataset_url = ('http://t12s-418866851410-ltu-adl.s3-website.eu-central-1.amazonaws.com'
               '/ltu-M7016H-dataset-2-20240523T104021Z-001.zip')

if __name__ == "__main__":
    prepare_dataset(source_url=dataset_url)

    train_datagen, train_generator, val_generator, test_generator, (
        target_image_width, target_image_height) = initialize_generators()

    callbacks, run_fit_log_dir = prepare(
        fit_logs_dir=os.path.join(dp.fit_logs_dir, 'resnet50'),
        checkpoint_dir=os.path.join(dp.checkpoint_dir, 'resnet50'),
        force_clean_fit=False,
        force_clean_checkpoints=False
    )
    train = True

    # Check if validation generator has data
    if val_generator.samples == 0:
        raise ValueError("Validation data not found. Please check the dataset path and structure.")

    base_model, model = create_model_phase_1(target_image_width=target_image_width,
                                             target_image_height=target_image_height)
    if train:
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
        all_callbacks = [early_stopping, model_checkpoint] + callbacks

        history, model = start_training_resnet50_extract_features(
            model=model,
            train_generator=train_generator,
            val_generator=val_generator,
            callbacks=all_callbacks.copy(),
            epochs=dp.epochs
        )

        model = create_model_phase_2(base_model=base_model, model=model)

        history, model = start_training_resnet50_phase_2(model,
                                                         train_generator,
                                                         val_generator,
                                                         callbacks=all_callbacks.copy(),
                                                         epochs=dp.epochs)

    model.load_weights('best_model.keras')

    val_loss, val_accuracy, val_auc, val_binary_accuracy, val_false_negatives, val_precision = model.evaluate(
        val_generator)
    print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
    print(f'Validation Precision: {val_precision * 100:.2f}%')
    print(f'Validation Loss: {val_loss * 100:.2f}%')

    test_loss, test_accuracy, test_auc, test_binary_accuracy, test_false_negatives, test_precision = model.evaluate(
        test_generator)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    print(f'Test Precision: {test_precision * 100:.2f}%')
    print(f'Test Loss: {test_loss * 100:.2f}%')
