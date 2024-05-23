from m7016h.after_training import get_best_model_weights, load_best_model_weights
from m7016h.dataset_generator import initialize_generators
from m7016h.download_dataset import prepare_dataset
from m7016h.evaluate_model import evalutate
from m7016h.model import create_model
from m7016h.prepare_training import prepare
from m7016h.tarining import start_training
from m7016h import default_parameters as dp

dataset_url = ('http://t12s-418866851410-ltu-adl.s3-website.eu-central-1.amazonaws.com'
               '/ltu-M7016H-dataset-2-20240523T104021Z-001.zip')

if __name__ == "__main__":
    prepare_dataset(source_url=dataset_url)

    train_datagen, train_generator, val_generator, test_generator, (
        target_image_width, target_image_height) = initialize_generators()

    model = create_model(target_image_width, target_image_height)

    tensorboard_callback, checkpoint_callback, run_fit_log_dir = prepare(
        force_clean_fit=False,
        force_clean_checkpoints=False
    )

    train = False
    if train:
        start_training(
            model=model,
            train_generator=train_generator,
            val_generator=val_generator,
            callbacks=[tensorboard_callback, checkpoint_callback],
            epochs=dp.epochs
            # epochs=500
        )

    best_model_weights = get_best_model_weights(checkpoint_dir=dp.checkpoint_dir)
    model = load_best_model_weights(model=model, best_model_weight_path=best_model_weights)
    evalutate(model=model, test_generator=test_generator, tensorboard_callback=tensorboard_callback)
