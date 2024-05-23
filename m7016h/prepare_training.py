import datetime
import os
import shutil
from m7016h import default_parameters as dp

if dp.is_mac:
    from keras.api.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
else:
    from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


def prepare(
    fit_logs_dir=dp.fit_logs_dir,
    checkpoint_dir=dp.checkpoint_dir,
    force_clean_fit=False,
    force_clean_checkpoints=False,
    run_id=dp.run_id
):
    if force_clean_fit and os.path.exists(fit_logs_dir):
        shutil.rmtree(fit_logs_dir)

    run_id = run_id if run_id is not None else f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    run_fit_log_dir = os.path.join(fit_logs_dir, run_id)

    if not os.path.exists(run_fit_log_dir):
        os.makedirs(run_fit_log_dir)

    # TensorBoard callback
    tensorboard_callback = TensorBoard(log_dir=run_fit_log_dir, histogram_freq=dp.histogram_freq)

    if force_clean_checkpoints and os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Model checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, f'model-{run_id}-' + '{epoch:04d}.keras'),
        monitor='val_loss',
        save_best_only=True
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    return [tensorboard_callback, checkpoint_callback, early_stopping], run_fit_log_dir
