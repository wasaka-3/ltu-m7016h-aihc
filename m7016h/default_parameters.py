import platform
import datetime
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
home_folder = os.path.join(os.path.expanduser('~'), 'm7016h')
dataset_folder = 'ltu-M7016H-dataset-2'
train_dir = os.path.join(dataset_folder, 'train')
val_dir = os.path.join(dataset_folder, 'val')
test_dir = os.path.join(dataset_folder, 'test')
logs_dir = os.path.join(home_folder, 'logs')
fit_logs_dir = os.path.join(logs_dir, 'fit')
checkpoint_dir = os.path.join(home_folder, 'checkpoints')
run_id=f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
image_size_divider = 3
batch_size = 16
histogram_freq=1
epochs=256
is_mac = platform.system() == 'Darwin'


def absolute_dataset_path(path):
    return os.path.join(home_folder, path)
