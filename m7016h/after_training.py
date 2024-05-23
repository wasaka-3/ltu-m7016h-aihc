import os


def get_best_model_weights(checkpoint_dir=None):
    # Save the final model
    model_path = None
    for files in os.walk(checkpoint_dir):
        files[2].sort()
        model_path = files[2][-1]
        break

    best_model_weights_abs_path = os.path.join(checkpoint_dir, model_path)
    print(f"best_model_weights_abs_path='{best_model_weights_abs_path}'")

    return best_model_weights_abs_path


def load_best_model_weights(model, best_model_weight_path):
    model.load_weights(best_model_weight_path)
    return model
