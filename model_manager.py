import joblib
import json
import os
import datetime


def save_surrogate_pack(models_dict, scalers_dict, output_folder="surrogate_models"):
    """
    Saves models and scalers into a structured folder with a registry file.
    """
    # 1. Create the Directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    registry = {"created_at": str(datetime.datetime.now()), "metrics": {}}

    print(f"--- Saving {len(models_dict)} Models to '{output_folder}/' ---")

    for metric_name in models_dict.keys():
        # Get the objects
        model = models_dict[metric_name]
        scaler = scalers_dict[metric_name]

        # Define Filenames
        model_filename = f"model_{metric_name}.joblib"
        scaler_filename = f"scaler_{metric_name}.joblib"

        # Save Binary Files (using compression to save disk space)
        joblib.dump(model, os.path.join(output_folder, model_filename), compress=3)
        joblib.dump(scaler, os.path.join(output_folder, scaler_filename), compress=3)

        # Update Registry (This is your map for later)
        registry["metrics"][metric_name] = {
            "model_path": model_filename,
            "scaler_path": scaler_filename,
            "input_dims": model.n_features_in_,  # Good safety check
        }
        print(f"✔ Saved: {metric_name}")

    # 2. Save the Registry JSON
    # This allows the loader to know exactly what to look for,
    # rather than guessing filenames.
    registry_path = os.path.join(output_folder, "registry.json")
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)

    print("--- Save Complete. Registry generated. ---")


def load_surrogate_pack(folder_path):
    # Read Registry
    with open(os.path.join(folder_path, "registry.json"), "r") as f:
        registry = json.load(f)

    loaded_models = {}
    loaded_scalers = {}

    for metric, paths in registry["metrics"].items():
        # Load Model & Scaler
        m_path = os.path.join(folder_path, paths["model_path"])
        s_path = os.path.join(folder_path, paths["scaler_path"])

        loaded_models[metric] = joblib.load(m_path)
        loaded_scalers[metric] = joblib.load(s_path)

    return loaded_models, loaded_scalers
