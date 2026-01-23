from train_data import (
    load_data,
    clean_data,
    histogram_train_data,
    audit_data_cleaning,
    OUTPUT_COLS,
)
from model_manager import save_surrogate_pack
from surrogate import train_surrogate, evaluate_surrogate_physics


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load
    df = load_data("surrogate_train_data_20p.csv")
    df_clean = clean_data(
        df,
        nan_filter=True,
        bounds_filter=True,
        mad_filter=True,
        mad_filter_threshold=3.5,
    )
    df_clean = clean_data(df_clean, "cd_min", mad_filter=True, mad_filter_threshold=2.5)

    # Optional: Audit Cleaning
    # audit_data_cleaning(df, df_clean)

    df_sample = df_clean  # .sample(n=1250, random_state=42)

    # histogram_train_data(df_clean)

    models = {}
    scalers = {}

    # 2. Train
    training_targets = OUTPUT_COLS
    for output_target in training_targets:
        print()
        print(f"--- Surrogate for {output_target} ---")
        gp_model, scaler, X_test, y_test, y_pred, y_std, r2, rmse = train_surrogate(
            df_sample,
            output_target=output_target,
            length_scale_min=0.025,
            matern_nu=2.5,
            white_kernel_noise=1e-3,
        )

        models[output_target] = gp_model
        scalers[output_target] = scaler

        # 3. Validate
        evaluate_surrogate_physics(
            gp_model, scaler, df_sample, output_target, ["log_Re", "c_max"]
        )

    save_surrogate_pack(models, scalers, output_folder="surrogate_models_v4")
