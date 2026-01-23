from surrogate.model_manager import load_surrogate_pack
from surrogate.surrogate import evaluate_surrogate_physics
from data.train_data import load_data, clean_data

df = load_data("surrogate_train_data_20p.csv")
df_clean = clean_data(
    df,
    nan_filter=True,
    bounds_filter=True,
    mad_filter=True,
    mad_filter_threshold=3.5,
)
df_clean = clean_data(df_clean, "cd_min", mad_filter=True, mad_filter_threshold=2.5)

df_sample = df_clean.sample(n=1000)

models, scalers = load_surrogate_pack("./.surrogate_models/surrogate_models_v4")

for key in models:
    print()
    print(f"--- Model {key} ---")
    print()
    gp = models[key]
    scaler = scalers[key]

    evaluate_surrogate_physics(
        gp, scaler, df_sample, key, ["c_max", "log_Re"], plot=False
    )
    print()
