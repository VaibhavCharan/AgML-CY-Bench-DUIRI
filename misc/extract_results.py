from cybench.runs.run_benchmark import compute_metrics

run_name = "maize_NL" 
df_metrics = compute_metrics(run_name)

print(
    df_metrics.groupby("model").agg(
        {"normalized_rmse": "mean", "mape": "mean", "r2": "mean"}
    )
)
