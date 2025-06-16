from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

client = MlflowClient()
experiments = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)

print("Eksperimen yang ada:")
for exp in experiments:
    print(f"- {exp.name} (ID: {exp.experiment_id})")

# Cari experiment 'Iris_Classification'
iris_exp = [e for e in experiments if e.name == 'Iris_Classification']
if iris_exp:
    runs = client.search_runs(iris_exp[0].experiment_id)
    print(f"\nRun yang ditemukan: {len(runs)}")
    for run in runs:
        print(f"Run ID: {run.info.run_id}, Metrics: {run.data.metrics}")
else:
    print("Eksperimen 'Iris_Classification' tidak ditemukan.")
