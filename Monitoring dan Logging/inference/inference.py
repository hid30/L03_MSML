import requests
import json
import pandas as pd

# Buat data dengan kolom string agar cocok dengan model
data_df = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=["0", "1", "2", "3"])

# Ubah jadi format JSON sesuai kebutuhan MLflow
data = {"inputs": data_df.to_dict(orient="records")}

# Kirim ke endpoint serving
response = requests.post(
    "http://127.0.0.1:5002/invocations",  # Pastikan port sesuai
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)

# Cetak hasil prediksi
print(response.json())
