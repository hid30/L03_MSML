from prometheus_client import Counter, Gauge, start_http_server
import time

# Metrik
prediction_count = Counter('model_predictions_total', 'Total number of predictions')
accuracy_gauge = Gauge('model_accuracy', 'Model accuracy')
latency_gauge = Gauge('prediction_latency_seconds', 'Prediction latency')

# Jalankan HTTP server untuk Prometheus
start_http_server(8000)
print("Exporter running at http://localhost:8000/metrics")

try:
    while True:
        prediction_count.inc()
        accuracy_gauge.set(0.95)
        latency_gauge.set(0.1)
        time.sleep(1)
except KeyboardInterrupt:
    print("Exporter stopped.")
