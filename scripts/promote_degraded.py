import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.loader import get_mlflow_client, load_production_model
from src.promoter import promote_candidate

client = get_mlflow_client("http://localhost:5000")

all_versions = client.search_model_versions("name='telco_churn'")
v9 = [v for v in all_versions if v.version == "9"][0]
_, current_prod = load_production_model(client, "telco_churn")

promote_candidate(client, "telco_churn", v9, current_prod)
print("v9 promoted to production — degraded model is now live")