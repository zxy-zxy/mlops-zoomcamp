* run mlflow ui with sqlite backend 
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

* run mlflow server with sqlite backend 
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts_local
```

