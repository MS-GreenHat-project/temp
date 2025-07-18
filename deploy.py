from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model
from azure.identity import DefaultAzureCredential

ml_client = MLClient(DefaultAzureCredential(), "ca3a7121-8c69-4769-a3c4-01dbafb3872d", "Green-Hat", "greenhat-ai")

# 모델 등록
model = ml_client.models.create_or_update(
    Model(
        path="outputs/model",  # 모델 파일 경로
        name="greenhat-ai-model",
        version="1"
    )
)

# 엔드포인트/배포 업데이트 (필요시)
endpoint_name = "greenhat-ai-endpoint"
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model.id,
    instance_type="Standard_DS3_v2",
    instance_count=1
)
ml_client.online_deployments.begin_create_or_update(deployment)