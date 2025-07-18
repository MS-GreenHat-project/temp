from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="ca3a7121-8c69-4769-a3c4-01dbafb3872d",
    resource_group_name="Green-Hat",
    workspace_name="greenhat-ai"
)

env = Environment(
    name="greenhat-ml-pipeline-env",
    description="greenhat ML 파이프라인 환경",
    image=None,
    build_context="ml-pipline",
    dockerfile="Dockerfile"
)
ml_client.environments.create_or_update(env)
print("환경등록완료")