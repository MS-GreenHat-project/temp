import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, Component
from azure.identity import DefaultAzureCredential

def register_evaluate_component():
    """평가 컴포넌트를 Azure ML에 등록"""
    
    # Azure ML 클라이언트 설정
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id="ca3a7121-8c69-4769-a3c4-01dbafb3872d",
        resource_group_name="Green-Hat",
        workspace_name="greenhat-ai"
    )
    
    # 환경 확인
    env_name = "yolo-env"
    try:
        environment = ml_client.environments.get(name=env_name, version="latest")
        print(f"[INFO] 환경 사용: {env_name}@{environment.version}")
    except Exception as e:
        print(f"[ERROR] 환경을 찾을 수 없습니다: {env_name}")
        print(f"에러: {e}")
        return
    
    # 평가 컴포넌트 정의
    evaluate_component = Component(
        name="evaluate",
        version="1.0.0",
        display_name="YOLO Model Evaluation",
        description="YOLO 모델 평가 및 비교 컴포넌트",
        type="command",
        code=".",
        command="python evaluate.py --new-model ${{inputs.new_model}} --data-folder ${{inputs.data_folder}} --output-dir ${{outputs.evaluation_results}}",
        environment=f"azureml:{env_name}@latest",
        inputs={
            "new_model": {
                "type": "uri_file",
                "description": "새로 훈련된 모델 파일 경로"
            },
            "data_folder": {
                "type": "uri_folder", 
                "description": "테스트 데이터 폴더"
            }
        },
        outputs={
            "evaluation_results": {
                "type": "uri_folder",
                "description": "평가 결과 및 비교 리포트"
            }
        }
    )
    
    try:
        # 컴포넌트 등록
        registered_component = ml_client.components.create_or_update(evaluate_component)
        print(f"[SUCCESS] 평가 컴포넌트 등록 완료: {registered_component.name}@{registered_component.version}")
        print(f"[INFO] 컴포넌트 ID: {registered_component.id}")
        
        return registered_component
        
    except Exception as e:
        print(f"[ERROR] 컴포넌트 등록 실패: {e}")
        return None

def test_evaluate_component():
    """평가 컴포넌트 테스트"""
    
    # Azure ML 클라이언트 설정
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id="ca3a7121-8c69-4769-a3c4-01dbafb3872d",
        resource_group_name="Green-Hat",
        workspace_name="greenhat-ai"
    )
    
    # 테스트 데이터 경로 (실제 경로로 수정 필요)
    test_data_path = "azureml://subscriptions/ca3a7121-8c69-4769-a3c4-01dbafb3872d/resourceGroups/Green-Hat/providers/Microsoft.MachineLearningServices/workspaces/greenhat-ai/data/labeling-data/versions/5"
    
    # 테스트 모델 경로 (실제 경로로 수정 필요)
    test_model_path = "azureml://subscriptions/ca3a7121-8c69-4769-a3c4-01dbafb3872d/resourceGroups/Green-Hat/providers/Microsoft.MachineLearningServices/workspaces/greenhat-ai/data/models/versions/1"
    
    try:
        # 컴포넌트 가져오기
        component = ml_client.components.get(name="evaluate", version="1.0.0")
        
        # 테스트 작업 생성
        from azure.ai.ml import dsl, Input, Output
        
        @dsl.pipeline(
            name="test-evaluate-pipeline",
            description="평가 컴포넌트 테스트 파이프라인"
        )
        def test_pipeline():
            evaluate_job = component(
                new_model=Input(type="uri_file", path=test_model_path),
                data_folder=Input(type="uri_folder", path=test_data_path)
            )
            return evaluate_job.outputs.evaluation_results
        
        # 파이프라인 제출
        pipeline_job = ml_client.jobs.create_or_update(
            test_pipeline(),
            experiment_name="test-evaluate"
        )
        
        print(f"[INFO] 테스트 파이프라인 제출됨: {pipeline_job.name}")
        print(f"[INFO] 작업 ID: {pipeline_job.id}")
        
        return pipeline_job
        
    except Exception as e:
        print(f"[ERROR] 테스트 실패: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("YOLO 평가 컴포넌트 등록")
    print("=" * 60)
    
    # 컴포넌트 등록
    component = register_evaluate_component()
    
    if component:
        print("\n" + "=" * 60)
        print("평가 컴포넌트 등록 완료!")
        print("=" * 60)
        
        # 테스트 실행 여부 확인
        test_choice = input("\n테스트를 실행하시겠습니까? (y/n): ").lower().strip()
        if test_choice == 'y':
            print("\n테스트 파이프라인 실행 중...")
            test_job = test_evaluate_component()
            if test_job:
                print(f"[SUCCESS] 테스트 파이프라인 실행됨: {test_job.id}")
            else:
                print("[ERROR] 테스트 파이프라인 실행 실패")
    else:
        print("[ERROR] 컴포넌트 등록 실패") 