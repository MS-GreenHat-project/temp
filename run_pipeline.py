from azureml.core import Workspace, Environment, Dataset
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep

ws = Workspace.from_config()
dataset = Dataset.get_by_name(ws, 'labeling-data', version='latest')
env = Environment.from_conda_specification(name='ml-pipeline-env', file_path='environment.yml')

train_step = PythonScriptStep(
    name="Train Model",
    script_name="train.py",
    arguments=['--data-folder', dataset.as_mount()],
    compute_target='greenhat-ai-cluster',
    source_directory='.',
    runconfig=env
)

pipeline = Pipeline(workspace=ws, steps=[train_step])
pipeline_run = pipeline.submit('labeling-pipeline')
pipeline_run.wait_for_completion(show_output=True) 