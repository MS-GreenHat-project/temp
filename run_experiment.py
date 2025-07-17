from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, Dataset

ws = Workspace.from_config()
dataset = Dataset.get_by_name(ws, 'labeling-data', version='latest')

env = Environment.from_conda_specification(name='ml-pipeline-env', file_path='environment.yml')

src = ScriptRunConfig(
    source_directory='.',
    script='train.py',
    arguments=['--data-folder', dataset.as_mount()],
    environment=env
)

experiment = Experiment(workspace=ws, name='labeling-experiment')
run = experiment.submit(src)
run.wait_for_completion(show_output=True) 