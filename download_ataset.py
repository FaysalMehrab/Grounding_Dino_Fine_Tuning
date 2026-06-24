from roboflow import Roboflow
rf = Roboflow(api_key="iPszZavvZwMi4DoF4mOu")
project = rf.workspace("uarts-workspace").project("sm_suas")
version = project.version(1)

# Change "yolo26" to "coco"
dataset = version.download("coco")