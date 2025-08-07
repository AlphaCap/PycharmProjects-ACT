from performance_objectives import ObjectiveManager

# Initialize the ObjectiveManager
obj_mgr = ObjectiveManager()

# Retrieve the primary objective
primary_objective = obj_mgr.get_primary_objective()
print("Primary objective:", primary_objective)

# List available objectives
print("Available objectives:", obj_mgr.list_objectives())
