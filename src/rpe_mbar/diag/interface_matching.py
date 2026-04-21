import numpy as np
from ..data.loaders import weights_split_into_sets

def recrossings_TIS_interface(cv_function, interface_value, data_max_values, dataset_forward_descriptors, weights):
    # Collect weighted descriptors

    weights_each_set_iface = weights_split_into_sets(weights, data_max_values)

    recrossings = 0
    weighted_recrossings = 0
    for origin_set in  data_max_values.keys():
        print(f"Analyzing recrossings for origin set: {origin_set}")
        for iface in  data_max_values[origin_set].keys():
            path_indices = data_max_values[origin_set][iface].get("path_indices", np.arange(data_max_values[origin_set][iface][origin_set].shape[0]))
            for traj_idx in range(len(path_indices)):
                trajectory = dataset_forward_descriptors.get_trajectory_descriptors(origin_set, iface, path_indices[traj_idx])
                cv_values = cv_function(trajectory)
                for i in range(1, len(cv_values)):
                    if cv_values[i] > interface_value and cv_values[i-1] < interface_value:
                        recrossings +=1
                        weighted_recrossings += weights_each_set_iface[origin_set][iface][traj_idx]
    return recrossings, weighted_recrossings

def compute_cv_all_sets(cv_function_list, trajectory, set_names=None):
    cv_values_all_sets = np.zeros((len(cv_function_list), trajectory.shape[0]))
    for i, cv_function in enumerate(cv_function_list):
        cv_values = cv_function(trajectory)
        cv_values_all_sets[i, :] = cv_values
    return cv_values_all_sets


def index_beyond_interface(cv_values, interface_value, direction="forward"):
    #remove all tis path segments before the first interface

    return cv_values>interface_value if direction=="forward" else cv_values<interface_value



def index_beyond_joined_interface(cv_values_all_sets, interface_values, state_function_values=None, state_boundary=None, direction="forward"):
    index_beyond = np.zeros((cv_values_all_sets.shape[1]))
    for set in range(cv_values_all_sets.shape[0]):
        index_beyond_set = index_beyond_interface(
            cv_values_all_sets[set],
            interface_values[set],
            direction=direction
        )
        index_beyond = (index_beyond + index_beyond_set).astype(bool)
    if state_function_values is not None and state_boundary is not None:
        index_beyond = index_beyond & (state_function_values > state_boundary if direction=="forward" else state_function_values < state_boundary)
    return np.where(index_beyond)



