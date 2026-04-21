import numpy as np
from ..utils.tools import robust_flatten, find_last_crossed_interfaces


def solve_multi_set_mbar_matrix(
    max_lambda_values_each_set,
    N_lambda_each_set,
    interface_values_lambda_each_set,
    max_iterations=10000,
    tolerance=1e-10,
    log: bool = False
):
    """
    Perform the multi-set MBAR calculation for multiple sets of maximum lambda values for each interface.
    Parameters:
        max_lambda_values_each_set (list of list of np.ndarray): List of maximum lambda values for each set (a list of lists).
        N_lambda_each_set (list of list of int): List of number of samples for each set and interface.
        interface_values_lambda_each_set (list of np.ndarray): List of interface values for lambda.
        max_iterations (int): Maximum number of iterations for convergence.
        tolerance (float): Convergence tolerance.
    """

    n_sets = len(max_lambda_values_each_set)
    

    max_lambda_set = []
    num_lambda_interfaces_set= []
    k_max_set = []
    Z_lambda_set = []
    lambda_contributions_set = []
    for set in range(n_sets):
        max_lambda_set.append(robust_flatten(max_lambda_values_each_set[set]))
        num_lambda_interfaces_set.append(len(interface_values_lambda_each_set[set]))
        k_max_set.append(find_last_crossed_interfaces(max_lambda_set[set], interface_values_lambda_each_set[set]))
        Z_lambda_set.append(np.ones(num_lambda_interfaces_set[set]))
        lambda_contributions_set.append((k_max_set[set][:, None] >= np.arange(num_lambda_interfaces_set[set])).astype(float))

    for iteration in range(max_iterations):
        denom_lambda_set = []
        for set in range(n_sets):
            denom_lambda_set.append((lambda_contributions_set[set] * (N_lambda_each_set[set] / Z_lambda_set[set])).sum(axis=1))
        denom = np.sum(denom_lambda_set, axis=0)
        weights = 1.0 / denom
        new_Z_lambda_set = []
        for set in range(n_sets):
            new_Z_lambda_set.append((lambda_contributions_set[set].T @ weights))
        
        #normalize each set's Z_lambda
        total_Z = new_Z_lambda_set[0][0]
        for set in range(n_sets):
            new_Z_lambda_set[set] /= total_Z

        diff = np.linalg.norm(np.concatenate([np.log(new_Z_lambda_set[set]) for set in range(n_sets)]) -
                              np.concatenate([np.log(Z_lambda_set[set]) for set in range(n_sets)]))
        if diff < tolerance:
            if log:
                print(f"Converged in {iteration} iterations.")
            break

        Z_lambda_set = new_Z_lambda_set
    if log:
        if diff >= tolerance:
            print(f"Did not converge within {max_iterations} iterations, final diff: {diff}")

    return Z_lambda_set, weights

def solve_two_set_mbar_matrix(
    max_lambda_values, max_mu_values,
    N_lambda, N_mu,
    interface_values_lambda, interface_values_mu,
    max_iterations=10000,
    tolerance=1e-10
):
    max_lambda_values = robust_flatten(max_lambda_values)
    max_mu_values = robust_flatten(max_mu_values)

    num_lambda_interfaces = len(interface_values_lambda)
    num_mu_interfaces = len(interface_values_mu)

    k_max_lambda = find_last_crossed_interfaces(max_lambda_values, interface_values_lambda)
    k_max_mu = find_last_crossed_interfaces(max_mu_values, interface_values_mu)

    Z_lambda = np.ones(num_lambda_interfaces) / num_lambda_interfaces
    Z_mu = np.ones(num_mu_interfaces) / num_mu_interfaces

    lambda_contributions = (k_max_lambda[:, None] >= np.arange(num_lambda_interfaces)).astype(float)
    mu_contributions = (k_max_mu[:, None] >= np.arange(num_mu_interfaces)).astype(float)

    for iteration in range(max_iterations):
        denom_lambda = (lambda_contributions * (N_lambda / Z_lambda)).sum(axis=1)
        denom_mu = (mu_contributions * (N_mu / Z_mu)).sum(axis=1)
        denom = denom_lambda + denom_mu

        weights = 1.0 / denom

        new_Z_lambda = (lambda_contributions.T @ weights)
        new_Z_mu = (mu_contributions.T @ weights)

        total_Z = np.sum(new_Z_lambda) + np.sum(new_Z_mu)
        new_Z_lambda /= total_Z
        new_Z_mu /= total_Z

        diff = np.linalg.norm(np.concatenate([new_Z_lambda, new_Z_mu]) -
                              np.concatenate([Z_lambda, Z_mu]))
        if diff < tolerance:
            print(f"Converged in {iteration} iterations.")
            break

        Z_lambda = new_Z_lambda
        Z_mu = new_Z_mu

    return Z_lambda, Z_mu, weights
