import cvxpy as cp
import numpy as np

# Define problem size
n = 3  # Number of variables

# Define optimization variable
x = cp.Variable(n)

# Define quadratic cost matrix (must be positive semi-definite)
P = np.array([[4, 1, 0], 
              [1, 2, 0], 
              [0, 0, 3]])

# Define linear cost vector
q = np.array([1, -2, 3])

# Define inequality constraint: Ax ≤ b
A = np.array([[1, 2, 3], 
              [-1, 0, 1]])
b = np.array([4, 1])

# Define QP objective function (0.5 * x^T P x + q^T x)
objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)

# Define constraints
constraints = [A @ x <= b]

# Define and solve the problem
prob = cp.Problem(objective, constraints)
result = prob.solve()

# Print results
print("Optimal value:", result)
print("Optimal x:", x.value)
