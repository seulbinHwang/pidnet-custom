import numpy as np

# Given variables (example)
variable_1 = np.random.randint(0, 256, (10, 20, 3))  # Assuming random RGB values for variable_1
variable_2 = np.array([100, 150, 200])  # Example RGB values for variable_2

# Compare variable_1 with variable_2
matches = np.all(variable_1 == variable_2, axis=-1)
print("matches:, ", matches.shape)
# Count the number of matches
num_matches = np.sum(matches)

# Print the result
print(f"The number of occurrences of variable_2's RGB values in variable_1: {num_matches}")
