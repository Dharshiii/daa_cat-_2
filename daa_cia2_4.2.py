import numpy as np
import matplotlib.pyplot as plt

# Define the function to be optimized
def f(x):
    return x ** 2

# Define the PSO algorithm
def PSO(f, n_particles=50, n_iterations=100, c1=2, c2=2, w=0.7):
    # Initialize the particles
    x_min = -10
    x_max = 10
    particles = np.random.uniform(x_min, x_max, size=(n_particles, 1))
    velocities = np.zeros((n_particles, 1))
    best_positions = particles.copy()
    best_scores = f(best_positions)

    # Initialize the global best position and score
    global_best_position = best_positions[np.argmin(best_scores)]
    global_best_score = np.min(best_scores)

    # Run the PSO algorithm
    for i in range(n_iterations):
        # Update the velocities and positions
        r1 = np.random.uniform(size=(n_particles, 1))
        r2 = np.random.uniform(size=(n_particles, 1))
        velocities = w * velocities \
                    + c1 * r1 * (best_positions - particles) \
                    + c2 * r2 * (global_best_position - particles)
        particles += velocities

        # Check if any particles have gone out of bounds
        particles = np.clip(particles, x_min, x_max)

        # Update the best positions and scores
        scores = f(particles)
        improved_indices = scores < best_scores
        best_positions[improved_indices] = particles[improved_indices]
        best_scores[improved_indices] = scores[improved_indices]

        # Update the global best position and score
        if np.min(best_scores) < global_best_score:
            global_best_position = best_positions[np.argmin(best_scores)]
            global_best_score = np.min(best_scores)
        # Plot the particles
        plt.clf()
        plt.plot(particles, f(particles), 'bo', label='particles')
        plt.plot(best_positions, f(best_positions), 'ro', label='best positions')
        plt.plot(global_best_position, global_best_score, 'go', label='global best')
        plt.legend()
        plt.xlim(x_min, x_max)
        plt.ylim(0, 100)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Iteration {i+1}/{n_iterations}')
        plt.pause(0.001)

    # Return the global best position and score
    return global_best_position, global_best_score

# Test the PSO algorithm on f(x) = x^2
best_position, best_score = PSO(f)
print(f'The minimum of f(x) = x^2 is at x = {best_position[0]}, with a value of {best_score}.')
