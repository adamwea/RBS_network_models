# Define the quadratic scoring function with an adaptive flipped sigmoid penalty
import numpy as np
import matplotlib.pyplot as plt

def the_scoring_function_quadratic_smooth_sigmoid(val, target_val, maxFitness, weight, min_val=None, max_val=None):
    """
    Quadratic scoring function with a smoothly adjusted sigmoid penalty:
    - If both bounds exist, use a quadratic function with sharper slopes based on weight.
    - If one bound is missing, apply a flipped sigmoid penalty with an iteratively adjusted shift
      so that it smoothly approaches 1 near the target without engaging piecewise rules.
    - The weight parameter controls sharpness:
      - Higher weight makes the quadratic function steeper and the sigmoid approach 1 or 1000 faster.
    """
    if val is None:
        return maxFitness

    # Case 1: Both bounds are present (standard quadratic)
    if min_val is not None and max_val is not None:
        a_left = (maxFitness - 1) / ((target_val - min_val) ** 2 / weight) if target_val != min_val else float('inf')
        a_right = (maxFitness - 1) / ((max_val - target_val) ** 2 / weight) if target_val != max_val else float('inf')

        if val <= target_val:
            return min(a_left * (val - target_val) ** 2 + 1, maxFitness)
        else:
            return min(a_right * (val - target_val) ** 2 + 1, maxFitness)

    # Adaptive shifted sigmoid function
    def shifted_sigmoid(x, direction):
        """
        Flipped sigmoid function with adaptive shifting:
        - Ensures that values near the target smoothly approach 1.
        - Instead of scaling, we shift the sigmoid away from the target.
        - The weight parameter adjusts the steepness of the sigmoid.
        """
        shift = direction * abs(target_val)  # Shift proportional to the target value magnitude
        scale = 0.1 / weight  # Higher weight makes the transition steeper
        return maxFitness - (maxFitness - 1) / (1 + np.exp(direction * scale * (x - (target_val + shift))))

    # Case 2: One bound is missing, apply the adaptive shifted sigmoid
    if min_val is None and max_val is not None:
        if max_val is not None and val >= target_val:
            return min((maxFitness - 1) / ((max_val - target_val) ** 2 / weight) * (val - target_val) ** 2 + 1, maxFitness)
        else:
            return shifted_sigmoid(val, -1)  # Shifted sigmoid for extreme low values

    elif max_val is None and min_val is not None:
        if min_val is not None and val <= target_val:
            return min((maxFitness - 1) / ((target_val - min_val) ** 2 / weight) * (val - target_val) ** 2 + 1, maxFitness)
        else:
            return shifted_sigmoid(val, 1)  # Shifted sigmoid for extreme high values

    # Case 3: No bounds at all (shifted sigmoid on both sides)
    elif min_val is None and max_val is None:
        if val < target_val:
            return shifted_sigmoid(val, -1)
        else:
            return shifted_sigmoid(val, 1)
        #return shifted_sigmoid(val, -1) if val < target_val else shifted_sigmoid(val, 1)

# Plotting function to visualize the scoring function
def plot_scoring_function(target_val, maxFitness, weight, min_val=None, max_val=None):
    x_values = np.linspace(target_val - 100, target_val + 100, 500)
    y_values = [the_scoring_function_quadratic_smooth_sigmoid(x, target_val, maxFitness, weight, min_val, max_val) for x in x_values]
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, label=f"Scoring Function (Weight={weight})", color='blue')
    plt.axvline(target_val, color='green', linestyle="--", label="Target Value")
    if min_val is not None:
        plt.axvline(min_val, color='red', linestyle="--", label="Min Bound")
    if max_val is not None:
        plt.axvline(max_val, color='red', linestyle="--", label="Max Bound")
    plt.xlabel("Input Value")
    plt.ylabel("Fitness Score")
    plt.title("Quadratic + Shifted Flipped Sigmoid Scoring Function")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(f"scoring_function_weight_{weight}_min_{min_val}_max_{max_val}.png")

# Example usage
plot_scoring_function(target_val=50, maxFitness=1000, weight=1, min_val=20, max_val=80)
plot_scoring_function(target_val=50, maxFitness=1000, weight=1, min_val=20, max_val=None)
plot_scoring_function(target_val=50, maxFitness=1000, weight=1, min_val=None, max_val=80)
plot_scoring_function(target_val=50, maxFitness=1000, weight=5, min_val=20, max_val=80)
plot_scoring_function(target_val=50, maxFitness=1000, weight=1, min_val=None, max_val=None)
plot_scoring_function(target_val=50, maxFitness=1000, weight=5, min_val=None, max_val=None)
