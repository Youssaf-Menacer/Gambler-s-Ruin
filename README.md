# Gambler-s-Ruin
Gambler's Problem
import numpy as np
import sys
import matplotlib.pyplot as plt

if "../" not in sys.path:
    sys.path.append("../")


    def value_iteration_for_gamblers(p_h, N, theta=0.0001, gamma=1.0):
        """
        Args:
            p_h: Probability of the coin coming up heads
        """
        # The reward is zero on all transitions except those on which the gambler reaches his goal,
        # when it is +1.
        rewards = np.zeros(N+1)
        rewards[N] = 1  # reaching his goal for s=N

        # We introduce two dummy states corresponding to termination with capital of 0 and N
        V = np.zeros(N+1)

        # 1. Policy evaluation:

        def policy_evaluation(s, V, rewards):

            A = np.zeros(N+1)
            stakes = range(1, min(s, N - s) + 1)  # Your minimum bet is 1, maximum bet is min(s, N-s).
            for a in stakes:
                # rewards[s+a], rewards[s-a] are immediate rewards.
                # V[s+a], V[s-a] are values of the next states.
                # This is the core of the Bellman equation: The expected value of your action is
                # the sum of immediate rewards and the value of the next state.
                A[a] = p_h * (rewards[s + a] + V[s + a] * gamma) + (1 - p_h) * (rewards[s - a] + V[s - a] * gamma)
            return A

        while True:
            delta = 0
            for s in range(1, N):
                # Use policy_evaluation function to find the best action
                A = policy_evaluation(s, V, rewards)
                # print(s,A,V) # if you want to debug.
                best_action_value = np.max(A)
                # Calculate delta across all states seen so far
                delta = max(delta, np.abs(best_action_value - V[s]))
                # Update the value function (Sutton book eq. 4.10).
                V[s] = best_action_value
                # Check if we can stop
            if delta < theta:
                break

        # 2. Policy Imporovement

        policy = np.zeros(N)
        for s in range(1, N):
            # policy_evaluation to find the best action for this state
            A = policy_evaluation(s, V, rewards)
            best_action = np.argmax(A)
            # Always take the best action
            policy[s] = best_action

        return policy, V

policy, v = value_iteration_for_gamblers(0.4,100) # you can choose N as the 
                                                  # second argument of the 
                                                   # function
print("Optimized Policy:")
print(policy)
print("")

print("Optimized Value Function:")
print(v)
print("")


# Plotting Value Funtion & Final Policy

# x axis values

x = range(100) # because N=100
# corresponding y axis values
y = policy
# plotting the bars
plt.bar(x, y, align='center', alpha=0.5)
# the x axis
plt.xlabel('Capital')
# the y axis
plt.ylabel('Final policy (stake)')
 
# giving a title to the graph
plt.title('Capital vs Final Policy')
 
# function to show the plot
plt.show()
