import random

student_name = "Xinyuan Quan"


# 1. Q-Learning
class QLearningAgent:
    """Implement Q Reinforcement Learning Agent using Q-table."""

    def __init__(self, game, discount, learning_rate, explore_prob):
        """Store any needed parameters into the agent object.
        Initialize Q-table.
        """
        self.game = game
        self.discount = discount
        self.learning_rate = learning_rate
        self.explore_prob = explore_prob
        self.q_values = {}

    def get_q_value(self, state, action):
        """Retrieve Q-value from Q-table.
        For an never seen (s,a) pair, the Q-value is by default 0.
        """
        return self.q_values.get(state, {}).get(action, 0.0)

    def get_value(self, state):
        """Compute state value from Q-values using Bellman Equation.
        V(s) = max_a Q(s,a)
        """
        actions = self.game.get_actions(state)
        if not actions:
            return 0.0
        return max(self.get_q_value(state, action) for action in actions)

    def get_best_policy(self, state):
        """Compute the best action to take in the state using
        Policy Extraction.
        π(s) = argmax_a Q(s,a)

        If there are ties, return a random one for better performance.
        Hint: use random.choice().
        """
        actions = list(self.game.get_actions(state))
        if not actions:
            return None

        best_value = self.get_value(state)
        best_actions = [
            action for action in actions
            if self.get_q_value(state, action) == best_value
        ]
        return random.choice(best_actions)

    def update(self, state, action, next_state, reward):
        """Update Q-values using running average.
        Q(s,a) = (1 - α) Q(s,a) + α (R + γ V(s'))
        Where α is the learning rate, and γ is the discount.

        Note: You should not call this function in your code.
        """
        current_value = self.get_q_value(state, action)
        target = reward + self.discount * self.get_value(next_state)
        updated_value = (
            (1 - self.learning_rate) * current_value
            + self.learning_rate * target
        )
        self.q_values.setdefault(state, {})[action] = updated_value

    # 2. Epsilon Greedy
    def get_action(self, state):
        """Compute the action to take for the agent, incorporating exploration.
        That is, with probability ε, act randomly.
        Otherwise, act according to the best policy.

        Hint: use random.random() < ε to check if exploration is needed.
        """
        actions = list(self.game.get_actions(state))
        if not actions:
            return None
        if random.random() < self.explore_prob:
            return random.choice(actions)
        return self.get_best_policy(state)


# 3. Bridge Crossing Revisited
def question3():
    return 'NOT POSSIBLE'


# 5. Approximate Q-Learning
class ApproximateQAgent(QLearningAgent):
    """Implement Approximate Q Learning Agent using weights."""

    def __init__(self, *args, extractor):
        """Initialize parameters and store the feature extractor.
        Initialize weights table."""

        super().__init__(*args)
        self.extractor = extractor
        self.weights = {}

    def get_weight(self, feature):
        """Get weight of a feature.
        Never seen feature should have a weight of 0.
        """
        return self.weights.get(feature, 0.0)

    def get_q_value(self, state, action):
        """Compute Q value based on the dot product of feature
        components and weights.
        Q(s,a) = w_1 * f_1(s,a) + w_2 * f_2(s,a) + ... + w_n * f_n(s,a)
        """
        features = self.extractor(state, action)
        return sum(
            self.get_weight(feature) * value
            for feature, value in features.items()
        )

    def update(self, state, action, next_state, reward):
        """Update weights using least-squares approximation.
        Δ = R + γ V(s') - Q(s,a)
        Then update weights: w_i = w_i + α * Δ * f_i(s, a)
        """
        q_value = self.get_q_value(state, action)
        difference = (
            reward + self.discount * self.get_value(next_state) - q_value
        )
        features = self.extractor(state, action)
        for feature, value in features.items():
            self.weights[feature] = (
                self.get_weight(feature)
                + self.learning_rate * difference * value
            )


# 6. Feedback
# Just an approximation is fine.
feedback_question_1 = """
8
"""

feedback_question_2 = """
This homework is easier than others, just to follow the rules Do not need to
build algorithms.
"""

feedback_question_3 = """
It is a fun game.s
"""
