#!/usr/bin/env python3
"""
Skeleton Optimization Agent - Template for students to implement optimization algorithms

IMPORTANT: This is NOT an RL agent - it's a template for 
students to implement their own optimization algorithms (greedy, Hungarian, etc.)

This agent currently makes terrible decisions:
- Random staffing decisions
- No layout optimization 
- Ignores order priorities
- No intelligent assignment logic

Students should replace the naive methods with proper optimization algorithms.
"""

import numpy as np
from typing import Dict, Optional
from .standardized_agents import BaselineAgent

class SkeletonOptimizationAgent(BaselineAgent):
    """
    Template for students to implement their own optimization algorithms.
    
    Students should replace the naive methods with:
    - Economic models for staffing decisions
    - Hungarian algorithm for order assignment  
    - Greedy search for layout optimization
    - Multi-objective optimization techniques
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.name = "StudentOptimization"
        
        # Enable unlimited hiring for true economic optimization
        self.env._unlimited_hiring = True
        
        # Students can add their own state tracking here
        self.performance_metrics = []
        self.decision_history = []
        
        # Students can add algorithm parameters here
        # Example: Hungarian algorithm matrices, greedy search parameters, etc.
        self.staffing_parameters = {
            'hire_threshold_ratio': 3.0,
            'fire_threshold_ratio': 2.0,
            'profit_threshold': 0
        }
        
        # Students can implement adaptive parameters that change based on performance
        self.adaptive_optimization_enabled = False
        
    def reset(self):
        """Reset agent state - students should expand this"""
        self.action_history = []
        self.reward_history = []
        # TODO: Reset any neural network states, replay buffers, etc.
    
    def get_action(self, observation: Dict) -> Dict:
        """
        Generate action based on observation.
        Current implementation is intentionally terrible - students should improve!
        """
        
        # Extract basic info from observation
        current_timestep = observation['time'][0]
        financial_state = observation['financial']  # [profit, revenue, costs, burn_rate]
        queue_info = observation['order_queue']
        employee_info = observation['employees']
        
        action = {
            'staffing_action': self._get_naive_staffing_action(financial_state, employee_info),
            'layout_swap': self._get_naive_layout_action(current_timestep),
            'order_assignments': self._get_naive_order_assignments(queue_info, employee_info)
        }
        
        # TODO: Students should implement proper action recording for optimization
        self.action_history.append(action.copy())
        
        return action
    
    def _get_naive_staffing_action(self, financial_state, employee_info) -> int:
        """
        WEEK 2 STEP 1: Economic staffing optimization
        Make hiring/firing decisions based on simple business economics
        """
        
        # Extract economic and operational data
        current_profit = financial_state[0]
        
        employees = self.env.employees
        num_workers = sum(1 for emp in employees if not emp.is_manager)
        has_manager = any(emp.is_manager for emp in employees)
        queue_length = len(self.env.order_queue.orders)
        
        # Algorithm parameters
        min_workers = 4            # Baseline capacity so orders always get picked up
        max_workers = 8            # Cap so salaries can't eat all the profit
        hire_threshold = 3.0       # Hire when more than 3 orders per worker are waiting
        fire_threshold = 1.5       # Fire when fewer than 1.5 orders per worker are waiting
        profit_threshold = 1000    # Only grow past baseline when we can afford wages
        
        # Business indicator: queue pressure (demand vs capacity)
        queue_pressure = queue_length / max(1, num_workers)
        
        # Always keep one manager so layout optimization can run
        if not has_manager:
            return 3  # Hire manager
        
        # Always keep a minimum crew
        if num_workers < min_workers:
            return 1  # Hire worker
        
        # HIRE: high demand + profitable + below the cap
        if (queue_pressure > hire_threshold and current_profit > profit_threshold
                and num_workers < max_workers):
            return 1  # Hire worker
        
        # FIRE: low demand + above the minimum crew
        # (buffer zone between 1.5 and 3.0 prevents hire/fire flip-flopping)
        if queue_pressure < fire_threshold and num_workers > min_workers:
            return 2  # Fire worker
        
        return 0  # No action
    
    def _get_naive_layout_action(self, current_timestep) -> list:
        """
        WEEK 1 STEP 1: Frequency-based layout optimization
        Move frequently accessed items closer to delivery areas
        """
        
        # Only optimize every 100 steps to avoid disrupting operations
        if current_timestep % 100 != 0:
            return [0, 0]
        
        grid = self.env.warehouse_grid
        
        # Get item frequency data from environment
        frequencies = grid.item_access_frequency
        active_frequencies = frequencies[frequencies > 0]
        if len(active_frequencies) == 0:
            return [0, 0]
        
        # Identify "hot" items (top 25% by access frequency)
        hot_threshold = np.percentile(active_frequencies, 75)
        hot_items = [item for item in range(len(frequencies))
                     if frequencies[item] >= hot_threshold and frequencies[item] > 0]
        # Most popular items first
        hot_items.sort(key=lambda item: frequencies[item], reverse=True)
        
        # Find delivery positions
        delivery_positions = getattr(grid, 'truck_bay_positions', [(grid.width // 2, grid.height // 2)])
        
        # For each hot item, check if it's far from delivery
        for item in hot_items:
            for location in grid.find_item_locations(item):
                distance = min(grid.manhattan_distance(location, d) for d in delivery_positions)
                
                # If hot item is >3 steps from delivery, find closer position
                if distance > 3:
                    swap = self._find_closer_position(location, delivery_positions)
                    if swap:
                        return swap
        
        return [0, 0]  # No beneficial swap found
    
    def _find_closer_position(self, current_pos, delivery_positions):
        """
        Greedy neighborhood search for better item placement.
        
        Algorithm: Exhaustive search of all storage positions to find
        the position that minimizes distance to delivery points.
        
        Returns [current_index, target_index] if beneficial swap found.
        """
        grid = self.env.warehouse_grid
        frequencies = grid.item_access_frequency
        current_idx = current_pos[1] * grid.width + current_pos[0]
        current_dist = min(grid.manhattan_distance(current_pos, delivery_pos) 
                          for delivery_pos in delivery_positions)
        current_item = grid.item_grid[current_pos[1], current_pos[0]]
        current_freq = frequencies[current_item] if current_item >= 0 else 0
        
        best_idx = None
        best_improvement = 1  # Require more than 1 step closer
        
        for y in range(grid.height):
            for x in range(grid.width):
                # Only consider storage cells
                if grid.cell_types[y, x] != 1:
                    continue
                
                # Don't push out an item that is just as popular
                target_item = grid.item_grid[y, x]
                if target_item >= 0 and frequencies[target_item] >= current_freq:
                    continue
                
                candidate_dist = min(grid.manhattan_distance((x, y), d) for d in delivery_positions)
                improvement = current_dist - candidate_dist
                
                # Greedy selection: keep the position with the biggest improvement
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_idx = y * grid.width + x
        
        if best_idx is not None:
            return [current_idx, best_idx]
        return None  # No better position found
    
    def _get_naive_order_assignments(self, queue_info, employee_info) -> list:
        """
        WEEK 2 STEP 2: Worker-to-order matching optimization
        Greedily match idle workers to pending orders using distance and value
        """
        
        assignments = [0] * 20  # Initialize with no assignments
        
        idle_workers = self._get_idle_workers(employee_info)
        orders = self.env.order_queue.orders[:20]
        
        # If no workers or orders available, nothing to assign
        if not idle_workers or not orders:
            return assignments
        
        distance_weight = 0.7  # Walking efficiency matters most
        value_weight = 0.3     # Revenue matters too
        max_value = max(order.value for order in orders) or 1
        
        # Score every worker-order pair
        candidates = []
        for order_idx, order in enumerate(orders):
            if order.status != "pending":
                continue
            value_score = order.value / max_value
            for worker_idx, worker_pos in idle_workers:
                distance = self._calculate_order_distance(worker_pos, order)
                distance_score = 1 / (1 + distance)
                score = distance_weight * distance_score + value_weight * value_score
                candidates.append((score, worker_idx, order_idx))
        
        # Greedy assignment: best-scoring pairs first, each worker/order used once
        candidates.sort(reverse=True)
        assigned_workers = set()
        assigned_orders = set()
        for score, worker_idx, order_idx in candidates:
            if worker_idx in assigned_workers or order_idx in assigned_orders:
                continue
            assignments[order_idx] = worker_idx + 1  # Environment uses 1-based worker numbers
            assigned_workers.add(worker_idx)
            assigned_orders.add(order_idx)
        
        return assignments
    
    def _calculate_order_distance(self, worker_pos, order):
        """
        Calculate minimum distance from worker to any item needed for this order.
        """
        grid = self.env.warehouse_grid
        min_distance = float('inf')
        
        for item_type in order.items:
            for location in grid.find_item_locations(item_type):
                distance = grid.manhattan_distance(worker_pos, location)
                if distance < min_distance:
                    min_distance = distance
        
        return min_distance if min_distance != float('inf') else 0
    
    def _get_idle_workers(self, employee_info):
        """
        Identify workers available for order assignment.
        Returns list of (worker_index, worker_position) for idle, non-manager workers.
        """
        idle_workers = []
        
        for i, employee in enumerate(self.env.employees[:len(employee_info)]):
            if (not employee.is_manager and employee.state.value == 0  # 0 = IDLE
                    and employee.current_order_id is None):
                idle_workers.append((i, employee.position))
        
        return idle_workers
    
    def record_reward(self, reward: float):
        """
        WEEK 3 STEP 1: Reward tracking and learning - students should expand this
        
        TODO WEEK 3 STEP 1: Students should implement:
        - Proper reward tracking
        - Performance analysis
        - Adaptive parameter adjustment
        - Multi-objective optimization
        """
        self.reward_history.append(reward)
        
        # Skeleton optimization - doesn't actually learn anything useful
        if len(self.reward_history) > 10:
            # "Update" weights randomly (this doesn't actually improve performance)
            if reward > 0:
                self.staffing_weights += np.random.randn(4) * 0.01
                self.layout_weights += np.random.randn(3) * 0.01
    
    def should_update_policy(self) -> bool:
        """
        WEEK 3 STEP 2: Policy updates and adaptation - students should improve this
        
        TODO WEEK 3 STEP 2: Students should implement proper update schedules
        """
        return len(self.action_history) % 50 == 0  # Arbitrary update frequency
    
    def get_performance_metrics(self) -> Dict:
        """
        Get agent performance metrics for analysis
        Students can use this to debug their improvements
        """
        if not self.reward_history:
            return {"avg_reward": 0, "total_actions": 0}
        
        return {
            "avg_reward": np.mean(self.reward_history[-100:]),  # Last 100 rewards
            "total_actions": len(self.action_history),
            "exploration_rate": self.exploration_rate,
            "recent_performance": np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else 0
        }


def create_skeleton_optimization_agent(env) -> SkeletonOptimizationAgent:
    """Factory function to create skeleton Optimization agent"""
    return SkeletonOptimizationAgent(env)


# TODO: Students should implement these advanced components:

class StudentOptimizationAgent(SkeletonOptimizationAgent):
    """
    Template for students to implement their improved Optimization agent
    
    Suggested improvements:
    1. Replace random staffing with demand-based hiring
    2. Implement proper layout optimization using item frequencies
    3. Add distance-based order assignment
    4. Implement basic Q-optimization or policy gradients
    5. Add proper exploration vs exploitation balance
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.name = "StudentOptimization"
        
        # TODO: Students implement these
        # self.q_table = {}
        # self.policy_network = SimpleNeuralNetwork()
        # self.experience_buffer = []
        # self.target_network = None
        
    def _get_improved_staffing_action(self, financial_state, employee_info, queue_info):
        """
        WEEK 2 STEP 1: Students implement intelligent staffing:
        - Hire when queue is growing
        - Fire when queue is empty for extended periods
        - Consider profit margins before hiring
        - Balance managers vs workers
        """
        pass
    
    def _get_improved_layout_action(self, observation):
        """
        WEEK 1 STEP 1: Students implement smart layout optimization:
        - Move frequently accessed items closer to delivery
        - Group items that are often ordered together
        - Only optimize when queue is manageable
        - Track swap effectiveness
        """
        pass
    
    def _get_improved_order_assignments(self, queue_info, employee_info):
        """
        WEEK 2 STEP 2: Students implement efficient order assignment:
        - Assign orders to closest available employees
        - Prioritize high-value or urgent orders
        - Consider employee current locations
        - Balance workload across employees
        """
        pass
    
    def learn_from_experience(self, state, action, reward, next_state, done):
        """
        WEEK 3 STEP 3: Students implement multi-objective optimization:
        - Performance trend analysis
        - Adaptive parameter tuning
        - Multi-objective trade-off handling
        - Robust optimization techniques
        """
        pass