
import numpy as np
import pybullet as p
import pybullet_data
import os
import time

class Node:
    def __init__(self, q):
        self.q = np.array(q)
        self.parent = None

class RRT:
    def __init__(self, start, goal, robot_id, joint_indices, step_size=0.1, max_iter=1000):
        self.start = Node(start)
        self.goal = Node(goal)
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.step_size = step_size
        self.max_iter = max_iter
        self.tree = [self.start]

    def distance(self, q1, q2):
        return np.linalg.norm(q1 - q2)

    def nearest_node(self, q_rand):
        nearest = self.tree[0]
        min_dist = self.distance(nearest.q, q_rand)
        for node in self.tree:
            dist = self.distance(node.q, q_rand)
            if dist < min_dist:
                nearest = node
                min_dist = dist
        return nearest

    def steer(self, q_nearest, q_rand):
        direction = (q_rand - q_nearest) / np.linalg.norm(q_rand - q_nearest)
        q_new = q_nearest + self.step_size * direction
        return q_new

    def is_collision_free(self, q):
        p.setJointMotorControlArray(self.robot_id, self.joint_indices, p.POSITION_CONTROL, q)
        p.stepSimulation()
        if p.getContactPoints(bodyA=self.robot_id):
            return False
        return True

    def build_tree(self):
        for i in range(self.max_iter):
            q_rand = np.random.uniform(low=-np.pi, high=np.pi, size=len(self.start.q))
            nearest_node = self.nearest_node(q_rand)
            q_new = self.steer(nearest_node.q, q_rand)
            if self.is_collision_free(q_new):
                new_node = Node(q_new)
                new_node.parent = nearest_node
                self.tree.append(new_node)
                if self.distance(q_new, self.goal.q) < self.step_size:
                    return self.get_path(new_node)
        return None

    def get_path(self, node):
        path = []
        while node is not None:
            path.append(node.q)
            node = node.parent
        return path[::-1]