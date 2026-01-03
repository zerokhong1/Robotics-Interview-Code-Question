🎯 Chuẩn bị câu hỏi Code Interview cho AI/Robotics
Tôi sẽ tổng hợp các câu hỏi coding có thể được hỏi trong interview, từ cơ bản đến nâng cao, phù hợp với các vị trí tại VinRobotics/VinMotion.

📋 Mục lục

Python Fundamentals
Data Structures & Algorithms
NumPy & Linear Algebra
Computer Vision Coding
SLAM/Robotics Specific
System Design Questions
Deep Learning & Neural Networks
ROS/ROS2 Questions
Sensor Fusion
Motion Planning
Optimization Problems
Practical Coding Challenges
Behavioral & System Design


1. Python Fundamentals
Q1.1: Giải thích sự khác nhau giữa List và Tuple?
python"""
ANSWER:
- List: Mutable, có thể thay đổi sau khi tạo
- Tuple: Immutable, không thể thay đổi

Khi nào dùng gì:
- List: Khi data cần thay đổi (append, remove, modify)
- Tuple: Khi data cố định (coordinates, RGB values, return multiple values)
"""

```python
# List - mutable
my_list = [1, 2, 3]
my_list.append(4)  # OK
my_list[0] = 10    # OK

# Tuple - immutable
my_tuple = (1, 2, 3)
# my_tuple.append(4)  # ERROR
# my_tuple[0] = 10    # ERROR

# Practical example in robotics
def get_robot_position():
    x, y, theta = 1.0, 2.0, 0.5
    return (x, y, theta)  # Tuple - position shouldn't be accidentally modified

position = get_robot_position()
x, y, theta = position  # Unpacking
```

Q1.2: Giải thích List Comprehension và khi nào nên dùng?
python"""
ANSWER:
List comprehension là cách viết ngắn gọn để tạo list mới từ iterable.
Syntax: [expression for item in iterable if condition]
"""
```
# Traditional way
squares = []
for i in range(10):
    squares.append(i ** 2)

# List comprehension (preferred)
squares = [i ** 2 for i in range(10)]

# With condition
even_squares = [i ** 2 for i in range(10) if i % 2 == 0]

# Practical example: Filter valid sensor readings
sensor_readings = [0.5, -1.0, 2.3, 0.0, 5.1, -0.5, 3.2]
valid_readings = [r for r in sensor_readings if r > 0]

# Nested comprehension (2D grid)
grid = [[i + j for j in range(3)] for i in range(3)]
# [[0, 1, 2], [1, 2, 3], [2, 3, 4]]

# WHEN NOT TO USE: Complex logic - use regular for loop for readability
```

Q1.3: Implement một Class cơ bản với các magic methods
python"""
QUESTION: Implement một class Point2D với các operations cơ bản
"""

```
import math

class Point2D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __repr__(self):
        """String representation for debugging"""
        return f"Point2D({self.x}, {self.y})"
    
    def __str__(self):
        """String representation for users"""
        return f"({self.x}, {self.y})"
    
    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, Point2D):
            return False
        return self.x == other.x and self.y == other.y
    
    def __add__(self, other):
        """Vector addition"""
        return Point2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Vector subtraction"""
        return Point2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Scalar multiplication"""
        return Point2D(self.x * scalar, self.y * scalar)
    
    def __abs__(self):
        """Magnitude (distance from origin)"""
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def distance_to(self, other):
        """Distance to another point"""
        return abs(self - other)
    
    def dot(self, other):
        """Dot product"""
        return self.x * other.x + self.y * other.y


# Usage
p1 = Point2D(3, 4)
p2 = Point2D(1, 2)

print(p1)           # (3, 4)
print(p1 + p2)      # (4, 6)
print(abs(p1))      # 5.0 (3-4-5 triangle)
print(p1.distance_to(p2))  # 2.828...
```

Q1.4: Decorators là gì? Viết một timing decorator
python"""
QUESTION: Viết decorator để measure execution time của function
"""

```
import time
from functools import wraps

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)  # Preserves function metadata
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


@timing_decorator
def slow_function(n):
    """Example slow function"""
    total = 0
    for i in range(n):
        total += i ** 2
    return total


# Usage
result = slow_function(1000000)
# Output: slow_function took 0.1234 seconds


# Advanced: Decorator with parameters
def repeat(times):
    """Decorator that repeats function n times"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(times):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator


@repeat(3)
def greet(name):
    return f"Hello, {name}!"

print(greet("Robot"))  # ['Hello, Robot!', 'Hello, Robot!', 'Hello, Robot!']
```

2. Data Structures & Algorithms
Q2.1: Implement Binary Search
python"""
QUESTION: Implement binary search iteratively and recursively
Time Complexity: O(log n)
Space Complexity: O(1) iterative, O(log n) recursive
"""
```
def binary_search_iterative(arr, target):
    """
    Find target in sorted array.
    Returns index if found, -1 otherwise.
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def binary_search_recursive(arr, target, left=None, right=None):
    """Recursive version"""
    if left is None:
        left, right = 0, len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)


# Test
arr = [1, 3, 5, 7, 9, 11, 13, 15]
print(binary_search_iterative(arr, 7))   # 3
print(binary_search_iterative(arr, 6))   # -1


# Practical application: Find closest timestamp in sensor data
def find_closest_timestamp(timestamps, query_time):
    """Find index of closest timestamp to query_time"""
    if not timestamps:
        return -1
    
    left, right = 0, len(timestamps) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if timestamps[mid] < query_time:
            left = mid + 1
        else:
            right = mid
    
    # Check neighbors for closest
    if left > 0:
        if abs(timestamps[left-1] - query_time) < abs(timestamps[left] - query_time):
            return left - 1
    
    return left
```

Q2.2: Implement BFS và DFS
python"""
QUESTION: Implement BFS and DFS for graph traversal
Application in robotics: Path planning, exploring environment
"""

```
from collections import deque

class Graph:
    def __init__(self):
        self.adjacency_list = {}
    
    def add_edge(self, u, v, directed=False):
        if u not in self.adjacency_list:
            self.adjacency_list[u] = []
        if v not in self.adjacency_list:
            self.adjacency_list[v] = []
        
        self.adjacency_list[u].append(v)
        if not directed:
            self.adjacency_list[v].append(u)
    
    def bfs(self, start):
        """
        Breadth-First Search
        - Uses QUEUE (FIFO)
        - Explores level by level
        - Good for: Shortest path (unweighted), level-order traversal
        """
        visited = set()
        queue = deque([start])
        result = []
        
        while queue:
            node = queue.popleft()
            
            if node not in visited:
                visited.add(node)
                result.append(node)
                
                for neighbor in self.adjacency_list.get(node, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return result
    
    def dfs(self, start):
        """
        Depth-First Search
        - Uses STACK (LIFO) or recursion
        - Explores as deep as possible first
        - Good for: Detecting cycles, topological sort, pathfinding
        """
        visited = set()
        result = []
        
        def dfs_recursive(node):
            if node in visited:
                return
            
            visited.add(node)
            result.append(node)
            
            for neighbor in self.adjacency_list.get(node, []):
                dfs_recursive(neighbor)
        
        dfs_recursive(start)
        return result
    
    def dfs_iterative(self, start):
        """DFS using explicit stack"""
        visited = set()
        stack = [start]
        result = []
        
        while stack:
            node = stack.pop()
            
            if node not in visited:
                visited.add(node)
                result.append(node)
                
                # Add neighbors in reverse for consistent ordering
                for neighbor in reversed(self.adjacency_list.get(node, [])):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return result


# Test
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(1, 4)
g.add_edge(2, 5)

print("BFS:", g.bfs(0))  # [0, 1, 2, 3, 4, 5] - level order
print("DFS:", g.dfs(0))  # [0, 1, 3, 4, 2, 5] - deep first
```

Q2.3: Implement A* Path Planning
python"""
QUESTION: Implement A* algorithm for path planning
This is VERY IMPORTANT for robotics interviews!
"""
```
import heapq
from typing import List, Tuple, Optional

def astar(grid: List[List[int]], 
          start: Tuple[int, int], 
          goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding algorithm.
    
    Args:
        grid: 2D grid where 0=free, 1=obstacle
        start: Starting position (row, col)
        goal: Goal position (row, col)
    
    Returns:
        List of positions from start to goal, or None if no path
    
    Time Complexity: O(E log V) where E=edges, V=vertices
    """
    rows, cols = len(grid), len(grid[0])
    
    # Validate inputs
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        return None
    
    def heuristic(a, b):
        """Euclidean distance heuristic"""
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    
    def get_neighbors(pos):
        """Get valid 8-connected neighbors"""
        r, c = pos
        neighbors = []
        
        # 8 directions: up, down, left, right, diagonals
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check bounds
            if 0 <= nr < rows and 0 <= nc < cols:
                # Check obstacle
                if grid[nr][nc] == 0:
                    # Diagonal movement cost
                    cost = 1.414 if dr != 0 and dc != 0 else 1.0
                    neighbors.append(((nr, nc), cost))
        
        return neighbors
    
    # Priority queue: (f_cost, g_cost, position, path)
    # f_cost = g_cost + h_cost
    start_h = heuristic(start, goal)
    heap = [(start_h, 0, start, [start])]
    
    # Track visited nodes and their g_costs
    visited = {}
    
    while heap:
        f_cost, g_cost, current, path = heapq.heappop(heap)
        
        # Goal reached
        if current == goal:
            return path
        
        # Skip if already visited with lower cost
        if current in visited and visited[current] <= g_cost:
            continue
        
        visited[current] = g_cost
        
        # Explore neighbors
        for neighbor, move_cost in get_neighbors(current):
            new_g = g_cost + move_cost
            
            if neighbor not in visited or visited[neighbor] > new_g:
                new_f = new_g + heuristic(neighbor, goal)
                heapq.heappush(heap, (new_f, new_g, neighbor, path + [neighbor]))
    
    return None  # No path found


def visualize_path(grid, path):
    """Visualize path on grid"""
    import copy
    vis_grid = copy.deepcopy(grid)
    
    for r, c in path:
        vis_grid[r][c] = 2  # Mark path
    
    vis_grid[path[0][0]][path[0][1]] = 3   # Start
    vis_grid[path[-1][0]][path[-1][1]] = 4  # Goal
    
    symbols = {0: '.', 1: '#', 2: '*', 3: 'S', 4: 'G'}
    
    for row in vis_grid:
        print(' '.join(symbols[cell] for cell in row))


# Test
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

path = astar(grid, (0, 0), (4, 4))
print("Path:", path)
visualize_path(grid, path)

"""
Output:
S * . . .
. # # # *
. . . # *
. # . . *
. . . . G
"""
```

3. NumPy & Linear Algebra
Q3.1: Matrix Operations cơ bản
python"""
QUESTION: Thực hiện các phép toán matrix cơ bản với NumPy
"""
```
import numpy as np

# ============================================
# 1. Tạo matrices
# ============================================

# Từ list
A = np.array([[1, 2], [3, 4]])

# Special matrices
identity = np.eye(3)          # Identity matrix
zeros = np.zeros((3, 3))      # All zeros
ones = np.ones((2, 4))        # All ones
random = np.random.rand(3, 3) # Random [0, 1)

# ============================================
# 2. Basic operations
# ============================================

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise operations
print(A + B)      # Addition
print(A * B)      # Element-wise multiplication
print(A ** 2)     # Element-wise square

# Matrix multiplication
print(A @ B)              # Preferred way
print(np.dot(A, B))       # Alternative
print(np.matmul(A, B))    # Alternative

# Transpose
print(A.T)

# ============================================
# 3. Important operations for robotics
# ============================================

# Inverse
A_inv = np.linalg.inv(A)
print(A @ A_inv)  # Should be identity

# Determinant
det = np.linalg.det(A)
print(f"Determinant: {det}")

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")

# SVD (Singular Value Decomposition)
U, S, Vt = np.linalg.svd(A)
# A = U @ np.diag(S) @ Vt

# Solve linear system Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print(f"Solution: {x}")

# ============================================
# 4. Practical: Rotation matrices
# ============================================

def rotation_matrix_2d(theta):
    """2D rotation matrix"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s, c]])

def rotation_matrix_3d_z(theta):
    """3D rotation around Z axis"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

# Rotate a point
point = np.array([1, 0])
R = rotation_matrix_2d(np.pi / 2)  # 90 degrees
rotated = R @ point
print(f"Rotated point: {rotated}")  # [0, 1]

Q3.2: Implement Homogeneous Transformations
python"""
QUESTION: Implement SE(3) transformation matrix operations
This is ESSENTIAL for robotics!
"""
```
import numpy as np

def rotation_matrix_x(theta):
    """Rotation around X axis"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_matrix_y(theta):
    """Rotation around Y axis"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rotation_matrix_z(theta):
    """Rotation around Z axis"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def make_transformation_matrix(R, t):
    """
    Create 4x4 homogeneous transformation matrix.
    
    T = [R  t]
        [0  1]
    
    Args:
        R: 3x3 rotation matrix
        t: 3x1 or (3,) translation vector
    
    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def inverse_transformation(T):
    """
    Inverse of transformation matrix.
    
    T^(-1) = [R^T  -R^T @ t]
             [0        1   ]
    """
    R = T[:3, :3]
    t = T[:3, 3]
    
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    
    return T_inv

def transform_point(T, point):
    """
    Transform a 3D point using transformation matrix.
    
    Args:
        T: 4x4 transformation matrix
        point: 3D point (3,) or (3, 1)
    
    Returns:
        Transformed 3D point
    """
    point_h = np.append(point.flatten(), 1)  # Homogeneous
    transformed_h = T @ point_h
    return transformed_h[:3]

def transform_points(T, points):
    """
    Transform multiple 3D points.
    
    Args:
        T: 4x4 transformation matrix
        points: Nx3 array of points
    
    Returns:
        Nx3 array of transformed points
    """
    N = points.shape[0]
    
    # Convert to homogeneous (Nx4)
    points_h = np.hstack([points, np.ones((N, 1))])
    
    # Transform
    transformed_h = (T @ points_h.T).T
    
    return transformed_h[:, :3]


# ============================================
# Example: Robot arm transformation chain
# ============================================

# Base to Link 1: Rotate 45° around Z, translate (0.5, 0, 0.3)
R1 = rotation_matrix_z(np.pi / 4)
t1 = np.array([0.5, 0, 0.3])
T_base_link1 = make_transformation_matrix(R1, t1)

# Link 1 to Link 2: Rotate 30° around Y, translate (0.3, 0, 0)
R2 = rotation_matrix_y(np.pi / 6)
t2 = np.array([0.3, 0, 0])
T_link1_link2 = make_transformation_matrix(R2, t2)

# Chain transformations: Base to Link 2
T_base_link2 = T_base_link1 @ T_link1_link2

# Point in Link 2 frame
point_link2 = np.array([0.1, 0, 0])

# Transform to base frame
point_base = transform_point(T_base_link2, point_link2)

print(f"Point in Link 2 frame: {point_link2}")
print(f"Point in Base frame: {point_base}")


# ============================================
# Verify inverse
# ============================================
T_link2_base = inverse_transformation(T_base_link2)
recovered = transform_point(T_link2_base, point_base)
print(f"Recovered point: {recovered}")  # Should match point_link2
```

Q3.3: Implement Quaternion Operations
python"""
QUESTION: Implement quaternion operations for rotation representation
Quaternions avoid gimbal lock and are efficient for rotation composition
"""
```
import numpy as np

class Quaternion:
    """
    Quaternion representation: q = w + xi + yj + zk
    Storage: [x, y, z, w] (Hamilton convention)
    """
    
    def __init__(self, x=0, y=0, z=0, w=1):
        self.q = np.array([x, y, z, w], dtype=np.float64)
    
    @property
    def x(self): return self.q[0]
    @property
    def y(self): return self.q[1]
    @property
    def z(self): return self.q[2]
    @property
    def w(self): return self.q[3]
    
    def __repr__(self):
        return f"Quaternion(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}, w={self.w:.4f})"
    
    def normalize(self):
        """Normalize to unit quaternion"""
        norm = np.linalg.norm(self.q)
        if norm > 0:
            self.q /= norm
        return self
    
    def conjugate(self):
        """Conjugate: q* = w - xi - yj - zk"""
        return Quaternion(-self.x, -self.y, -self.z, self.w)
    
    def inverse(self):
        """Inverse: q^(-1) = q* / |q|^2"""
        conj = self.conjugate()
        norm_sq = np.sum(self.q ** 2)
        return Quaternion(*(conj.q / norm_sq))
    
    def __mul__(self, other):
        """Quaternion multiplication (Hamilton product)"""
        x1, y1, z1, w1 = self.q
        x2, y2, z2, w2 = other.q
        
        return Quaternion(
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2,
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2,
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        )
    
    def rotate_vector(self, v):
        """
        Rotate vector v by this quaternion.
        v' = q * v * q^(-1)
        """
        v_quat = Quaternion(v[0], v[1], v[2], 0)
        rotated = self * v_quat * self.inverse()
        return np.array([rotated.x, rotated.y, rotated.z])
    
    def to_rotation_matrix(self):
        """Convert to 3x3 rotation matrix"""
        x, y, z, w = self.q
        
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    @classmethod
    def from_axis_angle(cls, axis, angle):
        """Create quaternion from axis-angle representation"""
        axis = np.array(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)  # Normalize
        
        half_angle = angle / 2
        s = np.sin(half_angle)
        
        return cls(
            x = axis[0] * s,
            y = axis[1] * s,
            z = axis[2] * s,
            w = np.cos(half_angle)
        ).normalize()
    
    @classmethod
    def from_rotation_matrix(cls, R):
        """Create quaternion from rotation matrix"""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return cls(x, y, z, w).normalize()


# Test
q1 = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)  # 90° around Z
print(f"Quaternion: {q1}")

# Rotate vector [1, 0, 0]
v = np.array([1, 0, 0])
v_rotated = q1.rotate_vector(v)
print(f"Rotated vector: {v_rotated}")  # Should be [0, 1, 0]

# Convert to rotation matrix
R = q1.to_rotation_matrix()
print(f"Rotation matrix:\n{R}")
```

4. Computer Vision Coding
Q4.1: Implement Camera Projection
python"""
QUESTION: Implement camera projection (3D → 2D)
Essential for understanding how cameras work!
"""

import numpy as np

def project_point(point_3d, K, R=None, t=None):
    """
    Project 3D point to 2D image coordinates.
    
    Pipeline: World → Camera → Normalized → Pixel
    
    Args:
        point_3d: 3D point in world frame (3,)
        K: 3x3 camera intrinsic matrix
        R: 3x3 rotation matrix (world to camera)
        t: 3x1 translation vector
    
    Returns:
        2D pixel coordinates (u, v)
    """
    # Default: camera at origin, aligned with world
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    
    # Transform to camera frame
    point_cam = R @ point_3d + t
    
    # Check if point is in front of camera
    if point_cam[2] <= 0:
        return None  # Behind camera
    
    # Project to normalized image plane
    x_norm = point_cam[0] / point_cam[2]
    y_norm = point_cam[1] / point_cam[2]
    
    # Apply intrinsics to get pixel coordinates
    point_norm = np.array([x_norm, y_norm, 1])
    pixel = K @ point_norm
    
    return pixel[:2]


def project_points(points_3d, K, R=None, t=None):
    """Project multiple 3D points"""
    projected = []
    for point in points_3d:
        p = project_point(point, K, R, t)
        if p is not None:
            projected.append(p)
    return np.array(projected)


def unproject_point(pixel, depth, K):
    """
    Unproject 2D pixel to 3D point given depth.
    
    Args:
        pixel: (u, v) pixel coordinates
        depth: Z depth value
        K: Camera intrinsic matrix
    
    Returns:
        3D point in camera frame
    """
    u, v = pixel
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    return np.array([x, y, z])


# Example
K = np.array([
    [500, 0, 320],
    [0, 500, 240],
    [0, 0, 1]
])

# 3D point at (1, 0.5, 5)
point_3d = np.array([1.0, 0.5, 5.0])

# Project
pixel = project_point(point_3d, K)
print(f"3D point {point_3d} projects to pixel {pixel}")

# Unproject with known depth
recovered = unproject_point(pixel, point_3d[2], K)
print(f"Unprojected back to {recovered}")

Q4.2: Implement Feature Matching với Ratio Test
python"""
QUESTION: Implement feature matching with Lowe's ratio test
"""

import cv2
import numpy as np

def detect_and_match(img1, img2, detector_type='ORB', ratio_threshold=0.75):
    """
    Detect features and match between two images.
    
    Args:
        img1, img2: Input images (grayscale)
        detector_type: 'ORB' or 'SIFT'
        ratio_threshold: Lowe's ratio test threshold
    
    Returns:
        kp1, kp2: Keypoints
        good_matches: Filtered matches
    """
    # Create detector
    if detector_type == 'ORB':
        detector = cv2.ORB_create(nfeatures=2000)
        norm_type = cv2.NORM_HAMMING
    else:  # SIFT
        detector = cv2.SIFT_create(nfeatures=2000)
        norm_type = cv2.NORM_L2
    
    # Detect and compute descriptors
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    
    print(f"Found {len(kp1)} keypoints in image 1")
    print(f"Found {len(kp2)} keypoints in image 2")
    
    if desc1 is None or desc2 is None:
        return kp1, kp2, []
    
    # Create matcher
    matcher = cv2.BFMatcher(norm_type)
    
    # KNN match (k=2 for ratio test)
    matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            # If best match is significantly better than second best
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    print(f"Good matches after ratio test: {len(good_matches)}")
    
    return kp1, kp2, good_matches


def compute_homography(kp1, kp2, matches, ransac_threshold=5.0):
    """
    Compute homography matrix from matches using RANSAC.
    
    Homography maps points from image 1 to image 2:
    p2 = H @ p1
    """
    if len(matches) < 4:
        return None, None
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Compute homography with RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_threshold)
    
    # Count inliers
    inliers = np.sum(mask)
    print(f"Homography inliers: {inliers}/{len(matches)}")
    
    return H, mask


def compute_fundamental_matrix(kp1, kp2, matches):
    """
    Compute fundamental matrix from matches.
    
    Fundamental matrix: p2^T @ F @ p1 = 0
    """
    if len(matches) < 8:
        return None, None
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
    
    return F, mask


# Example usage (pseudo-code without actual images)
"""
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

kp1, kp2, matches = detect_and_match(img1, img2)
H, mask = compute_homography(kp1, kp2, matches)

# Visualize
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
cv2.imshow('Matches', img_matches)
"""

Q4.3: Implement Compute Essential Matrix và Recover Pose
python"""
QUESTION: Implement essential matrix computation and pose recovery
This is the CORE of visual odometry!
"""

import numpy as np
import cv2

def compute_essential_matrix(pts1, pts2, K):
    """
    Compute Essential Matrix from point correspondences.
    
    Essential Matrix: p2^T @ E @ p1 = 0 (normalized coordinates)
    E = K^T @ F @ K
    E = [t]_x @ R
    
    Args:
        pts1, pts2: Nx2 arrays of corresponding points (pixel coords)
        K: 3x3 camera intrinsic matrix
    
    Returns:
        E: 3x3 Essential Matrix
        mask: Inlier mask
    """
    E, mask = cv2.findEssentialMat(
        pts1, pts2,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    
    return E, mask


def recover_pose_from_essential(E, pts1, pts2, K):
    """
    Recover rotation and translation from Essential Matrix.
    
    E can be decomposed into 4 possible (R, t) combinations.
    The correct one is determined by cheirality constraint
    (points must be in front of both cameras).
    
    Args:
        E: Essential Matrix
        pts1, pts2: Corresponding points
        K: Camera intrinsics
    
    Returns:
        R: 3x3 Rotation matrix
        t: 3x1 Translation vector (unit norm)
        mask: Inlier mask
    """
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    
    return R, t.flatten(), mask


def triangulate_points(pts1, pts2, R, t, K):
    """
    Triangulate 3D points from two views.
    
    Args:
        pts1, pts2: Corresponding points (Nx2)
        R, t: Relative pose (camera 2 w.r.t camera 1)
        K: Camera intrinsics
    
    Returns:
        points_3d: Nx3 array of 3D points
    """
    # Projection matrices
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    
    # Triangulate (OpenCV wants 2xN arrays)
    pts1_t = pts1.T
    pts2_t = pts2.T
    
    points_4d = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t)
    
    # Convert from homogeneous
    points_3d = (points_4d[:3] / points_4d[3]).T
    
    return points_3d


def decompose_essential_matrix_manual(E):
    """
    Manually decompose Essential Matrix to understand the math.
    
    E = U @ diag(1, 1, 0) @ V^T
    
    Possible solutions:
    R1 = U @ W @ V^T
    R2 = U @ W^T @ V^T
    t = ±u3 (third column of U)
    """
    U, S, Vt = np.linalg.svd(E)
    
    # Ensure proper rotation (det = +1)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    
    # W matrix for rotation extraction
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    
    # Two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # Translation (up to scale)
    t = U[:, 2]
    
    # Four possible combinations
    solutions = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]
    
    return solutions


# Example
print("Essential Matrix Decomposition Example")
print("=" * 50)

# Create a known rotation and translation
theta = np.pi / 6  # 30 degrees
R_true = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])
t_true = np.array([1, 0, 0.5])
t_true = t_true / np.linalg.norm(t_true)  # Normalize

# Skew-symmetric matrix [t]_x
def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

# Construct Essential Matrix
E_true = skew(t_true) @ R_true

print(f"True R:\n{R_true}")
print(f"True t: {t_true}")
print(f"Essential Matrix:\n{E_true}")

# Decompose and verify
solutions = decompose_essential_matrix_manual(E_true)
print(f"\nFound {len(solutions)} possible solutions")

5. SLAM/Robotics Specific
Q5.1: Implement Kalman Filter
python"""
QUESTION: Implement Kalman Filter for 1D position tracking
This is FUNDAMENTAL for state estimation!
"""

import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter1D:
    """
    1D Kalman Filter for tracking position and velocity.
    
    State: [position, velocity]
    Measurement: position only
    """
    
    def __init__(self, initial_position=0, initial_velocity=0):
        # State vector [position, velocity]
        self.x = np.array([initial_position, initial_velocity])
        
        # State covariance
        self.P = np.array([
            [1, 0],
            [0, 1]
        ], dtype=float)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 1],  # position += velocity * dt (dt=1)
            [0, 1]   # velocity stays same
        ], dtype=float)
        
        # Measurement matrix (we only observe position)
        self.H = np.array([[1, 0]], dtype=float)
        
        # Process noise covariance
        self.Q = np.array([
            [0.1, 0],
            [0, 0.1]
        ], dtype=float)
        
        # Measurement noise covariance
        self.R = np.array([[1]], dtype=float)
    
    def predict(self):
        """Prediction step"""
        # State prediction
        self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy()
    
    def update(self, measurement):
        """Update step with measurement"""
        # Innovation (measurement residual)
        y = measurement - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + (K @ y).flatten()
        
        # Covariance update
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x.copy()
    
    def get_position(self):
        return self.x[0]
    
    def get_velocity(self):
        return self.x[1]


# Demonstration
np.random.seed(42)

# True trajectory: constant velocity with some acceleration
true_positions = []
true_velocity = 1.0
position = 0.0

for i in range(50):
    position += true_velocity
    if i == 25:  # Sudden acceleration
        true_velocity = 2.0
    true_positions.append(position)

true_positions = np.array(true_positions)

# Noisy measurements
measurement_noise = 3.0
measurements = true_positions + np.random.normal(0, measurement_noise, len(true_positions))

# Run Kalman Filter
kf = KalmanFilter1D()
estimated_positions = []
estimated_velocities = []

for z in measurements:
    kf.predict()
    kf.update(z)
    estimated_positions.append(kf.get_position())
    estimated_velocities.append(kf.get_velocity())

# Plot results
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(true_positions, 'g-', label='True Position', linewidth=2)
axes[0].plot(measurements, 'r.', label='Measurements', alpha=0.5)
axes[0].plot(estimated_positions, 'b-', label='Kalman Estimate', linewidth=2)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Position')
axes[0].legend()
axes[0].set_title('Kalman Filter: Position Tracking')

axes[1].plot([1]*25 + [2]*25, 'g-', label='True Velocity', linewidth=2)
axes[1].plot(estimated_velocities, 'b-', label='Estimated Velocity', linewidth=2)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Velocity')
axes[1].legend()
axes[1].set_title('Kalman Filter: Velocity Estimation')

plt.tight_layout()
plt.savefig('kalman_demo.png')
plt.show()

print("Notice how the filter:")
print("1. Smooths noisy measurements")
print("2. Estimates velocity even though we don't measure it")
print("3. Adapts when velocity changes (with some lag)")

Q5.2: Implement ICP (Iterative Closest Point)
python"""
QUESTION: Implement ICP algorithm for point cloud registration
Used in LiDAR SLAM for scan matching!
"""

import numpy as np
from scipy.spatial import KDTree

def icp(source, target, max_iterations=50, tolerance=1e-6):
    """
    Iterative Closest Point algorithm.
    
    Finds transformation (R, t) that aligns source to target:
    target ≈ R @ source + t
    
    Args:
        source: Nx3 source point cloud
        target: Mx3 target point cloud
        max_iterations: Maximum ICP iterations
        tolerance: Convergence threshold
    
    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        transformed_source: Aligned source points
    """
    # Initialize
    src = source.copy()
    R_total = np.eye(3)
    t_total = np.zeros(3)
    
    prev_error = float('inf')
    
    # Build KD-tree for target (fast nearest neighbor)
    tree = KDTree(target)
    
    for iteration in range(max_iterations):
        # Step 1: Find closest points in target for each source point
        distances, indices = tree.query(src)
        closest_points = target[indices]
        
        # Step 2: Compute centroids
        src_centroid = np.mean(src, axis=0)
        tgt_centroid = np.mean(closest_points, axis=0)
        
        # Step 3: Center the point clouds
        src_centered = src - src_centroid
        tgt_centered = closest_points - tgt_centroid
        
        # Step 4: Compute rotation using SVD
        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Step 5: Compute translation
        t = tgt_centroid - R @ src_centroid
        
        # Step 6: Apply transformation
        src = (R @ src.T).T + t
        
        # Accumulate transformation
        R_total = R @ R_total
        t_total = R @ t_total + t
        
        # Check convergence
        mean_error = np.mean(distances)
        
        if abs(prev_error - mean_error) < tolerance:
            print(f"Converged at iteration {iteration}")
            break
        
        prev_error = mean_error
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Mean error = {mean_error:.6f}")
    
    return R_total, t_total, src


def generate_test_data():
    """Generate test point clouds for ICP"""
    np.random.seed(42)
    
    # Create random source point cloud (bunny-like shape)
    n_points = 100
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = 1 + 0.2 * np.random.randn(n_points)
    
    source = np.column_stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ])
    
    # Create known transformation
    angle = np.pi / 6  # 30 degrees
    R_true = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    t_true = np.array([0.5, 0.3, 0.1])
    
    # Transform to create target
    target = (R_true @ source.T).T + t_true
    
    # Add noise to target
    target += np.random.randn(*target.shape) * 0.02
    
    return source, target, R_true, t_true


# Test ICP
source, target, R_true, t_true = generate_test_data()

print("Running ICP...")
print(f"True rotation angle: {np.arccos(R_true[0,0]) * 180 / np.pi:.1f} degrees")
print(f"True translation: {t_true}")
print()

R_est, t_est, aligned = icp(source, target)

print()
print(f"Estimated rotation angle: {np.arccos(np.clip(R_est[0,0], -1, 1)) * 180 / np.pi:.1f} degrees")
print(f"Estimated translation: {t_est}")

# Compute alignment error
error = np.mean(np.linalg.norm(aligned - target, axis=1))
print(f"Final alignment error: {error:.6f}")
```

---

## 6. System Design Questions

### Q6.1: Design một Object Tracking Pipeline
```
QUESTION: Thiết kế hệ thống tracking cho robot autonomous

ANSWER:

┌─────────────────────────────────────────────────────────────┐
│                   OBJECT TRACKING PIPELINE                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Camera ──► Detection ──► Tracking ──► Prediction ──► Output
│              (YOLOv8)      (Kalman)     (Motion)            │
│                 │             │            │                 │
│                 └─── Data Association ────┘                  │
│                      (Hungarian)                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Components:

1. DETECTION (per frame)
   - Input: Image
   - Output: List of bounding boxes + class + confidence
   - Algorithm: YOLOv8 or Faster R-CNN
   - Frequency: Every frame or every N frames

2. TRACKING (maintain state)
   - Input: Detections
   - Output: Tracks with IDs
   - Components:
     a. State representation: [x, y, w, h, vx, vy]
     b. Motion model: Constant velocity
     c. State estimator: Kalman Filter
   
3. DATA ASSOCIATION
   - Problem: Match detections to existing tracks
   - Metric: IoU (Intersection over Union)
   - Algorithm: Hungarian algorithm (optimal assignment)
   - Handling:
     - Matched: Update track
     - Unmatched detection: New track
     - Unmatched track: Increment age, predict only

4. TRACK MANAGEMENT
   - Track creation: min_hits before confirmed
   - Track deletion: max_age without match
   - Track states: Tentative → Confirmed → Lost → Deleted

Trade-offs:
- Detection frequency vs computation
- Track age parameters vs responsiveness
- Simple (IoU) vs complex (ReID features) association
```

---

### Q6.2: Design một Visual Odometry System
```
QUESTION: Thiết kế Visual Odometry system cho robot

ANSWER:

┌─────────────────────────────────────────────────────────────┐
│                 VISUAL ODOMETRY PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Frame t ──► Features ──► Matching ──► Geometry ──► Pose    │
│              (ORB)        (BF+Ratio)   (E-matrix)           │
│                               │                              │
│  Frame t-1 ─────────────────┘                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Components:

1. FEATURE EXTRACTION
   - Detector: ORB (fast) or SIFT (robust)
   - Number: 1000-2000 features
   - Distribution: Use grid-based extraction for uniform coverage

2. FEATURE MATCHING
   - Method: Brute force with ratio test
   - Ratio threshold: 0.75 (Lowe's)
   - Min matches: 8 (for Essential Matrix)

3. GEOMETRIC ESTIMATION
   - Compute Essential Matrix with RANSAC
   - Decompose to R, t (4 solutions → cheirality check)
   - Scale: Unknown in monocular (need external source)

4. POSE ACCUMULATION
   - T_world_current = T_world_prev @ T_prev_current
   - Watch for drift accumulation

Challenges & Solutions:

1. Scale drift
   - Solution: Ground plane constraint, known object sizes, IMU

2. Pure rotation
   - Solution: Check translation magnitude, use only rotation

3. Fast motion / blur
   - Solution: Limit max displacement, use motion prediction

4. Low texture
   - Solution: Fallback to direct methods, use IMU

5. Computational efficiency
   - Solution: Feature bucketing, parallel processing, GPU
```

---

## 📝 Quick Reference Card
```
┌─────────────────────────────────────────────────────────────┐
│            INTERVIEW CODING QUICK REFERENCE                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  TIME COMPLEXITY (nhớ!)                                     │
│  ─────────────────────                                       │
│  Binary Search: O(log n)                                    │
│  BFS/DFS: O(V + E)                                          │
│  A*: O(E log V)                                             │
│  Matrix multiplication: O(n³)                                │
│  SVD: O(min(mn², m²n))                                      │
│                                                              │
│  COMMON PATTERNS                                            │
│  ───────────────                                             │
│  Sliding window: Two pointers                               │
│  Graph: BFS (shortest), DFS (explore all)                   │
│  Optimization: Dynamic programming                          │
│  Robust estimation: RANSAC                                  │
│                                                              │
│  ROBOTICS FORMULAS                                          │
│  ─────────────────                                           │
│  Projection: p = K @ (R @ P + t) / Z                        │
│  Essential: x₂ᵀ @ E @ x₁ = 0                                │
│  Kalman: x̂ = x̂⁻ + K(z - Hx̂⁻)                               │
│  Transform: T₁₂ = T₁ @ T₂⁻¹                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘

7. Deep Learning & Neural Networks
Q7.1: Implement một Neural Network Layer từ đầu
python"""
QUESTION: Implement Linear Layer và ReLU activation từ đầu
Không dùng PyTorch/TensorFlow
"""

import numpy as np

class LinearLayer:
    """
    Fully connected layer: y = x @ W + b
    
    Forward: y = xW + b
    Backward: 
        - dL/dW = x^T @ dL/dy
        - dL/db = sum(dL/dy, axis=0)
        - dL/dx = dL/dy @ W^T
    """
    
    def __init__(self, input_dim, output_dim):
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.W = np.random.randn(input_dim, output_dim) * scale
        self.b = np.zeros(output_dim)
        
        # Cache for backward pass
        self.x = None
        
        # Gradients
        self.dW = None
        self.db = None
    
    def forward(self, x):
        """
        Forward pass
        x: (batch_size, input_dim)
        returns: (batch_size, output_dim)
        """
        self.x = x  # Cache input for backward
        return x @ self.W + self.b
    
    def backward(self, grad_output):
        """
        Backward pass
        grad_output: dL/dy (batch_size, output_dim)
        returns: dL/dx (batch_size, input_dim)
        """
        batch_size = self.x.shape[0]
        
        # Gradient w.r.t weights: dL/dW = x^T @ dL/dy
        self.dW = self.x.T @ grad_output / batch_size
        
        # Gradient w.r.t bias: dL/db = sum(dL/dy)
        self.db = np.mean(grad_output, axis=0)
        
        # Gradient w.r.t input: dL/dx = dL/dy @ W^T
        grad_input = grad_output @ self.W.T
        
        return grad_input
    
    def update(self, learning_rate):
        """Update parameters using gradients"""
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class ReLU:
    """
    ReLU activation: y = max(0, x)
    
    Forward: y = max(0, x)
    Backward: dL/dx = dL/dy * (x > 0)
    """
    
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
    
    def backward(self, grad_output):
        return grad_output * self.mask


class Sigmoid:
    """
    Sigmoid activation: y = 1 / (1 + exp(-x))
    
    Backward: dL/dx = dL/dy * y * (1 - y)
    """
    
    def __init__(self):
        self.output = None
    
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.output
    
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)


class SoftmaxCrossEntropy:
    """
    Combined Softmax + Cross Entropy for numerical stability
    
    Softmax: p_i = exp(x_i) / sum(exp(x_j))
    Cross Entropy: L = -sum(y_true * log(p))
    
    Combined gradient: dL/dx = p - y_true
    """
    
    def __init__(self):
        self.probs = None
        self.y_true = None
    
    def forward(self, logits, y_true):
        """
        logits: (batch_size, num_classes)
        y_true: (batch_size, num_classes) one-hot encoded
        """
        # Stable softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.y_true = y_true
        
        # Cross entropy loss
        epsilon = 1e-10
        loss = -np.mean(np.sum(y_true * np.log(self.probs + epsilon), axis=1))
        
        return loss
    
    def backward(self):
        """Gradient of loss w.r.t logits"""
        return (self.probs - self.y_true) / self.y_true.shape[0]


class SimpleNeuralNetwork:
    """
    Simple 2-layer neural network
    Input -> Linear -> ReLU -> Linear -> Softmax
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.fc1 = LinearLayer(input_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = LinearLayer(hidden_dim, output_dim)
        self.loss_fn = SoftmaxCrossEntropy()
    
    def forward(self, x, y_true):
        """Forward pass and compute loss"""
        h1 = self.fc1.forward(x)
        h1_relu = self.relu.forward(h1)
        logits = self.fc2.forward(h1_relu)
        loss = self.loss_fn.forward(logits, y_true)
        return loss
    
    def backward(self):
        """Backward pass"""
        grad = self.loss_fn.backward()
        grad = self.fc2.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad)
    
    def update(self, learning_rate):
        """Update all parameters"""
        self.fc1.update(learning_rate)
        self.fc2.update(learning_rate)
    
    def predict(self, x):
        """Get predictions"""
        h1 = self.fc1.forward(x)
        h1_relu = self.relu.forward(h1)
        logits = self.fc2.forward(h1_relu)
        return np.argmax(logits, axis=1)


# Test with XOR problem
print("Training Neural Network on XOR problem")
print("=" * 50)

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)  # One-hot

# Create network
net = SimpleNeuralNetwork(input_dim=2, hidden_dim=8, output_dim=2)

# Train
for epoch in range(1000):
    loss = net.forward(X, y)
    net.backward()
    net.update(learning_rate=0.5)
    
    if epoch % 200 == 0:
        predictions = net.predict(X)
        accuracy = np.mean(predictions == np.argmax(y, axis=1))
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.2%}")

print("\nFinal predictions:", net.predict(X))
print("True labels:      ", np.argmax(y, axis=1))

Q7.2: Giải thích Convolution và Implement
python"""
QUESTION: Implement 2D Convolution từ đầu
Quan trọng cho Computer Vision!
"""

import numpy as np

def conv2d_naive(image, kernel, stride=1, padding=0):
    """
    2D Convolution (naive implementation)
    
    Args:
        image: (H, W) input image
        kernel: (kH, kW) convolution kernel
        stride: Step size
        padding: Zero padding
    
    Returns:
        output: Convolved image
    
    Output size: ((H + 2*padding - kH) / stride) + 1
    """
    # Add padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    
    H, W = image.shape
    kH, kW = kernel.shape
    
    # Calculate output dimensions
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    
    output = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            # Extract patch
            h_start = i * stride
            w_start = j * stride
            patch = image[h_start:h_start+kH, w_start:w_start+kW]
            
            # Element-wise multiplication and sum
            output[i, j] = np.sum(patch * kernel)
    
    return output


def conv2d_vectorized(image, kernel, stride=1, padding=0):
    """
    Vectorized 2D convolution using im2col
    Much faster than naive implementation!
    """
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
    
    H, W = image.shape
    kH, kW = kernel.shape
    
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    
    # im2col: Extract all patches as columns
    cols = np.zeros((kH * kW, out_H * out_W))
    
    col_idx = 0
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            patch = image[h_start:h_start+kH, w_start:w_start+kW]
            cols[:, col_idx] = patch.flatten()
            col_idx += 1
    
    # Convolution as matrix multiplication
    kernel_flat = kernel.flatten()
    output_flat = kernel_flat @ cols
    
    return output_flat.reshape(out_H, out_W)


# Common kernels used in Computer Vision
KERNELS = {
    'identity': np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]),
    
    'edge_detect': np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ]),
    
    'sharpen': np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]),
    
    'blur': np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]) / 9,
    
    'gaussian_3x3': np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16,
    
    'sobel_x': np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]),
    
    'sobel_y': np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ])
}


# Test
print("Convolution Examples")
print("=" * 50)

# Create test image
image = np.array([
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [1, 2, 3, 4, 5]
], dtype=np.float32)

print("Input image:")
print(image)
print()

# Apply edge detection
edge = conv2d_naive(image, KERNELS['edge_detect'], padding=1)
print("After edge detection:")
print(edge)

Q7.3: Giải thích Backpropagation
python"""
QUESTION: Giải thích backpropagation qua ví dụ đơn giản
"""

import numpy as np

def backprop_example():
    """
    Simple computational graph:
    
    x ──┐
        ├──► z = x * y ──► L = (z - target)²
    y ──┘
    
    Forward: z = x * y, L = (z - target)²
    
    Backward (Chain Rule):
    dL/dz = 2(z - target)
    dL/dx = dL/dz * dz/dx = dL/dz * y
    dL/dy = dL/dz * dz/dy = dL/dz * x
    """
    
    # Forward pass
    x = 3.0
    y = 4.0
    target = 10.0
    
    # z = x * y
    z = x * y
    print(f"Forward: x={x}, y={y}, z=x*y={z}")
    
    # Loss = (z - target)²
    loss = (z - target) ** 2
    print(f"Loss = (z - target)² = ({z} - {target})² = {loss}")
    
    # Backward pass
    print("\nBackward pass (Chain Rule):")
    
    # dL/dz = 2(z - target)
    dL_dz = 2 * (z - target)
    print(f"dL/dz = 2(z - target) = 2({z} - {target}) = {dL_dz}")
    
    # dz/dx = y, dz/dy = x
    dz_dx = y
    dz_dy = x
    
    # Chain rule
    dL_dx = dL_dz * dz_dx
    dL_dy = dL_dz * dz_dy
    
    print(f"dL/dx = dL/dz * dz/dx = {dL_dz} * {dz_dx} = {dL_dx}")
    print(f"dL/dy = dL/dz * dz/dy = {dL_dz} * {dz_dy} = {dL_dy}")
    
    # Gradient descent update
    learning_rate = 0.01
    x_new = x - learning_rate * dL_dx
    y_new = y - learning_rate * dL_dy
    
    print(f"\nGradient descent (lr={learning_rate}):")
    print(f"x_new = x - lr * dL/dx = {x} - {learning_rate} * {dL_dx} = {x_new}")
    print(f"y_new = y - lr * dL/dy = {y} - {learning_rate} * {dL_dy} = {y_new}")
    
    # Verify loss decreases
    z_new = x_new * y_new
    loss_new = (z_new - target) ** 2
    print(f"\nNew z = {z_new:.4f}, New loss = {loss_new:.4f}")
    print(f"Loss decreased by {loss - loss_new:.4f}")


backprop_example()

print("\n" + "=" * 60)
print("KEY INSIGHT: Backpropagation = Chain Rule applied systematically")
print("=" * 60)
print("""
For any computation graph:

1. FORWARD: Compute outputs layer by layer
   - Save intermediate values for backward pass

2. BACKWARD: Compute gradients from output to input
   - Start with dL/d(output) 
   - Apply chain rule: dL/dx = dL/dy * dy/dx
   - Propagate gradients backward

3. UPDATE: Adjust parameters
   - param = param - learning_rate * gradient
""")

8. ROS/ROS2 Questions
Q8.1: Giải thích ROS2 Concepts
python"""
QUESTION: Giải thích các concepts cơ bản của ROS2 và viết code example
"""

# ============================================
# ROS2 CONCEPTS EXPLANATION
# ============================================

"""
1. NODE
   - Basic computation unit in ROS2
   - Single process that performs computation
   - Communicates via topics, services, actions

2. TOPIC
   - Named bus for message passing
   - Publisher/Subscriber pattern (async)
   - One-to-many, many-to-one, many-to-many
   - Example: /camera/image, /cmd_vel, /odom

3. SERVICE
   - Request/Response pattern (sync)
   - One-to-one communication
   - For quick operations
   - Example: /spawn, /set_pen

4. ACTION
   - Long-running tasks with feedback
   - Goal -> Feedback -> Result
   - Cancellable
   - Example: /navigate_to_pose

5. PARAMETER
   - Configuration values for nodes
   - Can be changed at runtime
   - Declared and accessed via node

6. TF2
   - Transform library
   - Manages coordinate frame relationships
   - Time-aware transformations
"""

# ============================================
# EXAMPLE: Simple Publisher/Subscriber
# ============================================

"""
# publisher_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        
        # Create publisher
        # queue_size=10: buffer up to 10 messages
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        
        # Create timer (0.5 second interval)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = MinimalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
"""

"""
# subscriber_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
"""

# ============================================
# QoS (Quality of Service) Profiles
# ============================================

"""
QoS determines communication reliability:

1. RELIABILITY
   - RELIABLE: Guaranteed delivery (TCP-like)
   - BEST_EFFORT: May lose messages (UDP-like)

2. DURABILITY
   - TRANSIENT_LOCAL: Late joiners get last message
   - VOLATILE: Late joiners miss old messages

3. HISTORY
   - KEEP_LAST(n): Keep last n messages
   - KEEP_ALL: Keep all messages (unbounded)

Common profiles:
- Sensor data: BEST_EFFORT (high frequency, ok to lose)
- Commands: RELIABLE (must not lose)
- Parameters: RELIABLE + TRANSIENT_LOCAL
"""

"""
# QoS Example
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Custom QoS for sensor data
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# Use in subscriber
self.subscription = self.create_subscription(
    Image,
    '/camera/image_raw',
    self.image_callback,
    sensor_qos
)
"""

print("ROS2 Architecture Overview")
print("=" * 60)
print("""
┌─────────────────────────────────────────────────────────────┐
│                      ROS2 ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────┐    Topics     ┌─────────┐                     │
│   │  Node A │──────────────►│  Node B │                     │
│   │ (Sensor)│   /image      │ (Vision)│                     │
│   └─────────┘               └────┬────┘                     │
│                                  │                           │
│                                  │ /detections              │
│                                  ▼                           │
│   ┌─────────┐    Service    ┌─────────┐                     │
│   │  Node C │◄─────────────►│  Node D │                     │
│   │(Planner)│   /plan_path  │(Control)│                     │
│   └─────────┘               └─────────┘                     │
│                                  │                           │
│                                  │ /cmd_vel                 │
│                                  ▼                           │
│                            ┌─────────┐                       │
│                            │  Robot  │                       │
│                            └─────────┘                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
""")

Q8.2: TF2 Transformations
python"""
QUESTION: Giải thích TF2 và cách sử dụng trong ROS2
"""

"""
TF2 (Transform Library):
- Manages coordinate frame transformations
- Time-aware (can query transforms at specific time)
- Tree structure (frames have parent-child relationships)

Common frames:
- map: Global fixed frame
- odom: Odometry frame (may drift)
- base_link: Robot body frame
- camera_link: Camera frame
- lidar_link: LiDAR frame

Transform chain: map -> odom -> base_link -> sensor_links
"""

# Pseudo-code for TF2 usage in ROS2
"""
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import tf_transformations

class TFExample(Node):
    def __init__(self):
        super().__init__('tf_example')
        
        # For publishing transforms
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # For listening to transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Timer to publish transforms
        self.timer = self.create_timer(0.1, self.broadcast_transform)
    
    def broadcast_transform(self):
        '''Publish a transform from base_link to camera_link'''
        t = TransformStamped()
        
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'  # Parent frame
        t.child_frame_id = 'camera_link'  # Child frame
        
        # Translation (camera is 0.1m forward, 0.05m up from base)
        t.transform.translation.x = 0.1
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.05
        
        # Rotation (camera tilted 30 degrees down)
        q = tf_transformations.quaternion_from_euler(0, 0.52, 0)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        
        self.tf_broadcaster.sendTransform(t)
    
    def lookup_transform(self):
        '''Look up transform between two frames'''
        try:
            # Get transform from camera_link to map
            transform = self.tf_buffer.lookup_transform(
                'map',           # Target frame
                'camera_link',   # Source frame
                rclpy.time.Time()  # Latest available
            )
            
            # Use transform
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            self.get_logger().info(f'Camera position in map: ({x}, {y})')
            
        except Exception as e:
            self.get_logger().warn(f'Could not get transform: {e}')
    
    def transform_point(self, point_in_camera):
        '''Transform a point from camera frame to map frame'''
        from geometry_msgs.msg import PointStamped
        
        # Create point in camera frame
        point = PointStamped()
        point.header.frame_id = 'camera_link'
        point.header.stamp = self.get_clock().now().to_msg()
        point.point.x = point_in_camera[0]
        point.point.y = point_in_camera[1]
        point.point.z = point_in_camera[2]
        
        try:
            # Transform to map frame
            point_in_map = self.tf_buffer.transform(point, 'map')
            return point_in_map
        except Exception as e:
            self.get_logger().error(f'Transform failed: {e}')
            return None
"""

print("TF2 Frame Tree Example")
print("=" * 60)
print("""
Typical robot TF tree:

                    map
                     │
                     │ (localization)
                     ▼
                   odom
                     │
                     │ (odometry)
                     ▼
                base_link
                /    │    \\
               /     │     \\
              ▼      ▼      ▼
         camera  lidar  imu_link
           │
           ▼
       camera_optical


Key relationships:
- map -> odom: From localization (SLAM, GPS)
- odom -> base_link: From wheel encoders
- base_link -> sensors: Fixed (from URDF)
""")

9. Sensor Fusion
Q9.1: IMU Preintegration Concept
python"""
QUESTION: Giải thích IMU preintegration và implement đơn giản
Đây là concept quan trọng trong Visual-Inertial Odometry!
"""

import numpy as np

class SimpleIMUPreintegration:
    """
    Simplified IMU preintegration between keyframes.
    
    IMU measures:
    - Angular velocity (gyroscope): ω
    - Linear acceleration (accelerometer): a
    
    Preintegration computes relative motion between keyframes
    WITHOUT knowing absolute pose at integration start.
    
    This allows relinearization when pose estimates change.
    """
    
    def __init__(self, gravity=np.array([0, 0, -9.81])):
        self.gravity = gravity
        self.reset()
    
    def reset(self):
        """Reset preintegration"""
        self.delta_R = np.eye(3)      # Rotation
        self.delta_v = np.zeros(3)     # Velocity
        self.delta_p = np.zeros(3)     # Position
        self.dt_sum = 0.0              # Total time
        
        # Covariance (for uncertainty)
        self.covariance = np.zeros((9, 9))
    
    def integrate(self, gyro, accel, dt):
        """
        Integrate one IMU measurement.
        
        Args:
            gyro: Angular velocity [wx, wy, wz] (rad/s)
            accel: Linear acceleration [ax, ay, az] (m/s²)
            dt: Time step (seconds)
        """
        # Rotation increment from gyroscope
        angle = np.linalg.norm(gyro) * dt
        if angle > 1e-10:
            axis = gyro / np.linalg.norm(gyro)
            dR = self._axis_angle_to_rotation(axis, angle)
        else:
            dR = np.eye(3)
        
        # Rotate acceleration to reference frame (remove body rotation)
        accel_ref = self.delta_R @ accel
        
        # Update preintegrated measurements
        # Position: p += v*dt + 0.5*a*dt²
        self.delta_p += self.delta_v * dt + 0.5 * accel_ref * dt**2
        
        # Velocity: v += a*dt
        self.delta_v += accel_ref * dt
        
        # Rotation: R = R @ dR
        self.delta_R = self.delta_R @ dR
        
        self.dt_sum += dt
    
    def _axis_angle_to_rotation(self, axis, angle):
        """Convert axis-angle to rotation matrix (Rodrigues)"""
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R
    
    def predict_state(self, R_i, v_i, p_i):
        """
        Predict state at time j given state at time i.
        
        Args:
            R_i: Rotation at time i (3x3)
            v_i: Velocity at time i (3,)
            p_i: Position at time i (3,)
        
        Returns:
            R_j, v_j, p_j: Predicted state at time j
        """
        # R_j = R_i @ delta_R
        R_j = R_i @ self.delta_R
        
        # v_j = v_i + g*dt + R_i @ delta_v
        v_j = v_i + self.gravity * self.dt_sum + R_i @ self.delta_v
        
        # p_j = p_i + v_i*dt + 0.5*g*dt² + R_i @ delta_p
        p_j = (p_i + 
               v_i * self.dt_sum + 
               0.5 * self.gravity * self.dt_sum**2 + 
               R_i @ self.delta_p)
        
        return R_j, v_j, p_j


# Example usage
print("IMU Preintegration Example")
print("=" * 60)

preint = SimpleIMUPreintegration()

# Simulate IMU measurements (robot moving forward with rotation)
dt = 0.01  # 100 Hz IMU
num_steps = 100  # 1 second of data

for i in range(num_steps):
    # Simulated IMU readings
    # Robot rotating 45 deg/s around Z and accelerating forward
    gyro = np.array([0, 0, np.pi/4])  # rad/s
    accel = np.array([1.0, 0, 9.81])   # m/s² (including gravity)
    
    preint.integrate(gyro, accel, dt)

print(f"Preintegrated over {preint.dt_sum:.2f} seconds")
print(f"Delta rotation (deg): {np.arccos((np.trace(preint.delta_R)-1)/2) * 180/np.pi:.1f}")
print(f"Delta velocity: {preint.delta_v}")
print(f"Delta position: {preint.delta_p}")

# Predict state from known initial state
R_init = np.eye(3)
v_init = np.zeros(3)
p_init = np.zeros(3)

R_pred, v_pred, p_pred = preint.predict_state(R_init, v_init, p_init)

print(f"\nPredicted position: {p_pred}")
print(f"Predicted velocity: {v_pred}")

Q9.2: Extended Kalman Filter for Sensor Fusion
python"""
QUESTION: Implement EKF for fusing GPS and IMU
"""

import numpy as np

class EKFSensorFusion:
    """
    Extended Kalman Filter for GPS + IMU fusion.
    
    State: [x, y, vx, vy, theta]
    - x, y: Position
    - vx, vy: Velocity
    - theta: Heading angle
    
    IMU provides:
    - Angular velocity (for theta prediction)
    - Linear acceleration (for velocity prediction)
    
    GPS provides:
    - Position measurement (x, y)
    """
    
    def __init__(self):
        # State: [x, y, vx, vy, theta]
        self.x = np.zeros(5)
        
        # State covariance
        self.P = np.eye(5) * 1.0
        
        # Process noise
        self.Q = np.diag([0.1, 0.1, 0.5, 0.5, 0.1])
        
        # GPS measurement noise
        self.R_gps = np.diag([1.0, 1.0])  # GPS position noise
        
        # IMU noise
        self.accel_noise = 0.1
        self.gyro_noise = 0.01
    
    def predict(self, accel, gyro, dt):
        """
        Prediction step using IMU data.
        
        Args:
            accel: Linear acceleration in body frame [ax, ay]
            gyro: Angular velocity [omega_z]
            dt: Time step
        """
        x, y, vx, vy, theta = self.x
        
        # Rotate acceleration to world frame
        c, s = np.cos(theta), np.sin(theta)
        ax_world = c * accel[0] - s * accel[1]
        ay_world = s * accel[0] + c * accel[1]
        
        # State prediction (nonlinear motion model)
        x_new = x + vx * dt + 0.5 * ax_world * dt**2
        y_new = y + vy * dt + 0.5 * ay_world * dt**2
        vx_new = vx + ax_world * dt
        vy_new = vy + ay_world * dt
        theta_new = theta + gyro * dt
        
        self.x = np.array([x_new, y_new, vx_new, vy_new, theta_new])
        
        # Jacobian of motion model w.r.t state
        F = np.eye(5)
        F[0, 2] = dt  # dx/dvx
        F[1, 3] = dt  # dy/dvy
        
        # Include rotation effect on acceleration
        F[0, 4] = (-s * accel[0] - c * accel[1]) * 0.5 * dt**2
        F[1, 4] = (c * accel[0] - s * accel[1]) * 0.5 * dt**2
        F[2, 4] = (-s * accel[0] - c * accel[1]) * dt
        F[3, 4] = (c * accel[0] - s * accel[1]) * dt
        
        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q * dt
    
    def update_gps(self, gps_pos):
        """
        Update step using GPS measurement.
        
        Args:
            gps_pos: GPS position [x, y]
        """
        # Measurement model: z = H @ x
        # We measure [x, y]
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])
        
        # Innovation
        z = np.array(gps_pos)
        y = z - H @ self.x
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_gps
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        I = np.eye(5)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_gps @ K.T
    
    def get_position(self):
        return self.x[:2]
    
    def get_velocity(self):
        return self.x[2:4]
    
    def get_heading(self):
        return self.x[4]


# Simulation
print("EKF Sensor Fusion: GPS + IMU")
print("=" * 60)

ekf = EKFSensorFusion()

# Simulate circular motion
dt = 0.01  # 100 Hz
gps_rate = 1.0  # GPS at 1 Hz

true_positions = []
estimated_positions = []
gps_measurements = []

omega = 0.5  # Angular velocity
speed = 2.0  # Forward speed
time_elapsed = 0

for step in range(500):  # 5 seconds
    time_elapsed += dt
    
    # True state (circular motion)
    true_theta = omega * time_elapsed
    true_x = (speed / omega) * np.sin(true_theta)
    true_y = (speed / omega) * (1 - np.cos(true_theta))
    
    true_positions.append([true_x, true_y])
    
    # IMU measurements (with noise)
    accel = np.array([0, speed * omega]) + np.random.randn(2) * 0.1
    gyro = omega + np.random.randn() * 0.01
    
    # EKF prediction
    ekf.predict(accel, gyro, dt)
    
    # GPS update (at lower rate)
    if step % int(1.0 / (dt * gps_rate)) == 0:
        gps_noise = np.random.randn(2) * 1.0
        gps_pos = np.array([true_x, true_y]) + gps_noise
        gps_measurements.append(gps_pos)
        ekf.update_gps(gps_pos)
    
    estimated_positions.append(ekf.get_position().copy())

true_positions = np.array(true_positions)
estimated_positions = np.array(estimated_positions)
gps_measurements = np.array(gps_measurements)

# Compute error
final_error = np.linalg.norm(true_positions[-1] - estimated_positions[-1])
print(f"Final position error: {final_error:.3f} m")
print(f"GPS measurements: {len(gps_measurements)}")
print(f"Total estimates: {len(estimated_positions)}")

10. Motion Planning
Q10.1: Implement RRT (Rapidly-exploring Random Tree)
python"""
QUESTION: Implement RRT algorithm for motion planning
"""

import numpy as np
from typing import List, Tuple, Optional

class Node:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent: Optional['Node'] = None
    
    def __repr__(self):
        return f"Node({self.x:.2f}, {self.y:.2f})"


class RRT:
    """
    Rapidly-exploring Random Tree
    
    Algorithm:
    1. Sample random point
    2. Find nearest node in tree
    3. Extend toward sample (limited step size)
    4. Check collision
    5. Add to tree if valid
    6. Repeat until goal reached
    """
    
    def __init__(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        bounds: Tuple[float, float, float, float],  # xmin, ymin, xmax, ymax
        obstacles: List[Tuple[float, float, float]],  # List of (x, y, radius)
        step_size: float = 0.5,
        goal_threshold: float = 0.5,
        max_iterations: int = 5000
    ):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.bounds = bounds
        self.obstacles = obstacles
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.max_iterations = max_iterations
        
        self.nodes: List[Node] = [self.start]
    
    def plan(self) -> Optional[List[Node]]:
        """
        Run RRT planning.
        Returns path from start to goal, or None if failed.
        """
        for i in range(self.max_iterations):
            # Sample random point (with goal bias)
            if np.random.random() < 0.1:  # 10% chance to sample goal
                sample = (self.goal.x, self.goal.y)
            else:
                sample = self._sample_random()
            
            # Find nearest node
            nearest = self._nearest_node(sample)
            
            # Extend toward sample
            new_node = self._extend(nearest, sample)
            
            if new_node is None:
                continue
            
            # Check collision
            if self._collision_free(nearest, new_node):
                new_node.parent = nearest
                self.nodes.append(new_node)
                
                # Check if goal reached
                if self._distance(new_node, self.goal) < self.goal_threshold:
                    self.goal.parent = new_node
                    return self._extract_path()
            
            if i % 500 == 0:
                print(f"Iteration {i}, nodes: {len(self.nodes)}")
        
        print("Failed to find path")
        return None
    
    def _sample_random(self) -> Tuple[float, float]:
        """Sample random point in bounds"""
        x = np.random.uniform(self.bounds[0], self.bounds[2])
        y = np.random.uniform(self.bounds[1], self.bounds[3])
        return (x, y)
    
    def _nearest_node(self, point: Tuple[float, float]) -> Node:
        """Find nearest node to point"""
        min_dist = float('inf')
        nearest = None
        
        for node in self.nodes:
            dist = np.sqrt((node.x - point[0])**2 + (node.y - point[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _extend(self, from_node: Node, to_point: Tuple[float, float]) -> Optional[Node]:
        """Extend from node toward point (limited by step_size)"""
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < 1e-6:
            return None
        
        # Limit to step size
        if dist > self.step_size:
            dx = dx / dist * self.step_size
            dy = dy / dist * self.step_size
        
        new_x = from_node.x + dx
        new_y = from_node.y + dy
        
        # Check bounds
        if not (self.bounds[0] <= new_x <= self.bounds[2] and
                self.bounds[1] <= new_y <= self.bounds[3]):
            return None
        
        return Node(new_x, new_y)
    
    def _collision_free(self, from_node: Node, to_node: Node) -> bool:
        """Check if path between nodes is collision-free"""
        # Check multiple points along the path
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            x = from_node.x + t * (to_node.x - from_node.x)
            y = from_node.y + t * (to_node.y - from_node.y)
            
            # Check against all obstacles
            for ox, oy, r in self.obstacles:
                if np.sqrt((x - ox)**2 + (y - oy)**2) < r:
                    return False
        
        return True
    
    def _distance(self, node1: Node, node2: Node) -> float:
        """Euclidean distance between nodes"""
        return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def _extract_path(self) -> List[Node]:
        """Extract path from goal to start"""
        path = []
        node = self.goal
        
        while node is not None:
            path.append(node)
            node = node.parent
        
        return path[::-1]  # Reverse to get start-to-goal


# Test
print("RRT Path Planning")
print("=" * 60)

# Setup environment
start = (1.0, 1.0)
goal = (9.0, 9.0)
bounds = (0, 0, 10, 10)
obstacles = [
    (3, 3, 1.5),
    (5, 5, 1.0),
    (7, 3, 1.2),
    (3, 7, 1.0),
    (7, 7, 1.0),
]

rrt = RRT(start, goal, bounds, obstacles)
path = rrt.plan()

if path:
    print(f"\nPath found with {len(path)} nodes:")
    for i, node in enumerate(path):
        print(f"  {i}: ({node.x:.2f}, {node.y:.2f})")
    
    # Path length
    length = sum(
        np.sqrt((path[i+1].x - path[i].x)**2 + (path[i+1].y - path[i].y)**2)
        for i in range(len(path) - 1)
    )
    print(f"\nTotal path length: {length:.2f}")

Q10.2: Implement PID Controller
python"""
QUESTION: Implement PID controller for robot motion control
"""

import numpy as np

class PIDController:
    """
    PID Controller
    
    Output = Kp * error + Ki * integral(error) + Kd * d(error)/dt
    
    Used for:
    - Velocity control
    - Position control
    - Heading control
    """
    
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limits: Tuple[float, float] = (-float('inf'), float('inf')),
        integral_limits: Tuple[float, float] = (-float('inf'), float('inf'))
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
    
    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        """
        Compute PID output.
        
        Args:
            setpoint: Desired value
            measurement: Current measured value
            dt: Time step
        
        Returns:
            Control output
        """
        # Error
        error = setpoint - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, *self.integral_limits)
        i_term = self.ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0
        d_term = self.kd * derivative
        
        self.prev_error = error
        
        # Total output
        output = p_term + i_term + d_term
        
        # Clamp output
        output = np.clip(output, *self.output_limits)
        
        return output
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.prev_error = 0.0


class DifferentialDriveRobot:
    """
    Simple differential drive robot simulation.
    
    State: [x, y, theta]
    Control: [v, omega] (linear velocity, angular velocity)
    """
    
    def __init__(self, x=0, y=0, theta=0):
        self.x = x
        self.y = y
        self.theta = theta
    
    def move(self, v: float, omega: float, dt: float):
        """Update robot state given controls"""
        # Simple kinematic model
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt
        
        # Normalize theta to [-pi, pi]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
    
    @property
    def state(self):
        return np.array([self.x, self.y, self.theta])


def go_to_goal(robot: DifferentialDriveRobot, 
               goal: Tuple[float, float],
               dt: float = 0.1,
               max_steps: int = 1000) -> List[np.ndarray]:
    """
    Drive robot to goal using PID control.
    
    Uses two PID controllers:
    1. Distance controller -> linear velocity
    2. Heading controller -> angular velocity
    """
    # PID controllers
    distance_pid = PIDController(kp=1.0, ki=0.0, kd=0.1, 
                                 output_limits=(0, 2.0))
    heading_pid = PIDController(kp=3.0, ki=0.0, kd=0.5,
                               output_limits=(-2.0, 2.0))
    
    trajectory = [robot.state.copy()]
    
    for step in range(max_steps):
        # Calculate errors
        dx = goal[0] - robot.x
        dy = goal[1] - robot.y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Check if goal reached
        if distance < 0.1:
            print(f"Goal reached in {step} steps!")
            break
        
        # Desired heading (angle to goal)
        desired_theta = np.arctan2(dy, dx)
        
        # Heading error (normalized to [-pi, pi])
        heading_error = desired_theta - robot.theta
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Compute controls
        v = distance_pid.compute(distance, 0, dt)  # Distance error
        omega = heading_pid.compute(0, -heading_error, dt)  # Heading error
        
        # Reduce speed when not facing goal
        v *= np.cos(heading_error)
        v = max(0, v)
        
        # Move robot
        robot.move(v, omega, dt)
        trajectory.append(robot.state.copy())
    
    return trajectory


# Test
print("PID Control: Go to Goal")
print("=" * 60)

robot = DifferentialDriveRobot(x=0, y=0, theta=0)
goal = (5.0, 5.0)

print(f"Start: ({robot.x}, {robot.y})")
print(f"Goal: {goal}")

trajectory = go_to_goal(robot, goal)

print(f"Final: ({robot.x:.2f}, {robot.y:.2f})")
print(f"Distance to goal: {np.sqrt((robot.x-goal[0])**2 + (robot.y-goal[1])**2):.3f}")

11. Optimization Problems
Q11.1: Implement Gradient Descent Variants
python"""
QUESTION: Implement SGD, Momentum, và Adam optimizer
"""

import numpy as np

class Optimizer:
    """Base class for optimizers"""
    def __init__(self, learning_rate):
        self.lr = learning_rate
    
    def step(self, params, grads):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    
    θ = θ - lr * ∇L
    """
    def step(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]


class MomentumSGD(Optimizer):
    """
    SGD with Momentum
    
    v = β * v + ∇L
    θ = θ - lr * v
    
    Momentum helps:
    - Accelerate in consistent directions
    - Dampen oscillations
    """
    def __init__(self, learning_rate, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}
    
    def step(self, params, grads):
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            
            self.velocity[key] = self.momentum * self.velocity[key] + grads[key]
            params[key] -= self.lr * self.velocity[key]


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation)
    
    m = β1 * m + (1 - β1) * g          # First moment (mean)
    v = β2 * v + (1 - β2) * g²         # Second moment (variance)
    m̂ = m / (1 - β1^t)                 # Bias correction
    v̂ = v / (1 - β2^t)
    θ = θ - lr * m̂ / (√v̂ + ε)
    
    Combines:
    - Momentum (first moment)
    - RMSprop (second moment)
    - Bias correction
    """
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
    
    def step(self, params, grads):
        self.t += 1
        
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# Test on Rosenbrock function
def rosenbrock(x, y):
    """
    Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
    Minimum at (1, 1)
    """
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    """Gradient of Rosenbrock"""
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return {'x': dx, 'y': dy}


print("Optimizer Comparison on Rosenbrock Function")
print("=" * 60)

for OptimizerClass, name in [(SGD, "SGD"), (MomentumSGD, "Momentum"), (Adam, "Adam")]:
    params = {'x': -1.0, 'y': -1.0}
    
    if name == "SGD":
        optimizer = OptimizerClass(learning_rate=0.001)
    elif name == "Momentum":
        optimizer = OptimizerClass(learning_rate=0.001, momentum=0.9)
    else:
        optimizer = OptimizerClass(learning_rate=0.1)
    
    for i in range(1000):
        grads = rosenbrock_grad(params['x'], params['y'])
        optimizer.step(params, grads)
    
    loss = rosenbrock(params['x'], params['y'])
    print(f"{name:10s}: x={params['x']:.4f}, y={params['y']:.4f}, loss={loss:.6f}")

Q11.2: Implement Least Squares và RANSAC
python"""
QUESTION: Implement line fitting với Least Squares và RANSAC
"""

import numpy as np

def least_squares_line(points):
    """
    Fit line to points using least squares.
    
    Line model: y = mx + c
    Minimize: sum((y_i - (m*x_i + c))²)
    
    Solution: [m, c] = (X^T X)^(-1) X^T y
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Design matrix: [x, 1]
    X = np.column_stack([x, np.ones_like(x)])
    
    # Normal equations
    XtX = X.T @ X
    Xty = X.T @ y
    
    # Solve
    params = np.linalg.solve(XtX, Xty)
    
    return params[0], params[1]  # m, c


def ransac_line(points, n_iterations=100, threshold=0.5, min_inliers=None):
    """
    RANSAC line fitting.
    
    Algorithm:
    1. Randomly sample 2 points (minimum for line)
    2. Fit line through these points
    3. Count inliers (points within threshold)
    4. Keep best model (most inliers)
    5. Refit using all inliers
    
    Args:
        points: Nx2 array of points
        n_iterations: Number of RANSAC iterations
        threshold: Distance threshold for inliers
        min_inliers: Minimum inliers required (default: 50% of points)
    
    Returns:
        m, c: Line parameters
        inlier_mask: Boolean mask of inliers
    """
    n_points = len(points)
    if min_inliers is None:
        min_inliers = n_points // 2
    
    best_m, best_c = None, None
    best_inliers = None
    best_n_inliers = 0
    
    for _ in range(n_iterations):
        # 1. Random sample (2 points for line)
        idx = np.random.choice(n_points, 2, replace=False)
        sample = points[idx]
        
        # 2. Fit line through sample
        x1, y1 = sample[0]
        x2, y2 = sample[1]
        
        if abs(x2 - x1) < 1e-10:  # Vertical line
            continue
        
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        
        # 3. Count inliers
        # Distance from point (x0, y0) to line y = mx + c:
        # d = |y0 - mx0 - c| / sqrt(1 + m²)
        distances = np.abs(points[:, 1] - m * points[:, 0] - c) / np.sqrt(1 + m**2)
        inliers = distances < threshold
        n_inliers = np.sum(inliers)
        
        # 4. Update best model
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inliers = inliers
            best_m, best_c = m, c
    
    if best_n_inliers < min_inliers:
        return None, None, None
    
    # 5. Refit using all inliers
    inlier_points = points[best_inliers]
    final_m, final_c = least_squares_line(inlier_points)
    
    return final_m, final_c, best_inliers


# Test
print("Line Fitting: Least Squares vs RANSAC")
print("=" * 60)

np.random.seed(42)

# Generate data with outliers
n_inliers = 100
n_outliers = 30

# True line: y = 2x + 1
true_m, true_c = 2.0, 1.0

# Inliers (with noise)
x_inliers = np.random.uniform(0, 10, n_inliers)
y_inliers = true_m * x_inliers + true_c + np.random.randn(n_inliers) * 0.5

# Outliers (random)
x_outliers = np.random.uniform(0, 10, n_outliers)
y_outliers = np.random.uniform(-5, 25, n_outliers)

# Combine
x = np.concatenate([x_inliers, x_outliers])
y = np.concatenate([y_inliers, y_outliers])
points = np.column_stack([x, y])

# Shuffle
np.random.shuffle(points)

print(f"True line: y = {true_m}x + {true_c}")
print(f"Points: {n_inliers} inliers + {n_outliers} outliers")
print()

# Least squares (affected by outliers)
ls_m, ls_c = least_squares_line(points)
print(f"Least Squares: y = {ls_m:.3f}x + {ls_c:.3f}")
print(f"Error: m_err={abs(ls_m - true_m):.3f}, c_err={abs(ls_c - true_c):.3f}")
print()

# RANSAC (robust to outliers)
ransac_m, ransac_c, inliers = ransac_line(points, n_iterations=200, threshold=1.5)
if ransac_m is not None:
    print(f"RANSAC: y = {ransac_m:.3f}x + {ransac_c:.3f}")
    print(f"Error: m_err={abs(ransac_m - true_m):.3f}, c_err={abs(ransac_c - true_c):.3f}")
    print(f"Inliers found: {np.sum(inliers)}/{len(points)}")

12. Practical Coding Challenges
Q12.1: Process LiDAR Scan Data
python"""
QUESTION: Process LiDAR scan và detect obstacles
"""

import numpy as np

def process_lidar_scan(ranges, angle_min, angle_increment, 
                       range_min=0.1, range_max=30.0):
    """
    Process raw LiDAR scan data.
    
    Args:
        ranges: Array of distance measurements
        angle_min: Starting angle (radians)
        angle_increment: Angle between measurements (radians)
        range_min, range_max: Valid range bounds
    
    Returns:
        points: Nx2 array of (x, y) points in sensor frame
        valid_mask: Boolean mask of valid measurements
    """
    n_points = len(ranges)
    
    # Generate angles for each measurement
    angles = angle_min + np.arange(n_points) * angle_increment
    
    # Filter invalid measurements
    valid_mask = (ranges >= range_min) & (ranges <= range_max) & np.isfinite(ranges)
    
    # Convert to Cartesian coordinates
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    
    points = np.column_stack([x, y])
    
    return points, valid_mask


def cluster_points(points, eps=0.5, min_samples=3):
    """
    Simple clustering using distance-based method.
    
    Args:
        points: Nx2 array of points
        eps: Maximum distance between points in cluster
        min_samples: Minimum points to form cluster
    
    Returns:
        labels: Cluster label for each point (-1 for noise)
    """
    n_points = len(points)
    labels = np.full(n_points, -1)
    cluster_id = 0
    
    for i in range(n_points):
        if labels[i] != -1:
            continue
        
        # Find neighbors
        distances = np.linalg.norm(points - points[i], axis=1)
        neighbors = np.where(distances < eps)[0]
        
        if len(neighbors) < min_samples:
            continue  # Noise point
        
        # Start new cluster
        labels[neighbors] = cluster_id
        
        # Expand cluster
        seeds = list(neighbors)
        j = 0
        while j < len(seeds):
            q = seeds[j]
            q_distances = np.linalg.norm(points - points[q], axis=1)
            q_neighbors = np.where(q_distances < eps)[0]
            
            if len(q_neighbors) >= min_samples:
                for neighbor in q_neighbors:
                    if labels[neighbor] == -1:
                        labels[neighbor] = cluster_id
                        seeds.append(neighbor)
            
            j += 1
        
        cluster_id += 1
    
    return labels


def fit_bounding_box(points):
    """
    Fit axis-aligned bounding box to points.
    
    Returns:
        center: (x, y) center of box
        size: (width, height)
    """
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    
    center = (min_xy + max_xy) / 2
    size = max_xy - min_xy
    
    return center, size


def detect_obstacles(ranges, angle_min, angle_increment):
    """
    Full pipeline: LiDAR scan -> obstacle detections
    """
    # Process scan
    points, valid_mask = process_lidar_scan(ranges, angle_min, angle_increment)
    valid_points = points[valid_mask]
    
    if len(valid_points) < 3:
        return []
    
    # Cluster points
    labels = cluster_points(valid_points, eps=0.3, min_samples=5)
    
    # Extract obstacles
    obstacles = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1:  # Skip noise
            continue
        
        cluster_points_arr = valid_points[labels == label]
        center, size = fit_bounding_box(cluster_points_arr)
        
        obstacles.append({
            'center': center,
            'size': size,
            'n_points': len(cluster_points_arr),
            'distance': np.linalg.norm(center)
        })
    
    # Sort by distance
    obstacles.sort(key=lambda x: x['distance'])
    
    return obstacles


# Test
print("LiDAR Processing Example")
print("=" * 60)

# Simulate LiDAR scan with obstacles
np.random.seed(42)

n_beams = 360  # 1 degree resolution
angle_min = -np.pi
angle_increment = 2 * np.pi / n_beams

# Start with max range
ranges = np.full(n_beams, 10.0)

# Add obstacle 1: Object at 3m, 0 degrees (front)
for i in range(-15, 15):
    idx = (180 + i) % n_beams
    ranges[idx] = 3.0 + np.random.randn() * 0.05

# Add obstacle 2: Object at 5m, -45 degrees (right)
for i in range(-10, 10):
    idx = (135 + i) % n_beams
    ranges[idx] = 5.0 + np.random.randn() * 0.05

# Add some noise
ranges += np.random.randn(n_beams) * 0.02

# Detect obstacles
obstacles = detect_obstacles(ranges, angle_min, angle_increment)

print(f"Detected {len(obstacles)} obstacles:")
for i, obs in enumerate(obstacles):
    print(f"  Obstacle {i+1}:")
    print(f"    Center: ({obs['center'][0]:.2f}, {obs['center'][1]:.2f})")
    print(f"    Size: ({obs['size'][0]:.2f}, {obs['size'][1]:.2f})")
    print(f"    Distance: {obs['distance']:.2f}m")
    print(f"    Points: {obs['n_points']}")

Q12.2: Implement Image Stitching Pipeline
python"""
QUESTION: Implement basic image stitching (panorama)
"""

import numpy as np

def find_homography_dlt(src_pts, dst_pts):
    """
    Compute homography using Direct Linear Transform (DLT).
    
    H maps src to dst: dst = H @ src
    
    Args:
        src_pts: Nx2 source points
        dst_pts: Nx2 destination points
    
    Returns:
        H: 3x3 homography matrix
    """
    n = len(src_pts)
    
    if n < 4:
        raise ValueError("Need at least 4 point correspondences")
    
    # Build matrix A for DLT
    A = np.zeros((2 * n, 9))
    
    for i in range(n):
        x, y = src_pts[i]
        u, v = dst_pts[i]
        
        A[2*i] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        A[2*i + 1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    
    # Solution is last row of V^T (smallest singular value)
    H = Vt[-1].reshape(3, 3)
    
    # Normalize
    H = H / H[2, 2]
    
    return H


def ransac_homography(src_pts, dst_pts, n_iterations=1000, threshold=3.0):
    """
    Robust homography estimation using RANSAC.
    """
    n = len(src_pts)
    best_H = None
    best_inliers = 0
    
    for _ in range(n_iterations):
        # Random sample 4 points
        idx = np.random.choice(n, 4, replace=False)
        
        try:
            H = find_homography_dlt(src_pts[idx], dst_pts[idx])
        except:
            continue
        
        # Count inliers
        src_h = np.column_stack([src_pts, np.ones(n)])  # Nx3
        projected = (H @ src_h.T).T  # Nx3
        projected = projected[:, :2] / projected[:, 2:3]  # Normalize
        
        errors = np.linalg.norm(projected - dst_pts, axis=1)
        inliers = np.sum(errors < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H
            best_mask = errors < threshold
    
    # Refit with all inliers
    if best_H is not None and best_inliers >= 4:
        best_H = find_homography_dlt(src_pts[best_mask], dst_pts[best_mask])
    
    return best_H, best_mask


def warp_point(point, H):
    """Apply homography to a point"""
    p = np.array([point[0], point[1], 1])
    p_warped = H @ p
    return p_warped[:2] / p_warped[2]


def compute_panorama_bounds(img1_shape, img2_shape, H):
    """
    Compute bounds of stitched panorama.
    
    H transforms img2 into img1's coordinate system.
    """
    h1, w1 = img1_shape[:2]
    h2, w2 = img2_shape[:2]
    
    # Corners of img1
    corners1 = np.array([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ], dtype=np.float64)
    
    # Corners of img2 (transformed)
    corners2 = np.array([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ], dtype=np.float64)
    
    corners2_transformed = np.array([warp_point(c, H) for c in corners2])
    
    # Combined bounds
    all_corners = np.vstack([corners1, corners2_transformed])
    
    min_xy = np.floor(np.min(all_corners, axis=0)).astype(int)
    max_xy = np.ceil(np.max(all_corners, axis=0)).astype(int)
    
    return min_xy, max_xy


# Test
print("Image Stitching Pipeline")
print("=" * 60)

# Simulate matching points between two images
# True homography: slight rotation and translation
true_angle = 0.1  # 5.7 degrees
true_tx, true_ty = 300, 50

c, s = np.cos(true_angle), np.sin(true_angle)
H_true = np.array([
    [c, -s, true_tx],
    [s, c, true_ty],
    [0, 0, 1]
])

# Generate matching points
np.random.seed(42)
n_points = 50
n_outliers = 10

# Points in image 1
pts1 = np.random.uniform(100, 500, (n_points, 2))

# Corresponding points in image 2 (+ noise)
pts2 = np.array([warp_point(p, np.linalg.inv(H_true)) for p in pts1])
pts2 += np.random.randn(n_points, 2) * 2  # Add noise

# Add outliers
outlier_pts1 = np.random.uniform(100, 500, (n_outliers, 2))
outlier_pts2 = np.random.uniform(100, 500, (n_outliers, 2))

all_pts1 = np.vstack([pts1, outlier_pts1])
all_pts2 = np.vstack([pts2, outlier_pts2])

# Shuffle
perm = np.random.permutation(len(all_pts1))
all_pts1 = all_pts1[perm]
all_pts2 = all_pts2[perm]

# Estimate homography
H_estimated, inlier_mask = ransac_homography(all_pts2, all_pts1)

print(f"True Homography:\n{H_true}")
print(f"\nEstimated Homography:\n{H_estimated}")
print(f"\nInliers: {np.sum(inlier_mask)}/{len(all_pts1)}")

# Verify
test_point = np.array([200, 200])
true_warped = warp_point(test_point, H_true)
est_warped = warp_point(test_point, H_estimated)
print(f"\nTest point {test_point}:")
print(f"  True warp: {true_warped}")
print(f"  Estimated warp: {est_warped}")
print(f"  Error: {np.linalg.norm(true_warped - est_warped):.2f} pixels")
```

---

## 13. Behavioral & System Design

### Q13.1: System Design - Real-time Object Detection Pipeline
```
QUESTION: Design hệ thống object detection real-time cho robot

ANSWER:

┌─────────────────────────────────────────────────────────────┐
│           REAL-TIME DETECTION SYSTEM ARCHITECTURE            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Camera        Preprocessor       Detector       Tracker   │
│   (30fps) ────► (Resize/Norm) ────► (YOLO) ────► (DeepSORT) │
│                      │                 │              │      │
│                      │            ┌────┴────┐         │      │
│                      │            │ GPU     │         │      │
│                      │            │ Queue   │         │      │
│                      │            └─────────┘         │      │
│                      │                                │      │
│                      └────────────────────────────────┼──►   │
│                                                       │      │
│   Output: Tracked objects với ID, bbox, class, confidence    │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Components:

1. CAMERA INPUT
   - Resolution: 1280x720 @ 30fps
   - Interface: USB3 hoặc GigE
   - Buffer: Ring buffer để không block

2. PREPROCESSING (CPU)
   - Resize: 1280x720 → 640x640
   - Normalize: [0, 255] → [0, 1]
   - Format: BGR → RGB → Tensor
   - Batching: Accumulate frames if needed

3. DETECTION (GPU)
   - Model: YOLOv8n/s (based on speed requirement)
   - Batch size: 1 for low latency
   - TensorRT optimization for speed
   - Output: List of (bbox, class, confidence)

4. TRACKING (CPU/GPU)
   - Algorithm: DeepSORT or ByteTrack
   - Features: ReID for appearance matching
   - Output: Tracks with persistent IDs

5. POST-PROCESSING
   - NMS (usually in detector)
   - Confidence filtering
   - ROI filtering (ignore regions)

Performance Targets:
   - Latency: < 50ms end-to-end
   - Throughput: 30fps
   - Memory: < 4GB GPU

Optimization Strategies:
   - TensorRT/ONNX Runtime
   - FP16 inference
   - Async processing
   - Frame skipping if needed
```

---

### Q13.2: Behavioral Question Templates
```
QUESTION: Các câu hỏi behavioral phổ biến

1. "Kể về một project khó khăn bạn đã làm"

STRUCTURE (STAR):
- Situation: [Context và background]
- Task: [Mục tiêu của bạn]
- Action: [Bạn đã làm gì cụ thể]
- Result: [Kết quả và bài học]

EXAMPLE ANSWER:
"Trong project Visual Odometry, tôi gặp khó khăn khi implement
Essential Matrix decomposition. 

Situation: Code chạy nhưng trajectory sai hoàn toàn.

Task: Debug và tìm ra nguyên nhân.

Action: 
- Viết unit tests với synthetic data
- In ra intermediate values để verify từng step
- So sánh với OpenCV implementation
- Phát hiện lỗi ở phần cheirality check

Result: 
- Fix được bug sau 2 ngày debug
- Học được importance của testing
- Trajectory match với ground truth

Learning: Always test with known data first!"

---

2. "Khi nào bạn phải học skill mới nhanh?"

EXAMPLE ANSWER:
"Khi bắt đầu học SLAM, tôi chưa có background về Linear Algebra.

- Đặt timeline: 2 tuần để nắm basics
- Resources: MIT OCW, 3Blue1Brown videos
- Practice: Implement từ scratch với NumPy
- Apply: Dùng ngay vào project

Result: Có thể đọc hiểu papers về SLAM trong 2 tuần."

---

3. "Làm sao handle disagreement với teammate?"

FRAMEWORK:
- Listen first, understand their perspective
- Focus on facts and data, not opinions
- Find common ground
- Propose compromise or experiment

EXAMPLE:
"Disagreement về algorithm choice (Kalman vs Particle Filter):
- Cả hai present pros/cons
- Agree to implement both and compare
- Use metrics to decide (accuracy, speed)
- Choose Kalman cho case này, document why"

---

4. "Failure và học được gì?"

FRAMEWORK:
- Own the failure (don't blame others)
- Explain what went wrong
- What you learned
- How you prevent it in future

EXAMPLE:
"Project deadline missed vì underestimate complexity:
- Root cause: Không break down task đủ nhỏ
- Learning: Always add buffer time, break into smaller tasks
- Now: Use task estimation techniques, regular check-ins"
```

---

## 📝 Summary Cheat Sheet
```
┌─────────────────────────────────────────────────────────────┐
│                 INTERVIEW CHEAT SHEET                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ALGORITHMS                                                   │
│ ──────────                                                   │
│ Binary Search: O(log n) - sorted array                      │
│ BFS: O(V+E) - shortest path, level order                    │
│ DFS: O(V+E) - explore all, cycle detection                  │
│ A*: O(E log V) - optimal pathfinding                        │
│ Dijkstra: O(E log V) - shortest path (positive weights)     │
│ RANSAC: Random sample consensus - robust estimation         │
│                                                              │
│ DEEP LEARNING                                                │
│ ─────────────                                                │
│ Backprop: Chain rule - dL/dx = dL/dy * dy/dx               │
│ Adam: Momentum + RMSprop + bias correction                  │
│ Batch Norm: Normalize activations per batch                 │
│ Dropout: Regularization by randomly zeroing neurons         │
│                                                              │
│ ROBOTICS                                                     │
│ ────────                                                     │
│ Kalman: predict + update, optimal for linear Gaussian       │
│ EKF: Linearize nonlinear system with Jacobians              │
│ PID: P (response) + I (steady-state) + D (damping)         │
│ ICP: Align point clouds iteratively                         │
│                                                              │
│ COMPUTER VISION                                              │
│ ───────────────                                              │
│ Projection: p = K[R|t]P                                     │
│ Essential: x2^T E x1 = 0, E = [t]x R                        │
│ Homography: p2 = H p1 (planar transformation)               │
│ Epipolar: All matches lie on epipolar line                  │
│                                                              │
│ ROS2                                                         │
│ ────                                                         │
│ Topics: Pub/Sub, async, many-to-many                        │
│ Services: Request/Response, sync                             │
│ Actions: Long-running with feedback                          │
│ TF2: Coordinate frame transformations                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘


