import rospy
import numpy as np
import g2o
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt

Y = 0                   # Y position of the robot (since we are considering 1D)
THETA = 0               # Theta of robot (since we are consitering 1D)
SCAN_IDX = 0            # Index of the scan ranges for directly in front of bot
MAX_SCAN = 3            # largest acceptable scan value to prevent extraneous data
WALL_ID = 100           # set the ID for the wall (landmark) vertex
POSE_EDGE_CERT = 1      # uncertainty in the pose estimate
LAND_EDGE_CERT = 1000   # uncertainty in the lidar estimate
INIT = 0                # initial x-position

# class implementing graph (g2o) optimization. Adapted from template provided in assignment link
class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        g2o.SparseOptimizer.__init__(self)
        solver = g2o.BlockSolverSE2(g2o.LinearSolverCholmodSE2())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        self.set_algorithm(solver)

    # function to optimize graph
    def optimize(self, max_iterations=20):
        self.initialize_optimization()
        super(g2o.SparseOptimizer, self).optimize(max_iterations)
    
    # function to add a vertex to graph
    def add_vertex(self, id, pose, fixed=False):
        v_se2 = g2o.VertexSE2()
        v_se2.set_id(id)
        v_se2.set_estimate(pose)
        v_se2.set_fixed(fixed)
        super(g2o.SparseOptimizer, self).add_vertex(v_se2)

    # function to add landmark node
    def add_landmark(self, id, meas, fixed=True):
        v_se2 = g2o.VertexPointXY()
        v_se2.set_id(id)
        v_se2.set_estimate(meas)
        v_se2.set_fixed(fixed)
        super(g2o.SparseOptimizer, self).add_vertex(v_se2)

    # function to add edge between vertices
    def add_edge(self, vertices, meas, information=np.identity(3), robust_kernel=None):
        edge = g2o.EdgeSE2()
        for i,v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)
        edge.set_measurement(meas)
        edge.set_information(information)
        
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super(g2o.SparseOptimizer, self).add_edge(edge)

    # function to add edge to landmark node
    def add_landmark_edge(self, vertices, meas, information=np.identity(2), robust_kernel=None):
        edge = g2o.EdgeSE2PointXY()
        for i,v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)
        edge.set_measurement(meas)
        edge.set_information(information)
        
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super(g2o.SparseOptimizer, self).add_edge(edge)
    
    # Function to return the estimate info of a vertex
    def get_pose(self, id):
        return self.vertex(id).estimate()

# class to implement GraphPoseOptimization
class MakeGraph:
    def __init__(self):
        rospy.init_node('graph')
        self.scan_sub = rospy.Subscriber("scan", LaserScan, self.scan_callback)
        self.pose_sub = rospy.Subscriber("pose", PoseStamped, self.pose_callback)
        rospy.sleep(1)
        self.dist_wall = 2        # inital distance to wall
        self.prev_pos = 0         # init prev position
        self.pos = 0              # initial position
        self.wall_state = 0       # track if landmark node has been made
        self.id = 0               # inital node id      
        self.fig = plt.figure()   # figure init 
        self.poses = []           # track ground truth poses
        self.ids = []             # track ids for plotting with ground truth

        # make first vertex
        self.graph = PoseGraphOptimization()
        i = g2o.SE2()
        i.from_vector([INIT, Y, THETA])
        self.graph.add_vertex(self.id, i, True)
        self.poses.append(INIT)
        self.ids.append(self.id)
        self.id = self.id+1
        rospy.sleep(2)

    # callback information for scan. Initially will add landmard node to graph
    def scan_callback(self, msg):
        if msg.ranges[SCAN_IDX] <= MAX_SCAN:
            self.dist_wall = msg.ranges[SCAN_IDX]
        if self.wall_state == 0:
            # insert wall landmark node in graph
            p = [self.dist_wall, Y]
            self.graph.add_landmark(WALL_ID, p, True)
            self.wall_state = 1 

    # callback function for pose. Adds nodes and edges to graph
    def pose_callback(self, msg):
        self.prev_pos = self.pos
        self.pos = msg.pose.position.x
        self.poses.append(self.pos)
        self.ids.append(self.id)

        # make node for current position
        p = g2o.SE2()
        p.from_vector([self.pos, Y, THETA])
        self.graph.add_vertex(self.id, p)
        
        # add edge from previous node to current 
        dist = self.pos - self.prev_pos        # distance for edge
        m = g2o.SE2()
        m.from_vector([dist, Y, THETA]) 
        pose_info = np.eye(3)
        pose_info[0][0] = POSE_EDGE_CERT
        self.graph.add_edge([self.id-1, self.id], m, pose_info)

        # add edge from current node to landmark
        land_info = np.eye(2)
        land_info[0][0] = LAND_EDGE_CERT
        # add edge to landmark with current scan measurement 
        self.graph.add_landmark_edge([self.id, WALL_ID], [self.dist_wall, Y], land_info)
        self.id = self.id+1
   
    # function to call the optimization
    def optimize(self):
        self.graph.optimize()
        self.graph.save("optimized_graph.g2o")

    # function to plot the the graph information
    def plot(self, figname, dot):
        self.fig.suptitle("Comparison of Actual and Optimized", fontsize=20)
        plt.xlabel("Pose Number", fontsize=15)
        plt.ylabel("Estimate", fontsize=15)
        plt.plot([INIT], [self.graph.get_pose(INIT)[0]], dot, label=figname)
        for i in range(self.id):
            plt.plot([i], [self.graph.get_pose(i)[0]], dot)
        plt.plot([self.id+1], [self.graph.get_pose(WALL_ID)], dot)
        plt.legend(loc='upper left')

    # function to plot the ground truth
    def plot_ground(self):
        plt.plot(self.ids, self.poses, 'g-', label="ground truth")
 
if __name__ == '__main__':
    a = MakeGraph()
    rospy.sleep(18)
    a.graph.save("graph.g2o")
    a.plot_ground()
    a.plot("Odom", 'ro')
    a.optimize()
    a.plot("Optimized", 'b^')
    a.fig.savefig("plot.png")

