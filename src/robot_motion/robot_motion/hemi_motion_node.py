#!/usr/bin/env python3

import math
from typing import List

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PoseArray
from visualization_msgs.msg import MarkerArray, Marker


def normalize(v):
    n = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if n < 1e-9:
        return [0.0, 0.0, 0.0]
    return [v[0]/n, v[1]/n, v[2]/n]


def vadd(a, b):
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]


def vsub(a, b):
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]


def vcross(a, b):
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix (list of 3 lists) to quaternion [x,y,z,w]."""
    r00, r01, r02 = R[0]
    r10, r11, r12 = R[1]
    r20, r21, r22 = R[2]

    trace = r00 + r11 + r22
    if trace > 0.0:
        S = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * S
        qx = (r21 - r12) / S
        qy = (r02 - r20) / S
        qz = (r10 - r01) / S
    elif (r00 > r11) and (r00 > r22):
        S = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / S
        qx = 0.25 * S
        qy = (r01 + r10) / S
        qz = (r02 + r20) / S
    elif r11 > r22:
        S = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / S
        qx = (r01 + r10) / S
        qy = 0.25 * S
        qz = (r12 + r21) / S
    else:
        S = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / S
        qx = (r02 + r20) / S
        qy = (r12 + r21) / S
        qz = 0.25 * S

    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    return [qx/n, qy/n, qz/n, qw/n]


class HemiMotionNode(Node):
    def __init__(self):
        super().__init__("hemi_motion_node")

        # === Parameters ===
        self.target_frame = self.declare_parameter(
            "target_frame", "checkerboard_link"
        ).get_parameter_value().string_value

        center_default = [0.0, 0.0, 0.30]
        center_param = self.declare_parameter(
            "hemi_center_xyz", center_default
        ).get_parameter_value().double_array_value
        self.hemi_center_xyz = list(center_param) if len(center_param) == 3 else center_default

        self.hemi_radius = self.declare_parameter(
            "hemi_radius", 0.33
        ).get_parameter_value().double_value

        self.hemi_num_latitudes = self.declare_parameter(
            "hemi_num_latitudes", 20
        ).get_parameter_value().integer_value

        self.hemi_points_per_lat = self.declare_parameter(
            "hemi_points_per_lat", 50
        ).get_parameter_value().integer_value

        self.axis_yaw_deg = self.declare_parameter(
            "axis_yaw_deg", 0.0
        ).get_parameter_value().double_value

        self.max_poses = self.declare_parameter(
            "max_poses", 80
        ).get_parameter_value().integer_value

        self.get_logger().info(
            f"Hemisphere params: center={self.hemi_center_xyz}, "
            f"R={self.hemi_radius}, lat={self.hemi_num_latitudes}, "
            f"pts/lat={self.hemi_points_per_lat}, yaw={self.axis_yaw_deg} deg"
        )

        # === Publishers for RViz ===
        self.pose_array_pub = self.create_publisher(PoseArray, "hemi_waypoints", 1)
        self.marker_pub = self.create_publisher(MarkerArray, "hemi_waypoint_markers", 1)

        # === Generate poses & visualize ===
        self.poses: List[PoseStamped] = []
        self.generate_hemisphere_poses()
        self.publish_visualization()

        self.get_logger().warn(
            "MoveIt Python bindings (moveit_commander) not available in this container. "
            "This node only visualizes hemisphere waypoints; it does NOT move the robot yet."
        )

    def generate_hemisphere_poses(self):
        self.poses.clear()
        center = self.hemi_center_xyz
        look_at = [0.0, 0.0, 0.0]
        yaw = self.axis_yaw_deg * math.pi / 180.0

        for i in range(self.hemi_num_latitudes // 2):
            phi = math.pi * (i + 0.5) / float(self.hemi_num_latitudes)
            theta_shift = 0.0 if (i % 2 == 0) else (math.pi / float(self.hemi_points_per_lat))

            for j in range(self.hemi_points_per_lat):
                theta = 2.0 * math.pi * j / float(self.hemi_points_per_lat) + theta_shift

                x = self.hemi_radius * math.sin(phi) * math.cos(theta)
                yv = self.hemi_radius * math.sin(phi) * math.sin(theta)
                z = self.hemi_radius * math.cos(phi)

                pos = vadd(center, [x, yv, z])

                # z-axis points from pose to target
                z_axis = normalize(vsub(look_at, pos))

                # base x-axis rotated by yaw around Z
                x_base = [-1.0, 0.0, 0.0]
                x_desired = [
                    x_base[0] * math.cos(yaw) - x_base[1] * math.sin(yaw),
                    x_base[0] * math.sin(yaw) + x_base[1] * math.cos(yaw),
                    0.0,
                ]

                y_axis = normalize(vcross(z_axis, x_desired))
                x_axis = normalize(vcross(y_axis, z_axis))

                R = [
                    [x_axis[0], y_axis[0], z_axis[0]],
                    [x_axis[1], y_axis[1], z_axis[1]],
                    [x_axis[2], y_axis[2], z_axis[2]],
                ]
                qx, qy, qz, qw = rotation_matrix_to_quaternion(R)

                ps = PoseStamped()
                ps.header.frame_id = self.target_frame
                ps.pose.position.x = pos[0]
                ps.pose.position.y = pos[1]
                ps.pose.position.z = pos[2]
                ps.pose.orientation.x = qx
                ps.pose.orientation.y = qy
                ps.pose.orientation.z = qz
                ps.pose.orientation.w = qw

                self.poses.append(ps)

        self.get_logger().info(f"Generated {len(self.poses)} hemisphere poses")

        if len(self.poses) > self.max_poses:
            self.get_logger().info(
                f"Limiting hemisphere to {self.max_poses} poses (from {len(self.poses)})"
            )
            self.poses = self.poses[: self.max_poses]

    def publish_visualization(self):
        # PoseArray
        pa = PoseArray()
        pa.header.frame_id = self.target_frame
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.poses = [ps.pose for ps in self.poses]
        self.pose_array_pub.publish(pa)

        # MarkerArray (small spheres)
        ma = MarkerArray()
        for i, ps in enumerate(self.poses):
            m = Marker()
            m.header.frame_id = self.target_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "hemi_waypoints"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose = ps.pose
            m.scale.x = 0.02
            m.scale.y = 0.02
            m.scale.z = 0.02
            m.color.a = 1.0
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            ma.markers.append(m)

        self.marker_pub.publish(ma)
        self.get_logger().info("Published hemisphere visualization markers")


def main(args=None):
    rclpy.init(args=args)
    node = HemiMotionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
