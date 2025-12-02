#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <moveit/move_group_interface/move_group_interface.h>

#include <cmath>
#include <vector>
#include <string>
#include <thread>
#include <array>

// ====== simple vector helpers ======

struct Vec3
{
  double x, y, z;
};

static Vec3 vadd(const Vec3 &a, const Vec3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static Vec3 vsub(const Vec3 &a, const Vec3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

static Vec3 vcross(const Vec3 &a, const Vec3 &b)
{
  return {
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x};
}

static Vec3 vnorm(const Vec3 &v)
{
  double n = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  if (n < 1e-9)
    return {0.0, 0.0, 0.0};
  return {v.x / n, v.y / n, v.z / n};
}

// 3x3 rotation matrix -> quaternion (x,y,z,w)
static std::array<double, 4> rotationMatrixToQuaternion(double R[3][3])
{
  double r00 = R[0][0], r01 = R[0][1], r02 = R[0][2];
  double r10 = R[1][0], r11 = R[1][1], r12 = R[1][2];
  double r20 = R[2][0], r21 = R[2][1], r22 = R[2][2];

  double trace = r00 + r11 + r22;
  double qw, qx, qy, qz;

  if (trace > 0.0) {
    double S = std::sqrt(trace + 1.0) * 2.0;
    qw = 0.25 * S;
    qx = (r21 - r12) / S;
    qy = (r02 - r20) / S;
    qz = (r10 - r01) / S;
  } else if ((r00 > r11) && (r00 > r22)) {
    double S = std::sqrt(1.0 + r00 - r11 - r22) * 2.0;
    qw = (r21 - r12) / S;
    qx = 0.25 * S;
    qy = (r01 + r10) / S;
    qz = (r02 + r20) / S;
  } else if (r11 > r22) {
    double S = std::sqrt(1.0 + r11 - r00 - r22) * 2.0;
    qw = (r02 - r20) / S;
    qx = (r01 + r10) / S;
    qy = 0.25 * S;
    qz = (r12 + r21) / S;
  } else {
    double S = std::sqrt(1.0 + r22 - r00 - r11) * 2.0;
    qw = (r10 - r01) / S;
    qx = (r02 + r20) / S;
    qy = (r12 + r21) / S;
    qz = 0.25 * S;
  }

  double n = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  return {qx / n, qy / n, qz / n, qw / n};
}

// ====== node class ======

class HemiMotionMoveItNode : public rclcpp::Node
{
public:
  HemiMotionMoveItNode()
  : Node("hemi_motion_moveit_node")
  {
    // --- Parameters ---
    move_group_name_ = declare_parameter<std::string>("move_group", "ur_manipulator");
    ik_frame_        = declare_parameter<std::string>("ik_frame", "tool0");
    target_frame_    = declare_parameter<std::string>("target_frame", "checkerboard_link");

    std::vector<double> center_default{0.0, 0.0, 0.30};
    auto center_param = declare_parameter<std::vector<double>>("hemi_center_xyz", center_default);
    if (center_param.size() == 3)
      hemi_center_xyz_ = center_param;
    else
      hemi_center_xyz_ = center_default;

    hemi_radius_         = declare_parameter<double>("hemi_radius", 0.33);
    hemi_num_latitudes_  = declare_parameter<int>("hemi_num_latitudes", 20);
    hemi_points_per_lat_ = declare_parameter<int>("hemi_points_per_lat", 50);
    axis_yaw_deg_        = declare_parameter<double>("axis_yaw_deg", 0.0);
    max_poses_           = declare_parameter<int>("max_poses", 80);

    RCLCPP_INFO(get_logger(),
                "MoveIt hemi: group='%s', ik_frame='%s', target_frame='%s'",
                move_group_name_.c_str(), ik_frame_.c_str(), target_frame_.c_str());

    // --- Publishers for visualization ---
    pose_array_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
        "hemi_waypoints_moveit", 1);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "hemi_waypoint_markers_moveit", 1);

    // --- Generate poses & visualize (no MoveGroup yet) ---
    generatePoses();
    publishVisualization();
  }

  // This MUST be called after make_shared(), so shared_from_this() is valid
  void init_move_group()
  {
    // Now shared_from_this() is safe
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), move_group_name_);
    move_group_->setEndEffectorLink(ik_frame_);
    move_group_->setPlanningTime(3.0);

    // Start motion thread
    motion_thread_ = std::thread([this]() { this->runMotion(); });
  }

  ~HemiMotionMoveItNode() override
  {
    if (motion_thread_.joinable())
      motion_thread_.join();
  }

private:
  void generatePoses()
  {
    poses_.clear();

    Vec3 center{hemi_center_xyz_[0], hemi_center_xyz_[1], hemi_center_xyz_[2]};
    Vec3 look_at{0.0, 0.0, 0.0};
    double yaw = axis_yaw_deg_ * M_PI / 180.0;

    for (int i = 0; i < hemi_num_latitudes_ / 2; ++i) {
      double phi = M_PI * (i + 0.5) / static_cast<double>(hemi_num_latitudes_);
      double theta_shift = (i % 2 == 0) ? 0.0
                                        : (M_PI / static_cast<double>(hemi_points_per_lat_));

      for (int j = 0; j < hemi_points_per_lat_; ++j) {
        double theta = 2.0 * M_PI * j / static_cast<double>(hemi_points_per_lat_) + theta_shift;

        double x  = hemi_radius_ * std::sin(phi) * std::cos(theta);
        double yv = hemi_radius_ * std::sin(phi) * std::sin(theta);
        double z  = hemi_radius_ * std::cos(phi);

        Vec3 pos = vadd(center, Vec3{x, yv, z});

        // z-axis points from pose to target
        Vec3 z_axis = vnorm(vsub(look_at, pos));

        // base x-axis rotated by yaw around Z
        Vec3 x_base{-1.0, 0.0, 0.0};
        Vec3 x_desired{
            x_base.x * std::cos(yaw) - x_base.y * std::sin(yaw),
            x_base.x * std::sin(yaw) + x_base.y * std::cos(yaw),
            0.0};

        Vec3 y_axis = vnorm(vcross(z_axis, x_desired));
        Vec3 x_axis = vnorm(vcross(y_axis, z_axis));

        double R[3][3] = {
            {x_axis.x, y_axis.x, z_axis.x},
            {x_axis.y, y_axis.y, z_axis.y},
            {x_axis.z, y_axis.z, z_axis.z}
        };
        auto q = rotationMatrixToQuaternion(R);

        geometry_msgs::msg::PoseStamped ps;
        ps.header.frame_id = target_frame_;
        ps.pose.position.x = pos.x;
        ps.pose.position.y = pos.y;
        ps.pose.position.z = pos.z;
        ps.pose.orientation.x = q[0];
        ps.pose.orientation.y = q[1];
        ps.pose.orientation.z = q[2];
        ps.pose.orientation.w = q[3];

        poses_.push_back(ps);
      }
    }

    if (static_cast<int>(poses_.size()) > max_poses_) {
      RCLCPP_INFO(get_logger(),
                  "Generated %zu poses, limiting to %d",
                  poses_.size(), max_poses_);
      poses_.resize(max_poses_);
    } else {
      RCLCPP_INFO(get_logger(), "Generated %zu poses", poses_.size());
    }
  }

  void publishVisualization()
  {
    // PoseArray
    geometry_msgs::msg::PoseArray pa;
    pa.header.frame_id = target_frame_;
    pa.header.stamp = now();
    for (const auto &ps : poses_)
      pa.poses.push_back(ps.pose);
    pose_array_pub_->publish(pa);

    // MarkerArray
    visualization_msgs::msg::MarkerArray ma;
    int id = 0;
    for (const auto &ps : poses_) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = target_frame_;
      m.header.stamp = now();
      m.ns = "hemi_moveit";
      m.id = id++;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose = ps.pose;
      m.scale.x = 0.05;
      m.scale.y = 0.05;
      m.scale.z = 0.05;
      m.color.a = 1.0;
      m.color.r = 0.0;
      m.color.g = 1.0;
      m.color.b = 0.0;
      ma.markers.push_back(m);
    }
    marker_pub_->publish(ma);
  }

void runMotion()
{
  // small delay to let MoveIt + TF settle
  rclcpp::sleep_for(std::chrono::seconds(2));

  // ===== NEW: move to desired initial joint configuration =====
  std::map<std::string, double> start_joints;
  start_joints["shoulder_pan_joint"]   = 0.0;      // your value
  start_joints["shoulder_lift_joint"]  = -1.5708;  // your value
  start_joints["elbow_joint"]          = -1.5708;  // your value
  start_joints["wrist_1_joint"]        = 0.0;      // your value
  start_joints["wrist_2_joint"]        = 1.5708;   // your value
  start_joints["wrist_3_joint"]        = 0.0;      // your value

  move_group_->setJointValueTarget(start_joints);

  {
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group_->plan(plan) ==
                    moveit::planning_interface::MoveItErrorCode::SUCCESS);
    if (!success) {
      RCLCPP_ERROR(get_logger(), "Failed to plan to initial joint pose, aborting hemisphere motion");
      return;
    }
    auto exec_result = move_group_->execute(plan);
    if (exec_result != moveit::planning_interface::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(), "Failed to execute initial joint pose, aborting hemisphere motion");
      return;
    }
  }

  // optional: make sure future plans start from the new current state
  move_group_->setStartStateToCurrentState();
  // ===== END NEW BLOCK =====

  for (size_t i = 0; i < poses_.size() && rclcpp::ok(); ++i) {
    const auto &ps = poses_[i];

    RCLCPP_INFO(get_logger(),
                "Planning to pose %zu/%zu", i + 1, poses_.size());

    move_group_->setPoseTarget(ps);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group_->plan(plan) ==
                    moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (!success) {
      RCLCPP_WARN(get_logger(), "Planning failed for pose %zu, skipping", i);
      continue;
    }

    auto exec_result = move_group_->execute(plan);
    if (exec_result != moveit::planning_interface::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "Execution failed at pose %zu, stopping", i);
      break;
    }

    move_group_->stop();
    move_group_->clearPoseTargets();

    rclcpp::sleep_for(std::chrono::milliseconds(100));
  }

  RCLCPP_INFO(get_logger(), "Hemisphere motion finished");
}


  // --- Params ---
  std::string move_group_name_;
  std::string ik_frame_;
  std::string target_frame_;
  std::vector<double> hemi_center_xyz_;
  double hemi_radius_;
  int hemi_num_latitudes_;
  int hemi_points_per_lat_;
  double axis_yaw_deg_;
  int max_poses_;

  // --- MoveIt ---
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  // --- Poses ---
  std::vector<geometry_msgs::msg::PoseStamped> poses_;

  // --- RViz ---
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_array_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  std::thread motion_thread_;
};

// ====== main ======

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HemiMotionMoveItNode>();
  node->init_move_group();  // important: after make_shared()
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

