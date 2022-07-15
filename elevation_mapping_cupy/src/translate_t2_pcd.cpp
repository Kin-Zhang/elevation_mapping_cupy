/*
 * Author: Kin 
 * only for elevation mapping usage ====> 128 - 16线对齐使用 更多见主分支下readme
 * 发布新world frame下的tf, 和odom
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

#include <ros/ros.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

struct pose
{
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};


static bool _debug_print = true;
static std::string _odom_topic="/odom", _new_frame_id="/new_velodyne", _lidar_topic, _output_lidar_topic;
static ros::Publisher t2pcd_pub, t2odom_pub, ignore_point_pub;

static ros::Time current_scan_time;
Eigen::Matrix4f T_2, tf_origin, tf_after_T2;

#define PI 3.14159265

static int _ignore_af = 0;
static float _ignore_df = 0;
static int _ignore_ar = 0;
static float _ignore_dr = 0;

double tan_rear, tan_front;

static void points_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
{

  // single point
  pcl::PointXYZI p;
  pcl::PointCloud<pcl::PointXYZI> tmp, scan;

  pcl::fromROSMsg(*input, tmp);
  for (pcl::PointCloud<pcl::PointXYZI>::const_iterator item = tmp.begin(); item != tmp.end(); item++)
  {
    p.x = (double)item->x;
    p.y = (double)item->y;
    p.z = (double)item->z;
    p.intensity = (double)item->intensity;
    double dis_p = p.x*p.x+p.y*p.y;
    double tan_p = abs(p.y/p.x);

    // point in rear
    if(p.x<=0 && _ignore_dr != 0 && dis_p<(_ignore_dr*_ignore_dr) && tan_p < tan_rear)
      continue;
    else if(p.x>0 && _ignore_df != 0 && dis_p<(_ignore_df*_ignore_df) && tan_p < tan_front)
      continue;
      
    scan.push_back(p);
  }
  sensor_msgs::PointCloud2 map_cloud;
  pcl::toROSMsg(scan, map_cloud);
  map_cloud.header = input->header;
  ignore_point_pub.publish(map_cloud);
}

static void odom_callback(const nav_msgs::Odometry::ConstPtr& input)
{
    current_scan_time = input->header.stamp;

    // =============> from input to translation matrix
    double tf_x, tf_y, tf_z, tf_roll, tf_pitch, tf_yaw;
    tf_x = input->pose.pose.position.x;
    tf_y = input->pose.pose.position.y;
    tf_z = input->pose.pose.position.z;
    Eigen::Translation3f tl_btol(tf_x, tf_y, tf_z);// tl: translation

    double input_roll, input_pitch, input_yaw;
    tf::Quaternion input_orientation;
    tf::quaternionMsgToTF(input->pose.pose.orientation, input_orientation);
    tf::Matrix3x3(input_orientation).getRPY(input_roll, input_pitch, input_yaw);
    Eigen::AngleAxisf rot_x_btol(tf_roll, Eigen::Vector3f::UnitX());  // rot: rotation
    Eigen::AngleAxisf rot_y_btol(tf_pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rot_z_btol(tf_yaw, Eigen::Vector3f::UnitZ());
    tf_origin = (tl_btol * rot_z_btol * rot_y_btol * rot_x_btol).matrix();

    tf_after_T2 = T_2 * tf_origin;

    tf::Matrix3x3 mat_l;

    mat_l.setValue(static_cast<double>(tf_after_T2(0, 0)), static_cast<double>(tf_after_T2(0, 1)),
                    static_cast<double>(tf_after_T2(0, 2)), static_cast<double>(tf_after_T2(1, 0)),
                    static_cast<double>(tf_after_T2(1, 1)), static_cast<double>(tf_after_T2(1, 2)),
                    static_cast<double>(tf_after_T2(2, 0)), static_cast<double>(tf_after_T2(2, 1)),
                    static_cast<double>(tf_after_T2(2, 2)));
    
    pose current_pose, tf2_pose;
    // Update current_pose.
    current_pose.x = tf_after_T2(0, 3);
    current_pose.y = tf_after_T2(1, 3);
    current_pose.z = tf_after_T2(2, 3);
    mat_l.getRPY(current_pose.roll, current_pose.pitch, current_pose.yaw, 1);

    // odom pub
    nav_msgs::Odometry new_odom;
    tf::Quaternion q;
    q.setRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);
    new_odom.header.frame_id = "new_map";
    new_odom.child_frame_id = _new_frame_id;
    new_odom.header.stamp = current_scan_time;

    new_odom.pose.pose.position.x = current_pose.x;
    new_odom.pose.pose.position.y = current_pose.y;
    new_odom.pose.pose.position.z = current_pose.z;
    new_odom.pose.pose.orientation.x = q.x();
    new_odom.pose.pose.orientation.y = q.y();
    new_odom.pose.pose.orientation.z = q.z();
    new_odom.pose.pose.orientation.w = q.w();
    t2odom_pub.publish(new_odom);

    // TF pub
    Eigen::Matrix4f T_2_inv = T_2.inverse(); //
    static tf::Matrix3x3 mat_tf2;
    static tf::Quaternion qt_2;
    static tf::TransformBroadcaster br;
    static tf::Transform transform;
    mat_tf2.setValue(static_cast<double>(T_2_inv(0, 0)), static_cast<double>(T_2_inv(0, 1)),
                    static_cast<double>(T_2_inv(0, 2)),  static_cast<double>(T_2_inv(1, 0)),
                    static_cast<double>(T_2_inv(1, 1)),  static_cast<double>(T_2_inv(1, 2)),
                    static_cast<double>(T_2_inv(2, 0)),  static_cast<double>(T_2_inv(2, 1)),
                    static_cast<double>(T_2_inv(2, 2)));
    // Update current_pose.
    tf2_pose.x = T_2_inv(0, 3) - 0.243;
    tf2_pose.y = T_2_inv(1, 3);
    tf2_pose.z = T_2_inv(2, 3) - 0.1338;
    mat_tf2.getRPY(tf2_pose.roll, tf2_pose.pitch, tf2_pose.yaw, 1);
    qt_2.setRPY(tf2_pose.roll, tf2_pose.pitch, tf2_pose.yaw);

    transform.setOrigin(tf::Vector3(tf2_pose.x, tf2_pose.y, tf2_pose.z));
    transform.setRotation(qt_2);
    br.sendTransform(tf::StampedTransform(transform, current_scan_time, "map", "new_map"));
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "translate_t2_pcd");

    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    private_nh.getParam("odom_topic", _odom_topic);
    private_nh.getParam("lidar_topic", _lidar_topic);
    private_nh.getParam("new_frame_id", _new_frame_id);
    private_nh.getParam("output_lidar_topic", _output_lidar_topic);

    private_nh.getParam("ignore_angle_front", _ignore_af);
    private_nh.getParam("ignore_dis_front", _ignore_df);
    private_nh.getParam("ignore_angle_rear", _ignore_ar);
    private_nh.getParam("ignore_dis_rear", _ignore_dr);
    tan_rear = tan(_ignore_ar/2 * PI / 180.0);
    tan_front = tan(_ignore_af/2 * PI / 180.0);

    std::string _output_odom_topic;
    private_nh.getParam("output_odom_topic", _output_odom_topic);
    private_nh.getParam("debug_print", _debug_print);
    std::vector<float> data;

    if(!private_nh.getParam("t2", data))
        ROS_ERROR("Failed to get parameter from server.");
    else
        std::cout<<"loading success"<<std::endl;

    // to matrix4f array ==> to Eigen Map
    float array[16];
    for (int i=0;i<16;i++)
        array[i]=data[i];
    T_2 = Eigen::Map<Eigen::Matrix4f> (array,4,4);

    ros::Subscriber odom_sub = nh.subscribe(_odom_topic, 1000, odom_callback);
    ros::Subscriber points_sub = nh.subscribe(_lidar_topic, 100000, points_callback);
    ignore_point_pub = nh.advertise<sensor_msgs::PointCloud2>(_output_lidar_topic, 1000);
    t2odom_pub = nh.advertise<nav_msgs::Odometry>(_output_odom_topic, 1000);
    ros::spin();
    return 0;
}
