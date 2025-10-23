
import numpy as np
import os
import torch
from typing import Tuple

from isaacgym import gymtorch, gymapi
from envs.base_task import BaseTask
from isaacgym.torch_utils import to_torch, quat_apply
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

class EarthSim(BaseTask):

    def __init__(self, cfg):
        self.get_env_params(cfg_env=cfg['env'])
        super().__init__(config=cfg)
        
        # ACTION NORMALIZATION PARAMETERS
        drag_scale = cfg['env']['drag_scale']
        depth_scale = cfg['env']['depth_scale']
        swing_scale = cfg['env']['swing_scale']
        bucket_angle_scale = cfg['env']['bucket_angle_scale']
        self.action_scale =  torch.tensor([drag_scale, depth_scale, swing_scale, bucket_angle_scale], device=self.device)

        # OBSERVATION NORMALIZATION PARAMETERS
        obs_min_limits = cfg['env']['obs_min_limits']
        obs_max_limits = cfg['env']['obs_max_limits']
        min_init_rock_dof_pos = cfg['env']['min_init_rock_dof_pos']
        max_init_rock_dof_pos = cfg['env']['max_init_rock_dof_pos']
        min_rock_pos_bias = cfg['env']['min_rock_pos_bias']
        max_rock_pos_bias = cfg['env']['max_rock_pos_bias']
        self.obs_min_limits = torch.tensor(obs_min_limits, device=self.device)
        self.obs_max_limits = torch.tensor(obs_max_limits, device=self.device)
        self.min_init_rock_dof_pos = torch.tensor(min_init_rock_dof_pos, device=self.device)
        self.max_init_rock_dof_pos = torch.tensor(max_init_rock_dof_pos, device=self.device)
        self.min_rock_pos_bias = torch.tensor(min_rock_pos_bias, device=self.device)
        self.max_rock_pos_bias = torch.tensor(max_rock_pos_bias, device=self.device)

        # ACQUIRE TENSOR API
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        # REFRESH ALL TENSOR API
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)


        # Root Tensor with shape -> [Num Envs , Num Actors , 13]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_environments, -1, 13)
        self.bucket_root_state_tensor = self.root_state_tensor[:, self.buckets[0]]
        self.rock_root_state_tensor = self.root_state_tensor[:, self.rocks[0]]
        self.sand_root_state_tensor = self.root_state_tensor[:, 7:]
        # Global index NEEDED to -> reset root actor states
        actor_num = self.gym.get_sim_actor_count(self.sim)
        self.global_index = torch.arange(actor_num, dtype=torch.int32, device=self.device).view(self.num_environments, -1)


        # DOF Tensor with shape -> [num_dofs , 2]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.num_dof_per_env = self.gym.get_sim_dof_count(self.sim) // self.num_environments
        # BUCKET DOF STATES
        self.bucket_dof_state = self.dof_state.view(self.num_environments, -1, 2)[:, :self.num_bucket_dofs]
        self.bucket_dof_pos = self.bucket_dof_state[..., 0]
        self.bucket_dof_vel = self.bucket_dof_state[..., 1]
        # SAVE BUCKET DOF POSITION TARGETS
        self.bucket_dof_targets = torch.zeros((self.num_environments, self.num_dof_per_env), dtype=torch.float, device=self.device)
        # ROCK DOF STATES
        self.rock_dof_state = self.dof_state.view(self.num_environments, -1, 2)[:, self.num_bucket_dofs:]
        self.rock_dof_pos = self.rock_dof_state[..., 0]
        self.rock_dof_vel = self.rock_dof_state[..., 1]

        # RIGID BODY STATE Tensor with shape -> [num_rigid_bodies , 13]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_environments, -1, 13)
        self.bucket_pos = self.rigid_body_states[:, self.bucket_handle][:, 0:3]
        self.bucket_quat = self.rigid_body_states[:, self.bucket_handle][:, 3:7]
        self.bucket_fixpoint_pos = self.rigid_body_states[:, self.bucket_fixpoint_handle][:, 0:3]
        self.rock_pos = self.rigid_body_states[:, self.rock_handle][:, 0:3]
        self.rock_quat = self.rigid_body_states[:, self.rock_handle][:, 3:7]
        
        # NET CONTACT FORCES Tensor with Shape: [num_envs, num_bodies, 3]
        self.contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_environments, -1, 3)
        self.bucket_net_force_magnitude = torch.zeros((self.num_environments, 1), dtype=torch.float, device=self.device)

        # ROCK_INIT_DOF_POS | ROCK_POS_BIAS
        self.rock_pos_bias = torch.zeros((self.num_environments, 3), dtype=torch.float, device=self.device)

        self.reset_idx(torch.arange(self.num_environments, device=self.device))

        if self.graphics_device_id != -1:
            cam_pos = gymapi.Vec3(4, 3, 0)
            cam_target = gymapi.Vec3(-4, -3, 2)
            num_per_row = int(np.sqrt(self.num_environments))
            middle_env = self.envs[self.num_environments // 2 + num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_target, cam_pos)


    def get_env_params(self, cfg_env):
        self.ground_bottom_height = cfg_env['ground_bottom_height']
        self.ground_top_height_for_field = cfg_env['ground_top_height_for_field']
        self.wall_radius = cfg_env['wall_radius']
        self.wall_width = cfg_env['wall_width']
        self.height_limit = cfg_env['height_limit']

        self.bucket_default_pos = cfg_env['bucket_default_pos']  # X, Y, Z
        self.rock_default_pos = cfg_env['rock_default_pos']      # X, Y, Z

        self.rock_density = cfg_env['rock_density']
        self.sand_density = cfg_env['sand_density']
        self.sand_size = cfg_env['sand_size']
        sand_num_x = int(self.wall_radius // self.sand_size - 1)
        sand_num_y = sand_num_x
        sand_num_z = int((self.ground_top_height_for_field - self.ground_bottom_height) // self.sand_size )
        self.sand_num_xyz = [sand_num_x, sand_num_y, sand_num_z]
        self.num_props = sand_num_x * sand_num_y * sand_num_z
        
        self.asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg_env.get("assetRoot", "../assets/earth_sim"))
        self.asset_bucket_filename = cfg_env.get('assetFileName_Bucket', 'bucket/bucket_isaacgym.urdf')
        self.asset_rock_filename = cfg_env.get('assetFileName_Rock', 'rock_6dof.urdf')

        self.cam_width = cfg_env['cam_width']
        self.cam_height = cfg_env['cam_height']
        self.cam_pos = cfg_env['cam_pos']
        self.cam_target = cfg_env['cam_target']
        self.cam_fov = cfg_env['cam_fov']
        self.far_plane = cfg_env['far_plane']

    def create_sim(self):
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_environments, self.env_spacing, int(np.sqrt(self.num_environments)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        self.envs = []
        self.buckets = []
        self.rocks = []
        self.default_sand_states = []
        self.camera_handles = []
        self.cam_tensors = []
        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.add_bucket(env_ptr, -1, collision=-1)
            self.add_rock(env_ptr, -1, collision=-1)
            self.add_bottom(env_ptr, -1, collision=-1)
            self.add_wall(env_ptr, -1, collision=-1)
            self.add_sand(env_ptr, -1, collision=-1)
            self.add_camera_sensor(env_ptr)
            self.envs.append(env_ptr)

        self.bucket_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.buckets[0], "bucket_link")
        self.bucket_fixpoint_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.buckets[0], "fixpoint_link")
        self.rock_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.rocks[0], "rock_link")
        self.default_sand_states = to_torch(self.default_sand_states, device=self.device, dtype=torch.float).view(self.num_environments, self.num_props, 13)

        actor_prop = self.gym.get_actor_dof_properties(self.envs[0], self.buckets[0])
        self.swing_dof_handle = self.gym.find_actor_dof_handle(self.envs[0], self.buckets[0], "swing_joint_r")
        self.bucket_dof_handle = self.gym.find_actor_dof_handle(self.envs[0], self.buckets[0], "bucket_joint_r")
        # Manually set dof limits for BUCKET
        self.swing_lower = actor_prop["lower"][self.swing_dof_handle]
        self.swing_upper = actor_prop["upper"][self.swing_dof_handle]
        self.bucket_lower = actor_prop["lower"][self.bucket_dof_handle]
        self.bucket_upper = actor_prop["upper"][self.bucket_dof_handle]

    def add_bucket(self, env_ptr, env_id, collision=0):
        asset_path = os.path.join(self.asset_root, self.asset_bucket_filename)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 300000  # 300000
        asset_options.vhacd_params.max_convex_hulls = 20  # 10
        asset_options.vhacd_params.max_num_vertices_per_ch = 64  # 64
        asset_options.override_com = True
        asset_options.override_inertia = True

        bucket_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_bucket_bodies = self.gym.get_asset_rigid_body_count(bucket_asset)
        self.num_bucket_dofs = self.gym.get_asset_dof_count(bucket_asset)

        bucket_dof_props = self.gym.get_asset_dof_properties(bucket_asset)
        
        bucket_dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
        bucket_dof_props['hasLimits'].fill(True)
        bucket_dof_props['stiffness'].fill(2500.)
        bucket_dof_props['damping'].fill(400.)
        # bucket_dof_props['friction'].fill(100.)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(self.bucket_default_pos[0], self.bucket_default_pos[1], self.bucket_default_pos[2])

        bucket_actor = self.gym.create_actor(env_ptr, bucket_asset, pose, "bucket", env_id, collision, 0)
        self.gym.set_actor_dof_properties(env_ptr, bucket_actor, bucket_dof_props)

        self.buckets.append(bucket_actor)

    def add_rock(self, env_ptr, env_id, name_id=0, collision=0):
        asset_path = os.path.join(self.asset_root, self.asset_rock_filename)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.linear_damping = 1.0
        asset_options.angular_damping = 1.0
        asset_options.density = self.rock_density
        rock_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_rock_bodies = self.gym.get_asset_rigid_body_count(rock_asset)
        self.num_rock_dofs = self.gym.get_asset_dof_count(rock_asset)

        rock_dof_props = self.gym.get_asset_dof_properties(rock_asset)
        self.rock_dof_lower_limits = []
        self.rock_dof_upper_limits = []
        for i in range(self.num_rock_dofs):
            rock_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT  # gymapi.DOF_MODE_EFFORT, gymapi.DOF_MODE_NONE, gymapi.DOF_MODE_POS
            rock_dof_props['stiffness'][i] = 0.01
            rock_dof_props['damping'][i] = 0.
            rock_dof_props['friction'][i] = 0.
            self.rock_dof_lower_limits.append(rock_dof_props['lower'][i])
            self.rock_dof_upper_limits.append(rock_dof_props['upper'][i])

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(self.rock_default_pos[0], self.rock_default_pos[1], self.rock_default_pos[2])
        # pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))

        rock_actor = self.gym.create_actor(env_ptr, rock_asset, pose, "rock", env_id, collision, 0)
        self.gym.set_actor_dof_properties(env_ptr, rock_actor, rock_dof_props)
        self.gym.set_rigid_body_color(env_ptr, rock_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)  # MESH_VISUAL_AND_COLLISION,MESH_NONE

        self.rocks.append(rock_actor)

    def add_wall(self, env_ptr, env_id, collision=0):
        wall_thickness = self.ground_top_height_for_field - self.ground_bottom_height
        table_dims = gymapi.Vec3(self.wall_radius + self.wall_width * 2., self.wall_width, wall_thickness)
        color = gymapi.Vec3(0.6392156862745098, 0.40784313725490196, 0.25098039215686274)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        side1_pose = gymapi.Transform()
        side1_pose.p = gymapi.Vec3(0, (self.wall_radius + self.wall_width) / 2., self.ground_bottom_height + 0.5 * wall_thickness)
        side1_pose.r = gymapi.Quat(0, 0, 0, 1)
        field_actor = self.gym.create_actor(env_ptr, table_asset, side1_pose, "wall1", env_id, collision, 0)
        self.gym.set_rigid_body_color(env_ptr, field_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)  # MESH_VISUAL_AND_COLLISION,MESH_NONE

        side2_pose = gymapi.Transform()
        side2_pose.p = gymapi.Vec3(0, -(self.wall_radius + self.wall_width) / 2., self.ground_bottom_height + 0.5 * wall_thickness)
        side2_pose.r = gymapi.Quat(0, 0, 0, 1)
        field_actor = self.gym.create_actor(env_ptr, table_asset, side2_pose, "wall2", env_id, collision, 0)
        self.gym.set_rigid_body_color(env_ptr, field_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)  # MESH_VISUAL_AND_COLLISION,MESH_NONE

        side3_pose = gymapi.Transform()
        side3_pose.p = gymapi.Vec3((self.wall_radius + self.wall_width) / 2., 0, self.ground_bottom_height + 0.5 * wall_thickness)
        side3_pose.r = gymapi.Quat(0, 0, 1, 1)
        field_actor = self.gym.create_actor(env_ptr, table_asset, side3_pose, "wall3", env_id, collision, 0)
        self.gym.set_rigid_body_color(env_ptr, field_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)  # MESH_VISUAL_AND_COLLISION,MESH_NONE

        side4_pose = gymapi.Transform()
        side4_pose.p = gymapi.Vec3(-(self.wall_radius + self.wall_width) / 2., 0, self.ground_bottom_height + 0.5 * wall_thickness)
        side4_pose.r = gymapi.Quat(0, 0, 1, 1)
        field_actor = self.gym.create_actor(env_ptr, table_asset, side4_pose, "wall4", env_id, collision, 0)
        self.gym.set_rigid_body_color(env_ptr, field_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)  # MESH_VISUAL_AND_COLLISION,MESH_NONE

    def add_bottom(self, env_ptr, env_id, collision=0):
        table_dims = gymapi.Vec3(self.wall_radius + self.wall_width * 2., self.wall_radius + self.wall_width * 2., self.ground_bottom_height)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, self.ground_bottom_height / 2.)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        color = gymapi.Vec3(0.6392156862745098, 0.40784313725490196, 0.25098039215686274)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        bottom_actor = self.gym.create_actor(env_ptr, table_asset, pose, "bottom", env_id, collision, 0)
        self.gym.set_rigid_body_color(env_ptr, bottom_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)  # MESH_VISUAL_AND_COLLISION,MESH_NONE

    def add_sand(self, env_ptr, env_id, collision=0):
        sand_dims = gymapi.Vec3(self.sand_size, self.sand_size, self.sand_size)
        color = gymapi.Vec3(0.6392156862745098, 0.40784313725490196, 0.25098039215686274)
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.linear_damping = 10.
        asset_options.angular_damping = 10.  # 0.5
        asset_options.armature = 1000.  # 0
        asset_options.max_linear_velocity = 1.  # 1000
        asset_options.max_angular_velocity = 1.  # 64
        asset_options.density = self.sand_density
        sand_asset = self.gym.create_box(self.sim, sand_dims.x, sand_dims.y, sand_dims.z, asset_options)

        ball_num_xyz = np.array(self.sand_num_xyz)
        center_height = (self.ground_top_height_for_field - self.ground_bottom_height) * 5 / 8 + self.ground_bottom_height
        center_position = np.array([0., 0., center_height])

        for x in range(ball_num_xyz[0]):
            for y in range(ball_num_xyz[1]):
                for z in range(ball_num_xyz[2]):
                    step = np.array(
                        [
                            (x - ball_num_xyz[0] / 2.) * self.sand_size * 1.13,
                            (y - ball_num_xyz[1] / 2.) * self.sand_size * 1.13,
                            z * self.sand_size * 1.13,
                        ]
                    )
                    new = center_position + step
                    pose = gymapi.Transform()
                    pose.p = gymapi.Vec3(*new)
                    pose.r = gymapi.Quat(0, 0, 0, 1)
                    sand_actor = self.gym.create_actor(env_ptr, sand_asset, pose, "sand", env_id, collision, 0)
                    self.gym.set_rigid_body_color(env_ptr, sand_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)  # MESH_VISUAL_AND_COLLISION,MESH_NONE

                    self.default_sand_states.append([pose.p.x, pose.p.y, pose.p.z,
                                                     pose.r.x, pose.r.y, pose.r.z, pose.r.w,
                                                     0, 0, 0, 0, 0, 0])

    def add_camera_sensor(self, env_ptr):
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cam_width
        camera_props.height = self.cam_height
        camera_props.enable_tensors = True # MUST be True to get data
        camera_props.horizontal_fov = self.cam_fov
        camera_props.far_plane = self.far_plane 

        cam_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
        
        cam_pos = gymapi.Vec3(*self.cam_pos)
        cam_target = gymapi.Vec3(*self.cam_target)
        self.gym.set_camera_location(cam_handle, env_ptr, cam_pos, cam_target)
        
        self.camera_handles.append(cam_handle)

        # DEPTH IMAGE RENDERED TENSOR
        cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_DEPTH)
        torch_cam_tensor  = gymtorch.wrap_tensor(cam_tensor)
        self.cam_tensors.append(torch_cam_tensor)   # Shape List(num_environments) (height, width)

    def compute_reward(self):
        # self.rew_buf[:], self.reset_buf[:] = dummy_reward(self.reset_buf, self.progress_buf, self.max_episode_length)
        
        bucket_dof = self.bucket_dof_pos[:, 4]
        self.rew_buf[:], self.reset_buf[:] = lift_reward_v2(self.reset_buf,
                                                            self.progress_buf,
                                                            self.max_episode_length,
                                                            self.bucket_fixpoint_pos,
                                                            self.bucket_pos,
                                                            bucket_dof, 
                                                            self.rock_pos,
                                                            self.bucket_quat,
                                                            self.height_limit)


    def compute_observations(self):
        self._refresh()
        self.populate_observations()
        if self.num_states > 0: self.populate_states()

        return self.obs_buf

    def populate_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_environments, device=self.device)

        rotated_bias = quat_apply(self.rock_quat, self.rock_pos_bias)
        biased_rock_pos = self.rock_pos + rotated_bias
        # biased_rock_pos = self.rock_pos

        relative_pos = self.bucket_fixpoint_pos - biased_rock_pos
        abosulte_rock_height = 0.6 - biased_rock_pos[:,2]

        swing_dof = self.bucket_dof_pos[:, 3].unsqueeze(1)
        bucket_dof = self.bucket_dof_pos[:, 4].unsqueeze(1)

        self.gym.start_access_image_tensors(self.sim)
        
        if self.viewer:
            depth_image_env0 = self.cam_tensors[0]
            # import ipdb
            # ipdb.set_trace()
            
            depth_np = depth_image_env0.cpu().numpy()

            finite_depth = np.clip(depth_np, -0.52, 0.0)
            # print(f"Depth min: {np.min(finite_depth)}, max: {np.max(finite_depth)}")

            plt.imshow(finite_depth, cmap='viridis')
            plt.colorbar(label='Depth (Negative View Distance)')
            plt.title("Depth Map (Env 0)")
            
            plt.pause(0.001) 
            plt.clf()
        self.gym.end_access_image_tensors(self.sim)

        obs = torch.cat((relative_pos, swing_dof, bucket_dof, self.bucket_pos[:, :2], abosulte_rock_height.unsqueeze(1)), dim=1)
        self.obs_buf[env_ids] = obs

        return self.obs_buf
    
    def populate_states(self, env_ids=None):
        pass

    def get_obs_state_dict(self, env_ids=None):
        self.populate_observations(env_ids)
        self.obs_dict['obs'] = self.get_observation()

        if self.num_states > 0:
            self.populate_states(env_ids)
            self.obs_dict['states'] = self.get_state()

        return self.obs_dict

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_jacobian_tensors(self.sim)
        # self.gym.refresh_mass_matrix_tensors(self.sim)
        # self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def reset(self, force_random=None):
        self.reset_idx(torch.arange(self.num_environments, device=self.device))

    def reset_idx(self, env_ids):

        num_to_reset = len(env_ids)
        rand_floats_dof = torch.rand((num_to_reset, 6), device=self.device)
        rand_floats_bias = torch.rand((num_to_reset, 3), device=self.device)
        new_init_rock_dof_pos = rand_floats_dof * (self.max_init_rock_dof_pos - self.min_init_rock_dof_pos) + self.min_init_rock_dof_pos
        new_rock_pos_bias = rand_floats_bias * (self.max_rock_pos_bias - self.min_rock_pos_bias) + self.min_rock_pos_bias
        
        self.rock_pos_bias[env_ids] = new_rock_pos_bias

        self.sand_root_state_tensor[env_ids] = self.default_sand_states[env_ids]
        root_indices = self.global_index[env_ids, 7:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(root_indices),
                                                     len(root_indices))

        self.bucket_dof_pos[env_ids] = torch.zeros(self.num_bucket_dofs, device=self.device)
        self.bucket_dof_vel[env_ids] = torch.zeros(self.num_bucket_dofs, device=self.device)
        self.rock_dof_pos[env_ids] = new_init_rock_dof_pos
        self.rock_dof_vel[env_ids] = torch.zeros(self.num_rock_dofs, device=self.device)
        selected_rows = self.global_index[env_ids]
        column_indices = [self.buckets[0], self.rocks[0]]
        root_indices_ = selected_rows[:, column_indices].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(root_indices_),
                                              len(root_indices_))

        
        self.bucket_dof_targets[env_ids, :self.num_bucket_dofs] = self.bucket_dof_pos[env_ids]
        
        
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.task_achievement_buf[env_ids] = False

    def pre_physics_step(self, action): # ACTION TENSOR: [DRAG , DEPTH , SWING , BUCKET]
        scaled_action = action * self.action_scale
        calc_action = self.calc_next_setpoint(scaled_action)
        self.bucket_dof_targets[:, :self.num_bucket_dofs] += calc_action
        # self.bucket_dof_targets[:, :self.num_bucket_dofs] = self.bucket_dof_pos[:] + calc_action

        self.bucket_dof_targets[:, self.swing_dof_handle] = torch.clamp(self.bucket_dof_targets[:, self.swing_dof_handle], min=self.swing_lower, max=self.swing_upper)
        self.bucket_dof_targets[:, self.bucket_dof_handle] = torch.clamp(self.bucket_dof_targets[:, self.bucket_dof_handle], min=self.bucket_lower, max=self.bucket_upper)
        
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.bucket_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0: self.reset_idx(env_ids)

        if self.viewer:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            # self.draw_debug_lines()
            # self.draw_debug_com()
            self.gym.draw_viewer(self.viewer, self.sim)
        else:
             self.gym.render_all_camera_sensors(self.sim)

        self.compute_observations()
        self.compute_reward()

    def calc_next_setpoint(self, action):
        drag = action[:, 0].unsqueeze(1)        # Shape: [num_envs, 1]
        remaining_actions = action[:, 1:]       # Shape: [num_envs, 3]

        swing_angle = self.bucket_dof_pos[:, 3]
        cos_swing = torch.cos(swing_angle)
        sin_swing = torch.sin(swing_angle)
        leading_vector = torch.stack((-sin_swing, cos_swing), dim=1)
        
        scaled_projection = - drag * leading_vector[:, :2] # Shape: [num_envs, 2]
        output_tensor = torch.cat((scaled_projection, remaining_actions), dim=1)

        return output_tensor
    
    def draw_debug_com(self):
        self.gym.clear_lines(self.viewer)
        bucket_pos_np = self.bucket_pos.cpu().numpy()
        rock_pos_np = self.rock_pos.cpu().numpy()

        rotated_bias = quat_apply(self.rock_quat, self.rock_pos_bias)
        rock_pos_biased_np = (self.rock_pos + rotated_bias).cpu().numpy()

        for i in range(self.num_environments):
            line_vertices = np.array([
                bucket_pos_np[i], rock_pos_np[i],
                bucket_pos_np[i], rock_pos_biased_np[i]
            ], dtype=np.float32)
            
            colors = np.array([
                [1.0, 0.0, 0.0],  # Red: for the line to the rock
                [0.0, 0.0, 1.0]   # Blue: for the y-axis projection line
            ], dtype=np.float32)

            self.gym.add_lines(self.viewer,
                               self.envs[i],
                               2, # Number of lines to draw
                               line_vertices,
                               colors)

    def draw_debug_lines(self):
        self.gym.clear_lines(self.viewer)
        bucket_pos_np = self.bucket_pos.cpu().numpy()

        rock_pos_xy = self.rock_pos.clone()
        rock_pos_xy[:,2] = self.bucket_pos[:,2]
        rock_pos_np = rock_pos_xy.cpu().numpy()

        y_axis_batched  = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float).expand(self.num_environments, -1)   # Shape is now [25, 3]
        bucket_y_projection = quat_apply(self.bucket_quat, y_axis_batched)
        bucket_y_projection[:, 2] = 0.0
        bucket_y_proy = self.bucket_pos - 1.0 * bucket_y_projection
        bucket_y_proy_np = bucket_y_proy.cpu().numpy()

        for i in range(self.num_environments):
            line_vertices = np.array([
                bucket_pos_np[i], rock_pos_np[i],
                bucket_pos_np[i], bucket_y_proy_np[i]
            ], dtype=np.float32)
            
            colors = np.array([
                [1.0, 0.0, 0.0],  # Red: for the line to the rock
                [0.0, 0.0, 1.0]   # Blue: for the y-axis projection line
            ], dtype=np.float32)

            self.gym.add_lines(self.viewer,
                               self.envs[i],
                               2, # Number of lines to draw
                               line_vertices,
                               colors)

    def check_collisions(self):
        rock_forces = self.contact_forces[:, self.rock_handle]
        self.bucket_net_force_magnitude[:] = torch.linalg.norm(rock_forces, dim=1)
        
        # self.is_colliding = force_magnitudes > 0.1

"""
JIT FUNCTION
"""

@torch.jit.script
def lift_reward_v2(reset_buf, progress_buf, max_episode_length, bucket_fix_pos, bucket_pos, bucket_dof, rock_pos, bucket_quat, height_limit):
    # type: (torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float) -> Tuple[torch.Tensor, torch.Tensor]
    
    max_xyz_dist = 0.6
    max_height_diff = 0.6
    pos_weight    = 0.15
    height_weight = 0.7
    angle_weight  = 0.15

    bucket_rock_xyz_distance = torch.linalg.norm(bucket_fix_pos[:, :3] - rock_pos[:, :3], dim=1)
    normalized_xyz_dist_penalty = torch.clamp(bucket_rock_xyz_distance / max_xyz_dist, 0, 1.0)
    is_below_rock = bucket_fix_pos[:, 2] < rock_pos[:, 2]
    is_upright_side = bucket_dof < -0.1
    is_bucket_close = bucket_rock_xyz_distance < 0.08
    is_good_condition = is_below_rock & is_upright_side & is_bucket_close
    pos_penalty = torch.where(is_good_condition, torch.zeros_like(normalized_xyz_dist_penalty), normalized_xyz_dist_penalty)

    height_difference = torch.abs(height_limit - rock_pos[:, 2])
    normalized_height_penalty = torch.clamp(height_difference / max_height_diff, 0, 1.0)

    # --- Component 3: Angle Penalty ---
    rock_to_bucket_xy = (rock_pos - bucket_pos)[:, :2]
    # Calculate the bucket's y-axis vector and its XY projection
    q_xyz = bucket_quat[:, :3]
    q_w = bucket_quat[:, 3].unsqueeze(-1)
    y_axis_base = torch.tensor([0.0, -1.0, 0.0], device=bucket_pos.device, dtype=torch.float)
    y_axis_batched = y_axis_base.expand(q_xyz.shape[0], -1)
    t = 2.0 * torch.cross(q_xyz, y_axis_batched, dim=-1)
    bucket_y_vec_3d = y_axis_batched + q_w * t + torch.cross(q_xyz, t, dim=-1)
    bucket_y_vec_xy = bucket_y_vec_3d[:, :2]
    # Calculate the cosine of the angle between the two XY vectors
    dot_product = torch.sum(rock_to_bucket_xy * bucket_y_vec_xy, dim=1)
    norm_product = torch.linalg.norm(rock_to_bucket_xy, dim=1) * torch.linalg.norm(bucket_y_vec_xy, dim=1)
    cos_angle = dot_product / (norm_product + 1e-8)

    normalized_angle_penalty = (1.0 - cos_angle) / 2.0
    is_alignment_close = bucket_rock_xyz_distance < 0.1
    angle_penalty = torch.where(is_alignment_close, torch.zeros_like(normalized_angle_penalty), normalized_angle_penalty)


    total_penalty = (pos_weight * pos_penalty +
                     height_weight * normalized_height_penalty +
                     angle_weight * angle_penalty)
    reward = -total_penalty

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return reward, reset


@torch.jit.script
def dummy_reward(reset_buf, progress_buf, max_episode_length):
    # type: (torch.Tensor, torch.Tensor, float) -> Tuple[torch.Tensor, torch.Tensor]
    reward = torch.ones_like(reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    return reward, reset
