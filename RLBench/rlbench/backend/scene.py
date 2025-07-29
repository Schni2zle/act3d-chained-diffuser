from typing import Any,List, Callable, Dict, Optional
from functools import partial

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType
from pyrep.errors import ConfigurationPathError
from pyrep.objects import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.backend.exceptions import (NoViewpointsError,
    WaypointError, BoundaryError, NoWaypointsError, DemoError)
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.backend.utils import rgb_handles_to_mask
from rlbench.demo import Demo
from rlbench.noise_model import NoiseModel
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.backend.const import TABLE_COORD
from rlbench.backend.waypoints import Point, PredefinedPath, Waypoint
import  rlbench.backend.utils as utils

STEPS_BEFORE_EPISODE_START = 10


class Scene(object):
    """Controls what is currently in the vrep scene. This is used for making
    sure that the tasks are easily reachable. This may be just replaced by
    environment. Responsible for moving all the objects. """

    def __init__(self,
                 pyrep: PyRep,
                 robot: Robot,
                 obs_config: ObservationConfig = ObservationConfig(),
                 robot_setup: str = 'panda'):
        self.pyrep = pyrep
        self.robot = robot
        self.robot_setup = robot_setup
        self.task = None
        self._obs_config = obs_config
        self._initial_task_state = None
        self._start_arm_joint_pos = robot.arm.get_joint_positions()
        self._starting_gripper_joint_pos = robot.gripper.get_joint_positions()
        self._workspace = Shape('workspace')
        self._workspace_boundary = SpawnBoundary([self._workspace])
        self._cam_over_shoulder_left = VisionSensor('cam_over_shoulder_left')
        self._cam_over_shoulder_right = VisionSensor('cam_over_shoulder_right')
        self._cam_overhead = VisionSensor('cam_overhead')
        self._cam_wrist = VisionSensor('cam_wrist')
        self._cam_active = VisionSensor('cam_active')
        self._cam_front = VisionSensor('cam_front')
        self._cam_over_shoulder_left_mask = VisionSensor(
            'cam_over_shoulder_left_mask')
        self._cam_over_shoulder_right_mask = VisionSensor(
            'cam_over_shoulder_right_mask')
        self._cam_overhead_mask = VisionSensor('cam_overhead_mask')
        self._cam_wrist_mask = VisionSensor('cam_wrist_mask')
        self._cam_active_mask = VisionSensor('cam_active_mask')
        self._cam_front_mask = VisionSensor('cam_front_mask')
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

        self._initial_robot_state = (robot.arm.get_configuration_tree(),
                                     robot.gripper.get_configuration_tree())

        # Set camera properties from observation config
        self._set_camera_properties()

        x, y, z = self._workspace.get_position()
        minx, maxx, miny, maxy, _, _ = self._workspace.get_bounding_box()
        self._workspace_minx = x - np.fabs(minx) - 0.2
        self._workspace_maxx = x + maxx + 0.2
        self._workspace_miny = y - np.fabs(miny) - 0.2
        self._workspace_maxy = y + maxy + 0.2
        self._workspace_minz = z
        self._workspace_maxz = z + 1.0  # 1M above workspace

        self.target_workspace_check = Dummy.create()
        self._step_callback = None

        self._robot_shapes = self.robot.arm.get_objects_in_tree(
            object_type=ObjectType.SHAPE)

        self._execute_demo_joint_position_action = None

        self._human = None

        self._transition_index = None

        self._processing_stage = None
        # 
        self._debug_point = Dummy.create(size=0.05)
        
        self._active_cam_placeholder = Dummy('active_cam_placeholder')


    def show_virtual_cam(self,pose:np.ndarray):

        shape = pose.shape[-1]
        if shape == 7:
            self._active_cam_placeholder.set_pose(pose=pose)
        elif shape == 8:
            self._active_cam_placeholder.set_position(pose[:7])
        elif shape == 6:
            self._active_cam_placeholder.set_position(pose[:3])
            self._active_cam_placeholder.set_orientation(pose[3:])
        elif shape == 3:
            self._active_cam_placeholder.set_position(pose[:3])
        else:
            raise TypeError('pose error')
        self.pyrep.step()

    def show_point(self,pose:np.ndarray):

        shape = pose.shape[-1]
        if shape == 7:
            self._debug_point.set_pose(pose=pose)
        elif shape == 8:
            self._debug_point.set_position(pose[:7])
        elif shape == 6:
            self._debug_point.set_position(pose[:3])
            self._debug_point.set_orientation(pose[3:])
        elif shape == 3:
            self._debug_point.set_position(pose[:3])
        else:
            raise TypeError('pose error')
        self.pyrep.step()

    def load(self, task: Task) -> None:
        """Loads the task and positions at the centre of the workspace.

        :param task: The task to load in the scene.
        """
        task.load()  # Load the task in to the scene

        # Set at the centre of the workspace
        task.get_base().set_position(self._workspace.get_position())
        self._initial_task_state = task.get_state()
        self.task = task
        self._initial_task_pose = task.boundary_root().get_orientation()
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

    def unload(self) -> None:
        """Clears the scene. i.e. removes all tasks. """
        if self.task is not None:
            self.robot.gripper.release()
            if self._has_init_task:
                self.task.cleanup_()
            self.task.unload()
        self.task = None
        self._variation_index = 0

    def init_task(self) -> None:
        self.task.init_task()
        self._initial_task_state = self.task.get_state()
        self._has_init_task = True
        self._variation_index = 0

    def init_episode(self, index: int, randomly_place: bool=True,
                     max_attempts: int = 5) -> List[str]:
        """Calls the task init_episode and puts randomly in the workspace.
        """

        self._variation_index = index

        if not self._has_init_task:
            self.init_task()

        # Try a few times to init and place in the workspace
        attempts = 0
        descriptions = None
        while attempts < max_attempts:
            descriptions = self.task.init_episode(index)
            try:
                if (randomly_place and
                        not self.task.is_static_workspace()):
                    self._place_task()
                    if self.robot.arm.check_arm_collision():
                        raise BoundaryError()
                self.task.validate()
                break
            except (BoundaryError, WaypointError) as e:
                self.task.cleanup_()
                self.task.restore_state(self._initial_task_state)
                attempts += 1
                if attempts >= max_attempts:
                    raise e

        # Let objects come to rest
        [self.pyrep.step() for _ in range(STEPS_BEFORE_EPISODE_START)]
        self._has_init_episode = True
        return descriptions

    def reset(self) -> None:
        """Resets the joint angles. """
        self.robot.gripper.release()

        arm, gripper = self._initial_robot_state
        self.pyrep.set_configuration_tree(arm)
        self.pyrep.set_configuration_tree(gripper)
        self.robot.arm.set_joint_positions(self._start_arm_joint_pos, disable_dynamics=True)
        self.robot.arm.set_joint_target_velocities(
            [0] * len(self.robot.arm.joints))
        self.robot.gripper.set_joint_positions(
            self._starting_gripper_joint_pos, disable_dynamics=True)
        self.robot.gripper.set_joint_target_velocities(
            [0] * len(self.robot.gripper.joints))

        if self.task is not None and self._has_init_task:
            self.task.cleanup_()
            self.task.restore_state(self._initial_task_state)
        self.task.set_initial_objects_in_scene()

    def get_observation(self) -> Observation:
        tip = self.robot.arm.get_tip()

        joint_forces = None
        if self._obs_config.joint_forces:
            fs = self.robot.arm.get_joint_forces()
            vels = self.robot.arm.get_joint_target_velocities()
            joint_forces = self._obs_config.joint_forces_noise.apply(
                np.array([-f if v < 0 else f for f, v in zip(fs, vels)]))

        ee_forces_flat = None
        if self._obs_config.gripper_touch_forces:
            ee_forces = self.robot.gripper.get_touch_sensor_forces()
            ee_forces_flat = []
            for eef in ee_forces:
                ee_forces_flat.extend(eef)
            ee_forces_flat = np.array(ee_forces_flat)

        lsc_ob = self._obs_config.left_shoulder_camera
        rsc_ob = self._obs_config.right_shoulder_camera
        oc_ob = self._obs_config.overhead_camera
        wc_ob = self._obs_config.wrist_camera
        atc_ob = self._obs_config.active_camera
        fc_ob = self._obs_config.front_camera

        lsc_mask_fn, rsc_mask_fn, oc_mask_fn, wc_mask_fn, fc_mask_fn, atc_mask_fn = [
            (rgb_handles_to_mask if c.masks_as_one_channel else lambda x: x
             ) for c in [lsc_ob, rsc_ob, oc_ob, wc_ob, fc_ob, atc_ob]]

        def get_rgb_depth(sensor: VisionSensor, get_rgb: bool, get_depth: bool,
                          get_pcd: bool, rgb_noise: NoiseModel,
                          depth_noise: NoiseModel, depth_in_meters: bool):
            rgb = depth = pcd = None
            if sensor is not None and (get_rgb or get_depth):
                sensor.handle_explicitly()
                if get_rgb:
                    rgb = sensor.capture_rgb()
                    if rgb_noise is not None:
                        rgb = rgb_noise.apply(rgb)
                    rgb = np.clip((rgb * 255.).astype(np.uint8), 0, 255)
                if get_depth or get_pcd:
                    depth = sensor.capture_depth(depth_in_meters)
                    if depth_noise is not None:
                        depth = depth_noise.apply(depth)
                if get_pcd:
                    depth_m = depth
                    if not depth_in_meters:
                        near = sensor.get_near_clipping_plane()
                        far = sensor.get_far_clipping_plane()
                        depth_m = near + depth * (far - near)
                    pcd = sensor.pointcloud_from_depth(depth_m)
                    if not get_depth:
                        depth = None
            return rgb, depth, pcd

        def get_mask(sensor: VisionSensor, mask_fn):
            mask = None
            if sensor is not None:
                sensor.handle_explicitly()
                mask = mask_fn(sensor.capture_rgb())
            return mask

        left_shoulder_rgb, left_shoulder_depth, left_shoulder_pcd = get_rgb_depth(
            self._cam_over_shoulder_left, lsc_ob.rgb, lsc_ob.depth, lsc_ob.point_cloud,
            lsc_ob.rgb_noise, lsc_ob.depth_noise, lsc_ob.depth_in_meters)
        right_shoulder_rgb, right_shoulder_depth, right_shoulder_pcd = get_rgb_depth(
            self._cam_over_shoulder_right, rsc_ob.rgb, rsc_ob.depth, rsc_ob.point_cloud,
            rsc_ob.rgb_noise, rsc_ob.depth_noise, rsc_ob.depth_in_meters)
        overhead_rgb, overhead_depth, overhead_pcd = get_rgb_depth(
            self._cam_overhead, oc_ob.rgb, oc_ob.depth, oc_ob.point_cloud,
            oc_ob.rgb_noise, oc_ob.depth_noise, oc_ob.depth_in_meters)
        wrist_rgb, wrist_depth, wrist_pcd = get_rgb_depth(
            self._cam_wrist, wc_ob.rgb, wc_ob.depth, wc_ob.point_cloud,
            wc_ob.rgb_noise, wc_ob.depth_noise, wc_ob.depth_in_meters)
        active_rgb, active_depth, active_pcd = get_rgb_depth(
            self._cam_active, atc_ob.rgb, atc_ob.depth, atc_ob.point_cloud,
            atc_ob.rgb_noise, atc_ob.depth_noise, atc_ob.depth_in_meters)
        front_rgb, front_depth, front_pcd = get_rgb_depth(
            self._cam_front, fc_ob.rgb, fc_ob.depth, fc_ob.point_cloud,
            fc_ob.rgb_noise, fc_ob.depth_noise, fc_ob.depth_in_meters)

        left_shoulder_mask = get_mask(self._cam_over_shoulder_left_mask,
                                      lsc_mask_fn) if lsc_ob.mask else None
        right_shoulder_mask = get_mask(self._cam_over_shoulder_right_mask,
                                      rsc_mask_fn) if rsc_ob.mask else None
        overhead_mask = get_mask(self._cam_overhead_mask,
                                 oc_mask_fn) if oc_ob.mask else None
        wrist_mask = get_mask(self._cam_wrist_mask,
                              wc_mask_fn) if wc_ob.mask else None
        active_mask = get_mask(self._cam_active_mask,
                              atc_mask_fn) if atc_ob.mask else None
        front_mask = get_mask(self._cam_front_mask,
                              fc_mask_fn) if fc_ob.mask else None

        obs = Observation(
            left_shoulder_rgb=left_shoulder_rgb,
            left_shoulder_depth=left_shoulder_depth,
            left_shoulder_point_cloud=left_shoulder_pcd,
            right_shoulder_rgb=right_shoulder_rgb,
            right_shoulder_depth=right_shoulder_depth,
            right_shoulder_point_cloud=right_shoulder_pcd,
            overhead_rgb=overhead_rgb,
            overhead_depth=overhead_depth,
            overhead_point_cloud=overhead_pcd,
            wrist_rgb=wrist_rgb,
            wrist_depth=wrist_depth,
            wrist_point_cloud=wrist_pcd,
            active_rgb=active_rgb,
            active_depth=active_depth,
            active_point_cloud=active_pcd,
            front_rgb=front_rgb,
            front_depth=front_depth,
            front_point_cloud=front_pcd,
            left_shoulder_mask=left_shoulder_mask,
            right_shoulder_mask=right_shoulder_mask,
            overhead_mask=overhead_mask,
            wrist_mask=wrist_mask,
            active_mask=active_mask,
            front_mask=front_mask,
            joint_velocities=(
                self._obs_config.joint_velocities_noise.apply(
                    np.array(self.robot.arm.get_joint_velocities()))
                if self._obs_config.joint_velocities else None),
            joint_positions=(
                self._obs_config.joint_positions_noise.apply(
                    np.array(self.robot.arm.get_joint_positions()))
                if self._obs_config.joint_positions else None),
            joint_forces=(joint_forces
                          if self._obs_config.joint_forces else None),
            gripper_open=(
                (1.0 if self.robot.gripper.get_open_amount()[0] > self.task.gripper_open_threshold else 0.0)
                if self._obs_config.gripper_open else None),
            gripper_pose=(
                np.array(tip.get_pose())
                if self._obs_config.gripper_pose else None),
            gripper_matrix=(
                tip.get_matrix()
                if self._obs_config.gripper_matrix else None),
            gripper_touch_forces=(
                ee_forces_flat
                if self._obs_config.gripper_touch_forces else None),
            gripper_joint_positions=(
                np.array(self.robot.gripper.get_joint_positions())
                if self._obs_config.gripper_joint_positions else None),
            task_low_dim_state=(
                self.task.get_low_dim_state() if
                self._obs_config.task_low_dim_state else None),
            misc=self._get_misc())
        obs = self.task.decorate_observation(obs)
        return obs

    def step(self):
        self.pyrep.step()
        self.task.step()
        if self._step_callback is not None:
            self._step_callback()

    def register_step_callback(self, func):
        self._step_callback = func

    def get_demo(self, 
                 record: bool=True,
                 callable_each_step: Callable[[Observation], None]=None, 
                 randomly_place: bool=True,
                 human_demos: bool=False,
                 viewpoint_env_bounds: list=[1.3,20,-135,1.3,60,135],
                 floating_cam:bool = False) -> Demo:
        """Returns a demo (list of observations)

        """
        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:
            self.init_episode(self._variation_index,
                              randomly_place=randomly_place)
        self._has_init_episode = False
        self._transition_index = 0

        demo = []
        self._processing_stage = None


        waypoints = self.task.get_waypoints()
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)
        

        viewpoints = self.task.get_viewpoints()
        if len(viewpoints) == 0:
            raise NoViewpointsError(
                'No waypoints were found.', self.task)
        

        self._reset_cam_pose(viewpoint_env_bounds=viewpoint_env_bounds,
                             floating_cam=floating_cam,
                             preset_viewpoints=viewpoints,
                             demo=demo,
                             callable_each_step=callable_each_step,
                             record=False, 
                             random=False) 

        if floating_cam:
            viewpoint_env_bounds = np.array(viewpoint_env_bounds)
            viewpoints = self.preprocess_floating_viewpoints(preset_viewpoints=viewpoints,
                                                             legal_theta_range=viewpoint_env_bounds[[1,4]],
                                                             legal_phi_range=viewpoint_env_bounds[[2,5]],
                                                             waypoints=waypoints,vp_r=viewpoint_env_bounds[0])

            floating_vp_trajectories = self.get_floating_vp_trajectories(viewpoints=viewpoints,
                                                                    legal_theta_range=viewpoint_env_bounds[[1,4]],
                                                                    vp_r=viewpoint_env_bounds[0],
                                                                    random_sample_theta=False)  
        else:
            viewpoint_env_bounds = np.array([0.65,45,-45,0.65,45,45])

            viewpoints = self.preprocess_arm_viewpoints(preset_viewpoints=viewpoints,
                                                        waypoints=waypoints,
                                                        vp_r=viewpoint_env_bounds[0],
                                                        theta_range=viewpoint_env_bounds[[1,4]],
                                                        phi_range=viewpoint_env_bounds[[2,5]])


        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            demo.append(self.get_observation())

        #plot_and_save_voxel(demo[-1],'init',0)

        while True:
            success = False
            for i in range(len(waypoints)):
                self._transition_index += 1

                if self.robot.vision_arm is not None:
                    self._processing_stage = 'viewpoint'
                    
                    if floating_cam:

                        floating_vp_trajectory = floating_vp_trajectories[i]
                        self.execute_floating_viewpoint(floating_vp_trajectory,demo=demo,record=record,
                                        callable_each_step=callable_each_step)
                    else:

                        self.exect_point(demo=demo,record=record,
                                        callable_each_step=callable_each_step,
                                        point=viewpoints[i],point_index=i,point_role='viewpoint')

                self._processing_stage = 'waypoint'
                target_waypoint = waypoints[i]
                #plot_and_save_voxel(demo[-1],'cam',i+1)#target_waypoint._waypoint.get_position())
                


                self.exect_point(demo=demo,record=record,
                                 callable_each_step=callable_each_step,
                                 point=target_waypoint,point_index=i,point_role='waypoint')
                #plot_and_save_voxel(demo[-1],'gripper',i+1)
                
            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            self._transition_index += 1
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)

        processed_demo = Demo(demo)
        processed_demo.num_reset_attempts = self._attempts + 1
        return processed_demo
    
    def _rotation_matrix_distance(self,rot1, rot2):
        # Compute the relative rotation matrix
        relative_rotation = rot1.inv() * rot2
        # Calculate the angle of the relative rotation
        angle = relative_rotation.magnitude()
        return angle

    def _reset_cam_pose(self,viewpoint_env_bounds:list,
                        floating_cam:bool,
                        preset_viewpoints:List[Waypoint],
                        demo:list,
                        callable_each_step:Callable[[Observation], None],
                        record:bool=False,
                        random=False):

        if random:
            init_viewpoint = np.random.uniform(viewpoint_env_bounds[:3],viewpoint_env_bounds[3:])
        else:
            init_r = (viewpoint_env_bounds[0]+viewpoint_env_bounds[3])/2
            init_theta = (viewpoint_env_bounds[1]+viewpoint_env_bounds[4])/2
            init_phi = 0.0
            init_viewpoint = np.array([init_r,init_theta,init_phi])
        

        pos,euler,quat = utils.spherical_to_cartesian(init_viewpoint,
                                                                 fixation=np.array(TABLE_COORD),
                                                                 rad=False,
                                                                 ctg_rot=None)
        init_cam_pose = np.concatenate([pos,quat],-1)
        self.show_virtual_cam(init_cam_pose)

        if floating_cam:

            self._cam_active.set_pose(init_cam_pose)
            self.step()

        else:
            init_viewpoint = Point(Dummy.create(), self.robot.vision_arm)
            init_viewpoint._waypoint.set_position(pos)
            init_viewpoint._waypoint.set_quaternion(quat)
            self.exect_point(demo=demo,record=record,
                            callable_each_step=callable_each_step,
                            point=init_viewpoint,point_index=0,point_role='init_viewpoint')
            
        
        return None

    def preprocess_arm_viewpoints(self,preset_viewpoints:List[Waypoint],
                                    waypoints:List[Waypoint],vp_r:float=0.65,
                                    theta_range:List=[45,45],
                                    phi_range:List=[-45,45]) -> List[Waypoint]:

        final_viewpoints = []

        for _,waypoint in enumerate(waypoints):

            waypoint_index = waypoint._waypoint.get_name()[-1]
            waypoint_position = waypoint._waypoint.get_position()
            waypoint_candidate_viewpoints = []
            waypoint_candidate_viewpoint_sphers = []

            for preset_viewpoint in preset_viewpoints:

                if "viewpoint{}".format(waypoint_index) in preset_viewpoint._waypoint.get_name():
                    intersection_point = utils.intersect_with_sphere(start_point=waypoint_position,
                                                                    end_point=preset_viewpoint._waypoint.get_position(),
                                                                    sphere_center=np.array(TABLE_COORD),
                                                                    vp_r=vp_r)
                    intersection_point_spher = utils.cartesian_to_spherical(wtc_pos=intersection_point,
                                                                fixation=TABLE_COORD)
                    
                    waypoint_candidate_viewpoint_sphers.append(intersection_point_spher)
                    waypoint_candidate_viewpoints.append(preset_viewpoint)
                    
            selected_index = 0

            if len(waypoint_candidate_viewpoint_sphers)>1:
                minimum_abs_phi = abs(waypoint_candidate_viewpoint_sphers[0][-1])
                for sub_index,candidate_viewpoint_spher in enumerate(waypoint_candidate_viewpoint_sphers):
                    abs_phi = abs(candidate_viewpoint_spher[-1])
                    if abs_phi<minimum_abs_phi:
                            selected_index = sub_index 
                            minimum_abs_phi = abs_phi


            selected_viewpoint = waypoint_candidate_viewpoints[selected_index]
            selected_viewpoint_spher = waypoint_candidate_viewpoint_sphers[selected_index]

            selected_viewpoint_spher[1] = np.clip(selected_viewpoint_spher[1],theta_range[0],theta_range[1])
            selected_viewpoint_spher[2] = np.clip(selected_viewpoint_spher[2],phi_range[0],phi_range[1])


            new_vp_trans,new_vp_euler,new_vp_quat = utils.spherical_to_cartesian(
                                                        vp = selected_viewpoint_spher,
                                                        fixation=np.array(TABLE_COORD),
                                                        rad=False,
                                                        ctg_rot=None)
            
            selected_viewpoint._waypoint.set_position(new_vp_trans)
            selected_viewpoint._waypoint.set_quaternion(new_vp_quat)

            final_viewpoints.append(selected_viewpoint)

            self.show_virtual_cam(pose = np.concatenate([final_viewpoints[-1]._waypoint.get_pose()],axis=-1))
            return final_viewpoints

    def preprocess_floating_viewpoints(self,preset_viewpoints:List[Waypoint],legal_theta_range:np.ndarray,legal_phi_range:np.ndarray,
                                      waypoints:List[Waypoint],vp_r:float=2.0) -> List[Waypoint]:

        final_viewpoints = []
        

        for _,waypoint in enumerate(waypoints):

            waypoint_index = waypoint._waypoint.get_name()[-1]
            waypoint_position = waypoint._waypoint.get_position()
            waypoint_candidate_viewpoints = []
            extension_string = preset_viewpoints[0]._waypoint.get_extension_string()

            for preset_viewpoint in preset_viewpoints:
                extension_string = preset_viewpoint._waypoint.get_extension_string()

                if "viewpoint{}".format(waypoint_index) in preset_viewpoint._waypoint.get_name():
                    start_point = waypoint_position
                    if "always_front" in extension_string:
                        intersection_point_spher = np.array([intersection_point_spher[0],45,0.0]) 
                    elif "abs_viewpoint" in extension_string:
                        start_point = np.array(TABLE_COORD)

                    intersection_point = utils.intersect_with_sphere(start_point=start_point,
                                                                    end_point=preset_viewpoint._waypoint.get_position(),
                                                                    sphere_center=np.array(TABLE_COORD),
                                                                    vp_r=vp_r)
                    intersection_point_spher = utils.cartesian_to_spherical(wtc_pos=intersection_point,
                                                                fixation=TABLE_COORD)

                    intersection_point_spher[1] = np.clip(intersection_point_spher[1],legal_theta_range[0],legal_theta_range[1])
                    intersection_point_spher[2] = np.clip(intersection_point_spher[2],legal_phi_range[0],legal_phi_range[1])



                    new_vp_trans,new_vp_euler,new_vp_quat = utils.spherical_to_cartesian(
                                                                vp = intersection_point_spher,
                                                                fixation=np.array(TABLE_COORD),
                                                                rad=False,
                                                                ctg_rot=None)
                    
                    preset_viewpoint._waypoint.set_position(new_vp_trans)
                    preset_viewpoint._waypoint.set_quaternion(new_vp_quat)
                    #self.show_point(pose = np.concatenate([new_vp_trans,new_vp_quat],axis=-1))
                    waypoint_candidate_viewpoints.append(preset_viewpoint)
            selected_index = 0

            if len(waypoint_candidate_viewpoints)>1:
                
                if  "x_positive_first" in extension_string:
                    largest_x = waypoint_candidate_viewpoints[0]._waypoint.get_position()[0]

                    for sub_index,waypoint_candidate_viewpoint in enumerate(waypoint_candidate_viewpoints):
                        x = waypoint_candidate_viewpoint._waypoint.get_position()[0]
                        if x > largest_x:
                            selected_index = sub_index 
                            largest_x = x
                            
                elif "away_from_robot"in extension_string:
                    largest_x = waypoint_candidate_viewpoints[0]._waypoint.get_position()[0]

                    for sub_index,waypoint_candidate_viewpoint in enumerate(waypoint_candidate_viewpoints):
                        x = waypoint_candidate_viewpoint._waypoint.get_position()[0]
                        if x > largest_x:
                            selected_index = sub_index 
                            largest_x = x
                else:

                    pass
                
            
            final_viewpoints.append(waypoint_candidate_viewpoints[selected_index])
            #self.show_point(pose = np.concatenate([final_viewpoints[-1]._waypoint.get_position(),new_vp_quat],axis=-1))
        return final_viewpoints


    def get_next_floating_vp_trajectory(self,start_vp:np.ndarray,target_vp:np.ndarray,
                          legal_theta_range:np.ndarray=np.array([20,60]),
                          vp_r=1.3,delta_phi=1.0,delta_theta=1.0,random_sample_theta=True):

        vp_difference = target_vp - start_vp
        interpolations_num = abs(vp_difference[-1]) // delta_phi
        interpolations = []
        theta_choice_list = list(range(int((legal_theta_range[1]-legal_theta_range[0])//delta_theta)+1))


        for insert_id in range(int(interpolations_num)):

            if vp_difference[-1] >= 0:
                phi_inserts = start_vp[2] + (insert_id+1)*delta_phi
            else:
                phi_inserts = start_vp[2] - (insert_id+1)*delta_phi
            
            if random_sample_theta:

                theta_inserts = legal_theta_range[0] + random.choice(theta_choice_list)*delta_theta
            else:
                theta_inserts = start_vp[1] + (insert_id + 1)*(vp_difference[1] / interpolations_num)
                


            new_vp = np.array([vp_r,theta_inserts,phi_inserts])

            new_vp_position,new_vp_euler,new_vp_quat = utils.spherical_to_cartesian(
                                                        vp = new_vp,
                                                        fixation=np.array(TABLE_COORD),
                                                        rad=False,
                                                        ctg_rot=None)
            
            new_vp_pose = np.concatenate([new_vp_position,new_vp_quat],-1)
            #self.show_point(new_vp_pose)
            interpolations.append(new_vp_pose)

        return interpolations

    def get_floating_vp_trajectories(self,viewpoints:List[Waypoint],legal_theta_range:np.ndarray,
                                 vp_r:Any,delta_phi=1.0,
                                 delta_theta=1.0,random_sample_theta=True):

        #self._cam_wrist

        current_vp_position = self._cam_active.get_position()
        all_vp_trajectories = []

        start_vp_spher = utils.cartesian_to_spherical(wtc_pos=current_vp_position,
                                                        fixation=TABLE_COORD)
        # 
        for viewpoint in viewpoints:

            viewpoint_shper = utils.cartesian_to_spherical(wtc_pos=viewpoint._waypoint.get_position(),
                                                        fixation=TABLE_COORD)

            vp_trajectory_points = self.get_next_floating_vp_trajectory(start_vp=start_vp_spher,
                                                                 target_vp=viewpoint_shper,
                                                                 legal_theta_range=legal_theta_range,
                                                                 vp_r=vp_r,
                                                                 delta_phi=delta_phi,
                                                                 delta_theta=delta_theta,
                                                                 random_sample_theta=random_sample_theta)

            vp_trajectory_points.append(viewpoint._waypoint.get_pose())

            all_vp_trajectories.append(vp_trajectory_points)

            start_vp_spher = viewpoint_shper


        return all_vp_trajectories

    def execute_floating_viewpoint(self,vp_trajectory:List[np.ndarray],demo:list,record:bool,
                    callable_each_step:Callable[[Observation], None]):
        # 
        for vp_trajectory_point in vp_trajectory:
            self.show_virtual_cam(vp_trajectory_point)

            self._cam_active.set_pose(vp_trajectory_point)
            self.step()
            self._demo_record_step(demo, record, callable_each_step)
            self.step()
  
        


    def move_viewpoint_frame(self):

        current_viewpoint_position = self.robot.vision_arm.gripper.get_position()

        local_org = Dummy("viewpoint_frame")
        local_org_position = local_org.get_position()

        vp2org = (current_viewpoint_position - local_org_position)/np.linalg.norm(current_viewpoint_position - local_org_position)
        
        target_rot_z = np.arctan2(vp2org[1],vp2org[0])

        local_org.set_orientation(np.array([0,0,target_rot_z]))
        


    def exect_point(self,demo:list,record:bool,
                    callable_each_step:Callable[[Observation], None],
                    point:Point, point_index:int, point_role:str):

        point.start_of_path()
        robot = point._robot
        if point.skip:
            return None

        grasped_objects = robot.gripper.get_grasped_objects()

        colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
            object_type=ObjectType.SHAPE) if s not in grasped_objects
                            and s not in robot.robot_shapes 
                            and s.is_collidable()
                            and robot.arm.check_arm_collision(s)]
        [s.set_collidable(False) for s in colliding_shapes]


        try:
            
            path = point.get_path()
            [s.set_collidable(True) for s in colliding_shapes]
        except ConfigurationPathError as e:
            [s.set_collidable(True) for s in colliding_shapes]
            raise DemoError(
                'Could not get a path for {} {}.'.format(point_role,point_index),
                self.task) from e
        

        ext = point.get_ext()
        path.visualize()
        
        done = False
        success = False

        while not done:
            done = path.step()
            if point_role == "viewpoint":
                self.move_viewpoint_frame()
            self.step()
            self._execute_demo_joint_position_action = path.get_executed_joint_position_action()

            self._demo_record_step(demo, record, callable_each_step)
            success, term = self.task.success()

        point.end_of_path()

        path.clear_visualization()

        if len(ext) > 0:
            contains_param = False
            start_of_bracket = -1
            gripper = self.robot.worker_arm.gripper
            if 'open_gripper(' in ext:
                gripper.release()
                start_of_bracket = ext.index('open_gripper(') + 13
                contains_param = ext[start_of_bracket] != ')'
                if not contains_param:
                    done = False
                    while not done:

                        done = gripper.actuate(1.0, 0.04)
                        self.pyrep.step()
                        self.task.step()
                        if self._obs_config.record_gripper_closing:
                            self._demo_record_step(
                                demo, record, callable_each_step)
            elif 'close_gripper(' in ext:
                start_of_bracket = ext.index('close_gripper(') + 14
                contains_param = ext[start_of_bracket] != ')'
                if not contains_param:
                    done = False
                    while not done:
                        done = gripper.actuate(0.0, 0.04)
                        self.pyrep.step()
                        self.task.step()
                        if self._obs_config.record_gripper_closing:
                            self._demo_record_step(
                                demo, record, callable_each_step)

            if contains_param:
                rest = ext[start_of_bracket:]
                num = float(rest[:rest.index(')')])
                done = False
                while not done:
                    done = gripper.actuate(num, 0.04)
                    self.pyrep.step()
                    self.task.step()
                    if self._obs_config.record_gripper_closing:
                        self._demo_record_step(
                            demo, record, callable_each_step)

            if 'close_gripper(' in ext:
                for g_obj in self.task.get_graspable_objects():
                    gripper.grasp(g_obj)

            self._demo_record_step(demo, record, callable_each_step)



    def get_observation_config(self) -> ObservationConfig:
        return self._obs_config

    def check_target_in_workspace(self, target_pos: np.ndarray) -> bool:
        x, y, z = target_pos
        return (self._workspace_maxx > x > self._workspace_minx and
                self._workspace_maxy > y > self._workspace_miny and
                self._workspace_maxz > z > self._workspace_minz)

    def _demo_record_step(self, demo_list, record, func):
        if record:
            demo_list.append(self.get_observation())
        if func is not None:
            func(self.get_observation())

    def _set_camera_properties(self) -> None:
        def _set_rgb_props(rgb_cam: VisionSensor,
                           rgb: bool, depth: bool, conf: CameraConfig):
            if not (rgb or depth or conf.point_cloud):
                rgb_cam.remove()
            else:
                rgb_cam.set_explicit_handling(1)
                rgb_cam.set_resolution(conf.image_size)
                rgb_cam.set_render_mode(conf.render_mode)

        def _set_mask_props(mask_cam: VisionSensor, mask: bool,
                            conf: CameraConfig):
                if not mask:
                    mask_cam.remove()
                else:
                    # print(f"mask_cam: {mask_cam}, name: {mask_cam.get_name() if mask_cam else 'None'}")
                    mask_cam.set_explicit_handling(1)
                    mask_cam.set_resolution(conf.image_size)
        _set_rgb_props(
            self._cam_over_shoulder_left,
            self._obs_config.left_shoulder_camera.rgb,
            self._obs_config.left_shoulder_camera.depth,
            self._obs_config.left_shoulder_camera)
        _set_rgb_props(
            self._cam_over_shoulder_right,
            self._obs_config.right_shoulder_camera.rgb,
            self._obs_config.right_shoulder_camera.depth,
            self._obs_config.right_shoulder_camera)
        _set_rgb_props(
            self._cam_overhead,
            self._obs_config.overhead_camera.rgb,
            self._obs_config.overhead_camera.depth,
            self._obs_config.overhead_camera)
        _set_rgb_props(
            self._cam_wrist, self._obs_config.wrist_camera.rgb,
            self._obs_config.wrist_camera.depth,
            self._obs_config.wrist_camera)
        _set_rgb_props(
            self._cam_active, self._obs_config.active_camera.rgb,
            self._obs_config.active_camera.depth,
            self._obs_config.active_camera)
        _set_rgb_props(
            self._cam_front, self._obs_config.front_camera.rgb,
            self._obs_config.front_camera.depth,
            self._obs_config.front_camera)
        _set_mask_props(
            self._cam_over_shoulder_left_mask,
            self._obs_config.left_shoulder_camera.mask,
            self._obs_config.left_shoulder_camera)
        _set_mask_props(
            self._cam_over_shoulder_right_mask,
            self._obs_config.right_shoulder_camera.mask,
            self._obs_config.right_shoulder_camera)
        _set_mask_props(
            self._cam_overhead_mask,
            self._obs_config.overhead_camera.mask,
            self._obs_config.overhead_camera)
        _set_mask_props(
            self._cam_wrist_mask, self._obs_config.wrist_camera.mask,
            self._obs_config.wrist_camera)
        _set_mask_props(
            self._cam_active_mask, self._obs_config.active_camera.mask,
            self._obs_config.active_camera)
        _set_mask_props(
            self._cam_front_mask, self._obs_config.front_camera.mask,
            self._obs_config.front_camera)

    def _place_task(self) -> None:
        self._workspace_boundary.clear()
        # Find a place in the robot workspace for task
        self.task.boundary_root().set_orientation(
            self._initial_task_pose)
        min_rot, max_rot = self.task.base_rotation_bounds()
        self._workspace_boundary.sample(
            self.task.boundary_root(),
            min_rotation=min_rot, max_rotation=max_rot)

    def _get_state(self) -> Dict[str, Optional[np.ndarray]]:
        if self.task is None:
            raise RuntimeError('Initialize the task first')
        state = (
            self.task.state
            if self._obs_config.state
            else None
        )

        return {'state': state}

    def _get_misc(self):
        def _get_cam_data(cam: VisionSensor, name: str):
            d = {}
            if cam.still_exists():
                d = {
                    '%s_extrinsics' % name: cam.get_matrix(),
                    '%s_intrinsics' % name: cam.get_intrinsic_matrix(),
                    '%s_near' % name: cam.get_near_clipping_plane(),
                    '%s_far' % name: cam.get_far_clipping_plane(),
                }
            return d
        misc = _get_cam_data(self._cam_over_shoulder_left, 'left_shoulder_camera')
        misc.update(_get_cam_data(self._cam_over_shoulder_right, 'right_shoulder_camera'))
        misc.update(_get_cam_data(self._cam_overhead, 'overhead_camera'))
        misc.update(_get_cam_data(self._cam_front, 'front_camera'))
        misc.update(_get_cam_data(self._cam_wrist, 'wrist_camera'))
        misc.update(_get_cam_data(self._cam_active, 'active_camera'))

        misc.update({"variation_index": self._variation_index})
        if self._execute_demo_joint_position_action is not None:
            # Store the actual requested joint positions during demo collection
            misc.update({"executed_demo_joint_position_action": self._execute_demo_joint_position_action})
            self._execute_demo_joint_position_action = None
        return misc
