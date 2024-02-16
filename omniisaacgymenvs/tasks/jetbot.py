import torch
import omni
import time
import pdb
import numpy as np
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.objects import VisualCuboid

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.jetbot import Jetbot

class JetbotTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)
        self._max_episode_length = 256

        # jetbot position 3, jetbot orientation 4, linear velocity 3, angular velocity 3, goal position 3 
        self._num_observations = 16 
        self._num_actions = 2

        RLTask.__init__(self, name, env)
        return
    
    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]

    def set_up_scene(self, scene) -> None:
        self.get_jetbot()
        self.get_goal(scene)
        super().set_up_scene(scene)
        start_time = time.time()
        while True:
            current_time = time.time()
            if current_time - start_time >= 60:  
                break

            # Yield control for event processing
            omni.kit.app.get_app().update()
            time.sleep(0.01)  # Small delay to avoid excessive yielding


        stage = omni.usd.get_context().get_stage()
        # def print_prims(prim, indent=0):
        #     print(" "*indent + prim.GetPath().pathString)
        #     for child_prim in prim.GetChildren():
        #         print_prims(child_prim, indent=indent+2)

        def print_prims(prim, indent=0):
            print(" " * indent + prim.GetPath().pathString)
            for child_prim in prim.GetChildren():
                print_prims(child_prim, indent=indent+2)

        # print_prims(stage.GetDefaultPrim())
       
        self._jetbots = ArticulationView(
             prim_paths_expr="/World/envs/.*/Jetbot", name="jetbot_view", reset_xform_properties=False
        )

        self._goals = GeometryPrimView(
            prim_paths_expr="/World/envs/.*/Goal", name="goal_view", reset_xform_properties=False
        )
        scene.add(self._jetbots)
        scene.add(self._goals)

    def get_jetbot(self):
        jetbot = Jetbot(
            prim_path=self.default_zero_env_path + "/Jetbot",
            name="Jetbot"
        )

        print("jetbot path")
        print(jetbot.prim_path)
        self._sim_config.apply_articulation_settings(
            "Jetbot", get_prim_at_path(jetbot.prim_path), self._sim_config.parse_actor_config("Jetbot")
        )

    def get_goal(self, scene):

        goal = VisualCuboid(
                prim_path=self.default_zero_env_path + "/Goal",
                name="Goal",
                position=np.array([0.60, 0.30, 0.05]),
                size=0.1,
                color=np.array([1.0, 0, 0]),
            )
        
        scene.add(goal)


    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("jetbot_view"):
            scene.remove_object("jetbot_view", registry_only=True)
        if scene.object_exists("goal_view"):
            scene.remove_object("goal_view", registry_only=True)

        self._jetbots = ArticulationView(
            prim_paths_expr="/World/envs/.*/Jetbot", name="jetbot_view", reset_xform_properties=False
        )
        self._goals = GeometryPrimView(
            prim_paths_expr="/World/envs/.*/Goal", name="goal_view", reset_xform_properties=False
        )

        scene.add(self._jetbots)
        scene.add(self._goals)

    def get_observations(self) -> dict:
        jetbot_world_position, jetbot_world_orientation = self._jetbots.get_world_poses(clone = False)
        jetbot_linear_velocity = self._jetbots.get_linear_velocities()
        jetbot_angular_velocity = self._jetbots.get_angular_velocities()
        goal_world_position, _ = self._goals.get_world_poses()

        self.obs_buf[:, :3] = jetbot_world_position
        self.obs_buf[:, 3:7] = jetbot_world_orientation
        self.obs_buf[:, 7:10] = jetbot_linear_velocity
        self.obs_buf[:, 10:13] = jetbot_angular_velocity
        self.obs_buf[:, 13:16] = goal_world_position 

        observations = {self._jetbots.name: {"obs_buf": self.obs_buf}}
        return observations


    def pre_physics_step(self, action) -> None:
        if not self.world.is_playing():
            return
        self.jetbot_prev_pos = self.obs_buf[:, :3]


    def calculate_metrics(self) -> None:
        goal_position, _ = self._goals.get_world_poses()
        jetbot_cur_pos, _ = self._jetbots.get_world_poses(clone = False)
        previous_dist_to_goal = torch.linalg.norm(goal_position - self.jetbot_prev_pos, dim=1)
        current_dist_to_goal = torch.linalg.norm(goal_position - jetbot_cur_pos, dim=1)
        reward = previous_dist_to_goal - current_dist_to_goal

        self.rew_buf[:] = reward

    def is_done(self) -> None:
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, 0)
        goal_position, _ = self._goals.get_world_poses()
        jetbot_cur_pos, _ = self._jetbots.get_world_poses(clone = False)
        current_dist_to_goal = torch.linalg.norm(goal_position - jetbot_cur_pos, dim=1)
        resets = torch.where(current_dist_to_goal < .1, 1, resets)
        self.reset_buf[:] = resets

    def post_reset(self):
        None