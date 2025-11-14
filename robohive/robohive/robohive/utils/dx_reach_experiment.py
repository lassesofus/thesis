""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from robohive.utils import gym
from robohive.utils.paths_utils import plot as plotnsave_paths
import click
import numpy as np
import pickle
import time
import os
import pdb

from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.min_jerk import generate_joint_space_min_jerk

DESC = '''
Helper script to examine an environment and associated policy for behaviors; \n
- either onscreen, or offscreen, or just rollout without rendering.\n
- save resulting paths as pickle or as 2D plots \n
- rollout either learned policies or scripted policies (e.g. see rand_policy class below) \n
USAGE:\n
    $ python examine_env.py --env_name door-v1 \n
    $ python examine_env.py --env_name door-v1 --policy_path robohive.utils.examine_env.rand_policy \n
    $ python examine_env.py --env_name door-v1 --policy_path my_policy.pickle --mode evaluation --episodes 10 \n
'''

# Config
ARM_nJnt = 7
EE_SITE  = "end_effector" 

# Random policy
class rand_policy():
    def __init__(self, env, seed):
        self.env = env
        self.env.action_space.seed(seed) # requires explicit seeding

    def get_action(self, obs):
        # return self.env.np_random.uniform(high=self.env.action_space.high, low=self.env.action_space.low)
        return self.env.action_space.sample(), {'mode': 'random samples', 'evaluation':self.env.action_space.sample()}

class ScriptedPolicyVJEPA():
    """
    A scripted policy that executes a sequence of tasks:
    1.  Move to a target position using a smooth trajectory.
    2.  Take a snapshot (RGB image).
    3.  Move back to the starting home position.
    4.  Take another snapshot.
    5.  Initiate a placeholder for V-JEPA based CEM planning.
    This policy acts like a "state machine", moving through different phases.
    """
    def __init__(self, env, seed, fixed_target=False, fixed_target_offset=(0.0, 0.0, 0.2), camera_name=None, output_dir="/tmp"):
        # --- Store initial parameters passed from the main script ---
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.fixed_target = fixed_target
        self.fixed_target_offset = np.asarray(fixed_target_offset, dtype=np.float64)
        self.camera_name = camera_name or DEFAULT_CAMERA
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # --- State machine variables ---
        self.phase = "init" # The current phase of the policy (e.g., "to_target", "snap_at_home")
        self.waypoint_step = 0 # Counter for steps within a trajectory
        self.trajectory_to_target = None
        self.trajectory_to_home = None
        self.vjepa_plan = [] # To store the action plan from the V-JEPA planner

    def _initialize_episode(self):
        """
        This function is called once at the beginning of each new episode.
        It calculates the target position and plans the smooth trajectories.
        """
        print("\nInitializing new episode for scripted policy...")
        sim = self.env.sim

        # 1. Get the starting 'home' joint positions for the arm
        q_home = sim.data.qpos[:ARM_nJnt].copy()

        # 2. Determine the 3D world coordinate of the target
        start_pos = sim.data.site_xpos[sim.model.site_name2id(EE_SITE)].copy()
        target_pos = start_pos + self.fixed_target_offset
# """ =================================================
# Copyright (C) 2018 Vikash Kumar
# Author  :: Vikash Kumar (vikashplus@gmail.com)
# Source  :: https://github.com/vikashplus/robohive
# License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
# ================================================= """
DESC = """
TUTORIAL: Calculate min jerk trajectory using IK with reactive planning\n
    - Plans next target dynamically after reaching current target
    - NOTE: written for franka_busbin_v0.xml model and might not be too generic
EXAMPLE:\n
    - python tutorials/ik_minjerk_twoway_trajectory.py --sim_path envs/arms/franka/assets/franka_busbin_v0.xml --num_targets 5\n
"""

from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.min_jerk import *
from robohive.utils.quat_math import euler2quat
import click
import numpy as np

BIN_POS = np.array([.235, 0.5, .85])
BIN_DIM = np.array([.2, .3, 0])
BIN_TOP = 0.10
ARM_nJnt = 7

@click.command(help=DESC)
@click.option('-s', '--sim_path', type=str, help='environment to load', required= True, default=r'C:\Users\Lasse\miniconda3\envs\robohive\Lib\site-packages\robohive\envs\arms\franka\assets\franka_reach_v0.xml')
@click.option('-h', '--horizon', type=int, help='time (s) for each trajectory segment', default=4)
@click.option('--hold_time', type=float, help='hold duration at each target (s)', default=0.5)
@click.option('-n', '--num_targets', type=int, help='number of consecutive targets to reach', default=3)
@click.option('--return_home', type=bool, help='return to start position at end', default=True)
def main(sim_path, horizon, hold_time, num_targets, return_home):
    # Prep
    sim = SimScene.get_sim(model_handle=sim_path)

    # setup
    target_sid = sim.model.site_name2id("target")
    ARM_JNT0 = np.mean(sim.model.jnt_range[:ARM_nJnt], axis=-1)

    # State machine for reactive planning
    current_trajectory = None  # Current segment being executed
    traj_step = 0
    targets_reached = 0
    phase = 'idle'  # 'idle' | 'moving' | 'holding' | 'returning'
    hold_counter = 0
    hold_steps = int(hold_time / sim.model.opt.timestep)

    def sample_new_target():
        """Generate a random target pose"""
        target_pos = BIN_POS + np.random.uniform(high=BIN_DIM, low=-1*BIN_DIM) + np.array([0, 0, BIN_TOP])
        target_elr = np.random.uniform(high=[3.14, 0, 0], low=[3.14, 0, -3.14])
        target_quat = euler2quat(target_elr)
        return {'pos': target_pos, 'quat': target_quat}

    def plan_to_target(start_qpos, target_info):
        """Plan trajectory from start to target using IK"""
        # IK for target
        ik_result = qpos_from_site_pose(
            physics=sim,
            site_name="end_effector",
            target_pos=target_info['pos'],
            target_quat=target_info['quat'],
            inplace=False,
            regularization_strength=1.0
        )
        
        if not ik_result.success:
            print(f"  WARNING: IK failed (err: {ik_result.err_norm:.4f})")
        
        # Generate min-jerk trajectory
        trajectory = generate_joint_space_min_jerk(
            start=start_qpos,
            goal=ik_result.qpos[:ARM_nJnt],
            time_to_go=horizon,
            dt=sim.model.opt.timestep
        )
        
        return trajectory, ik_result.qpos[:ARM_nJnt]

    while True:

        # Initialize episode
        if sim.data.time == 0:
            print(f"\n=== New Episode: {num_targets} consecutive targets (reactive planning) ===")
            
            # Reset arm
            sim.data.qpos[:ARM_nJnt] = ARM_JNT0
            sim.forward()
            
            # Sample first target and plan
            current_target = sample_new_target()
            print(f"\nTarget 1/{num_targets}: pos={np.round(current_target['pos'], 3)}")
            
            # Update visualization
            sim.model.site_pos[target_sid][:] = current_target['pos'] - np.array([0, 0, BIN_TOP])
            sim.model.site_quat[target_sid][:] = current_target['quat']
            
            # Plan trajectory to first target
            current_trajectory, target_qpos = plan_to_target(ARM_JNT0, current_target)
            print(f"  Planned trajectory: {len(current_trajectory)} steps ({len(current_trajectory)*sim.model.opt.timestep:.2f}s)")
            
            phase = 'moving'
            traj_step = 0
            targets_reached = 0
            hold_counter = 0

        # Execute current phase
        if phase == 'moving' and current_trajectory is not None:
            # Execute trajectory
            if traj_step < len(current_trajectory):
                sim.data.ctrl[:ARM_nJnt] = current_trajectory[traj_step]['position']
                traj_step += 1
            else:
                # Reached target, start holding
                print(f"  Reached target {targets_reached + 1}, holding...")
                phase = 'holding'
                hold_counter = 0

        elif phase == 'holding':
            # Hold at current position
            sim.data.ctrl[:ARM_nJnt] = current_trajectory[-1]['position']
            hold_counter += 1
            
            if hold_counter >= hold_steps:
                targets_reached += 1
                
                # Check if we should plan next target or return home
                if targets_reached < num_targets:
                    # Sample and plan next target
                    current_target = sample_new_target()
                    print(f"\nTarget {targets_reached + 1}/{num_targets}: pos={np.round(current_target['pos'], 3)}")
                    
                    # Update visualization
                    sim.model.site_pos[target_sid][:] = current_target['pos'] - np.array([0, 0, BIN_TOP])
                    sim.model.site_quat[target_sid][:] = current_target['quat']
                    
                    # Plan from current position (use planned final pose for smoothness)
                    sim.forward()
                    current_qpos = current_trajectory[-1]['position']  # Use planned position
                    
                    current_trajectory, target_qpos = plan_to_target(current_qpos, current_target)
                    print(f"  Planned trajectory: {len(current_trajectory)} steps ({len(current_trajectory)*sim.model.opt.timestep:.2f}s)")
                    
                    phase = 'moving'
                    traj_step = 0
                    hold_counter = 0
                    
                elif return_home:
                    # Return to home position
                    print(f"\nReturning to home position...")
                    sim.forward()
                    current_qpos = current_trajectory[-1]['position']
                    
                    current_trajectory = generate_joint_space_min_jerk(
                        start=current_qpos,
                        goal=ARM_JNT0,
                        time_to_go=horizon,
                        dt=sim.model.opt.timestep
                    )
                    print(f"  Planned return: {len(current_trajectory)} steps ({len(current_trajectory)*sim.model.opt.timestep:.2f}s)")
                    
                    phase = 'returning'
                    traj_step = 0
                    
                else:
                    # Episode complete without return - reset immediately
                    print("\nEpisode complete, resetting...")
                    sim.reset()
                    continue  # Skip to next iteration to reinitialize

        elif phase == 'returning':
            # Execute return trajectory
            if traj_step < len(current_trajectory):
                sim.data.ctrl[:ARM_nJnt] = current_trajectory[traj_step]['position']
                traj_step += 1
            else:
                # Reached home, episode complete - reset
                print("Returned home, resetting...")
                sim.reset()
                continue  # Skip to next iteration to reinitialize

        sim.advance(render=True)

if __name__ == '__main__':
    main()
        # 3. Solve Inverse Kinematics (IK) to find the joint positions needed to reach the target
        print(f"IK: Moving from {start_pos.round(2)} to {target_pos.round(2)}")
        ik_result = qpos_from_site_pose(
            physics=sim,
            site_name=EE_SITE,
            target_pos=target_pos,
            target_quat=None, # Keep the same hand orientation
            inplace=False,
            regularization_strength=1.0
        )
        q_target = ik_result.qpos[:ARM_nJnt].copy()

        # 4. Generate smooth minimum-jerk trajectories for the arm's joints
        # This creates a list of joint positions for each timestep to move smoothly.
        trajectory_duration = 2.0 # seconds
        self.trajectory_to_target = generate_joint_space_min_jerk(start=q_home, goal=q_target, time_to_go=trajectory_duration, dt=sim.model.opt.timestep)
        self.trajectory_to_home = generate_joint_space_min_jerk(start=q_target, goal=q_home, time_to_go=trajectory_duration, dt=sim.model.opt.timestep)

        # 5. Reset state machine for the new episode
        self.waypoint_step = 0
        self.phase = "to_target"
        print("Initialization complete. Starting 'to_target' phase.")

    def _get_next_action_from_trajectory(self, trajectory):
        """
        Move to trajectory waypoints robustly:
         - compute desired absolute q_desired from trajectory
         - compute delta relative to current control target (sim.data.ctrl)
         - cap per-step delta to avoid overshoot/oscillations
         - form desired_ctrl = current_ctrl + capped_delta
         - return action that, when processed by the env, sets ctrl -> desired_ctrl
         - only advance waypoint when close enough to q_desired
        """
        if self.waypoint_step >= len(trajectory):
            return None  # Trajectory finished

        sim = self.env.sim
        q_desired = trajectory[self.waypoint_step]['position'].copy()

        # current control target used by the robot's internal controller
        current_ctrl = sim.data.ctrl[:ARM_nJnt].copy()

        # required change of the control target
        q_delta = q_desired - current_ctrl

        # --- cap per-step motion to avoid instability/oscillation ---
        # A small per-step angle (radians). Adjust if you want faster/slower motion.
        max_step = 0.05  # rad per control step (tune if needed)
        q_delta_capped = np.clip(q_delta, -max_step, max_step)

        # compute the control we will ask the env to set this step
        desired_ctrl = current_ctrl + q_delta_capped

        # Build the action the env expects
        action = np.zeros(self.env.action_space.shape, dtype=np.float32)

        if self.env.normalize_act:
            # Prefer actuator_ctrlrange mapping (common for position-actuated models)
            try:
                ctrl_range = sim.model.actuator_ctrlrange[:ARM_nJnt].copy()
                center = (ctrl_range[:, 1] + ctrl_range[:, 0]) / 2.0
                half = (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0
                half = np.where(half == 0.0, 1.0, half)
                action[:ARM_nJnt] = (desired_ctrl - center) / half
            except Exception:
                # fallback: use joint ranges
                jnt_ranges = sim.model.jnt_range[:ARM_nJnt].copy()
                centers = np.mean(jnt_ranges, axis=1)
                half_ranges = (jnt_ranges[:, 1] - jnt_ranges[:, 0]) / 2.0
                half_ranges = np.where(half_ranges == 0.0, 1.0, half_ranges)
                action[:ARM_nJnt] = (desired_ctrl - centers) / half_ranges
        else:
            action[:ARM_nJnt] = desired_ctrl

        # Decide whether we've reached the current waypoint (tolerance)
        tol = 1e-3  # radians
        if np.linalg.norm(q_desired - desired_ctrl) < tol:
            # waypoint reached -> advance to next
            self.waypoint_step += 1
        # else: keep waypoint_step so we continue moving toward it next step

        # Safety clip
        return np.clip(action, -1.0, 1.0)

    def _take_snapshot(self, name_prefix):
        """Captures an RGB image from the specified camera and saves it to a file."""
        if iio is None: return # imageio not installed, so we can't save images
        print(f"Taking snapshot: {name_prefix}")
        try:
            frame = self.env.sim.renderer.render_offscreen(
                width=640, height=480, camera_id=self.camera_name, device_id=0
            )
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.output_dir, f"{name_prefix}_{timestamp}.png")
            iio.imwrite(filename, frame)
            print(f"Saved snapshot to {filename}")
        except Exception as e:
            print(f"Warning: Could not save snapshot. Error: {e}")

    def _plan_with_vjepa(self, obs):
        """
        *** V-JEPA+CEM PLANNING PLACEHOLDER ***
        This is where you would integrate your actual V-JEPA based planner.
        It should take the current observation (and image) and return a sequence of actions.
        For now, it just returns a short sequence of 'do nothing' actions to simulate a plan.
        """
        print("Planning with V-JEPA (placeholder)...")
        plan_horizon = 50 # The number of steps in the generated plan
        action_dim = self.env.action_space.shape[0]
        # For now, the "plan" is just to stay still for 50 steps.
        self.vjepa_plan = [np.zeros(action_dim) for _ in range(plan_horizon)]
        self.waypoint_step = 0 # Reset step counter for executing the new plan
        print("Planning complete.")

    def get_action(self, obs):
        """The main policy function, which is called at every step of the simulation."""
        # Initialize at the very start of an episode (when time is 0)
        if self.env.time == 0.0:
            self._initialize_episode()

        action = np.zeros(self.env.action_space.shape) # Default action is to do nothing

        # --- STATE MACHINE LOGIC ---
        # The policy behaves differently depending on which "phase" it's in.
        if self.phase == "to_target":
            action = self._get_next_action_from_trajectory(self.trajectory_to_target)
            if action is None: # This means the trajectory is finished
                print("Phase transition: to_target -> snap_at_target")
                self.phase = "snap_at_target"
                self.waypoint_step = 0
                action = np.zeros(self.env.action_space.shape) # Do nothing for one step
        
        elif self.phase == "snap_at_target":
            self._take_snapshot("target_location")
            print("Phase transition: snap_at_target -> to_home")
            self.phase = "to_home"

        elif self.phase == "to_home":
            action = self._get_next_action_from_trajectory(self.trajectory_to_home)
            if action is None: # Trajectory finished
                print("Phase transition: to_home -> snap_at_home")
                self.phase = "snap_at_home"
                self.waypoint_step = 0
                action = np.zeros(self.env.action_space.shape) # Do nothing for one step

        elif self.phase == "snap_at_home":
            self._take_snapshot("home_location")
            print("Phase transition: snap_at_home -> plan_vjepa")
            self.phase = "plan_vjepa"

        elif self.phase == "plan_vjepa":
            self._plan_with_vjepa(obs)
            print("Phase transition: plan_vjepa -> execute_vjepa")
            self.phase = "execute_vjepa"

        elif self.phase == "execute_vjepa":
            if self.waypoint_step < len(self.vjepa_plan):
                action = self.vjepa_plan[self.waypoint_step]
                self.waypoint_step += 1
            else: # Plan finished
                print("V-JEPA plan execution finished. Episode is done.")
                self.phase = "done"

        # If phase is "done" or anything else, the default 'do nothing' action is used.
        
        # The policy must return the action and a dictionary with the 'evaluation' action.
        # This is required for compatibility with `examine_policy_new`.
        return action, {'phase': self.phase, 'evaluation': action}



def load_class_from_str(module_name, class_name):
    try:
        m = __import__(module_name, globals(), locals(), class_name)
        return getattr(m, class_name)
    except (ImportError, AttributeError):
        return None

# MAIN =========================================================
@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required= True)
@click.option('-p', '--policy_path', type=str, help='absolute path of the policy file', default=None)
@click.option('--use_vjepa_policy', is_flag=True, default=False, help='Use the built-in ScriptedPolicyVJEPA.')
@click.option('-m', '--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-n', '--num_episodes', type=int, help='number of episodes to visualize', default=3)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='onscreen')
@click.option('-c', '--camera_name', type=str, default=None, help=('Camera name for rendering'))
@click.option('-o', '--output_dir', type=str, default='/data/s185927/robohive/dx_exp', help=('Directory to save the outputs'))
@click.option('-on', '--output_name', type=str, default=None, help=('The name to save the outputs as'))
@click.option('-sp', '--save_paths', type=bool, default=False, help=('Save the rollout paths'))
@click.option('-pp', '--plot_paths', type=bool, default=False, help=('2D-plot of individual paths'))
@click.option('-rv', '--render_visuals', type=bool, default=False, help=('render the visual keys of the env, if present'))
@click.option('-ea', '--env_args', type=str, default=None, help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
@click.option('--fixed_target', is_flag=True, default=False,
              help='If set, use a fixed target offset relative to start instead of random sampling.')
@click.option('--fixed_target_offset', type=float, nargs=3, default=(0.0, 0.0, 0.2),
              help='dx dy dz (m) offset from start when --fixed_target is used (default: 0 0 0.2).')

def main(env_name, policy_path, use_vjepa_policy, mode, seed, num_episodes, render, camera_name, output_dir, output_name, save_paths, plot_paths, render_visuals, env_args, fixed_target, fixed_target_offset):

    # seed and load environments
    np.random.seed(seed)
    envw = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
    env = envw.unwrapped
    env.seed(seed)

    # resolve policy and outputs
    if use_vjepa_policy:
        # If the new flag is used, instantiate ScriptedPolicyVJEPA directly.
        print("Using the built-in ScriptedPolicyVJEPA.")
        pi = ScriptedPolicyVJEPA(env, seed,
                                 fixed_target=fixed_target,
                                 fixed_target_offset=fixed_target_offset,
                                 camera_name=camera_name,
                                 output_dir=output_dir)
        mode = 'evaluation'
        output_name = 'ScriptedPolicyVJEPA' if output_name is None else output_name

    elif policy_path is not None:
        # This is the previous logic for loading from a path.
        policy_tokens = policy_path.split('.')
        pi_class = load_class_from_str('.'.join(policy_tokens[:-1]), policy_tokens[-1])

        if pi_class is not None:
            # *** MODIFICATION: Pass extra arguments to the policy constructor ***
            # This allows our scripted policy to receive the target offset, camera name, etc.
            # from the command line.
            try:
                pi = pi_class(env, seed,
                              fixed_target=fixed_target,
                              fixed_target_offset=fixed_target_offset,
                              camera_name=camera_name,
                              output_dir=output_dir)
            except TypeError:
                # This is a fallback for older policies that don't accept the new arguments.
                print("Warning: Policy does not accept extended arguments. Initializing with (env, seed) only.")
                pi = pi_class(env, seed)
        else:
            # If not a class, assume it's a saved '.pickle' file.
            pi = pickle.load(open(policy_path, 'rb'))
            if output_dir == './': # overide the default
                output_dir, pol_name = os.path.split(policy_path)
                output_name = os.path.splitext(pol_name)[0]
            if output_name is None:
                pol_name = os.path.split(policy_path)[1]
                output_name = os.path.splitext(pol_name)[0]
    else:
        # If no policy is specified, use the simple random policy.
        print("No policy specified. Using a random policy.")
        pi = rand_policy(env, seed)
        mode = 'exploration'
        output_name ='random_policy' if output_name is None else output_name

    # resolve directory
    if (os.path.isdir(output_dir) == False) and (render=='offscreen' or save_paths or plot_paths is not None):
        os.mkdir(output_dir)

    # examine policy's behavior to recover paths
    paths = env.examine_policy_new(
        policy=pi,
        horizon=envw.spec.max_episode_steps,
        num_episodes=num_episodes,
        frame_size=(640,480),
        mode=mode,
        output_dir=output_dir+'/',
        filename=output_name,
        camera_name=camera_name,
        render=render)

    # evaluate paths
    success_percentage = env.evaluate_success(paths)
    print(f'Average success over rollouts: {success_percentage}%')

    # save paths
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    if save_paths:
        file_name = output_dir + '/' + output_name + '{}_trace.h5'.format(time_stamp)
        paths.save(trace_name=file_name, verify_length=True, f_res=np.float64)

    # plot paths
    if plot_paths:
        file_name = output_dir + '/' + output_name + '{}'.format(time_stamp)
        plotnsave_paths(paths, env=env, fileName_prefix=file_name)

    # render visuals keys
    if env.visual_keys and render_visuals:
        paths.close()
        render_keys = ['env_infos/visual_dict/'+ key for key in env.visual_keys]
        paths.render(output_dir=output_dir, output_format="mp4", groups=["Trial0",], datasets=render_keys, input_fps=1/env.dt)

if __name__ == '__main__':
    main()
