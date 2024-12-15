import numpy as np
from simulator import Simulator
from pathlib import Path
from typing import Dict
import os 
import pinocchio as pin
import matplotlib.pyplot as plt 

def quaternion_to_rotation_matrix(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

def tsid_controller(q: np.ndarray, dq: np.ndarray, t: float, desired: Dict) -> np.ndarray:
    """Task-space controller for the robot."""

    pin.computeAllTerms(model, data, q, dq)
    M = data.M
    nle = data.nle

    kp = np.array([1000, 1000, 1000])
    kd = np.array([200, 200, 200])
    
    # Desired pose
    desired_position = desired['pos'] 
    desired_quaternion = desired['quat']

    # Compute current end-effector pose using Pinocchio
    pin.forwardKinematics(model, data, q, dq)
    ee_frame_id = model.getFrameId("end_effector")
    frame = pin.LOCAL
    pin.updateFramePlacement(model, data, ee_frame_id)
    
    # Get velocities
    twist = pin.getFrameVelocity(model, data, ee_frame_id, frame)

    # Jacobian
    J = pin.getFrameJacobian(model, data, ee_frame_id, frame)
    dJ = pin.getFrameJacobianTimeVariation(model, data, ee_frame_id, frame)

    ee_pose = data.oMf[ee_frame_id]
    ee_position = ee_pose.translation
    ee_rotation = ee_pose.rotation
    
    # Compute position and orientation errors
    position_error = desired_position - ee_position
    
    # Convert quaternion to rotation matrix and compute orientation error
    desired_rotation_matrix = quaternion_to_rotation_matrix(desired_quaternion)
    
    # Corrected orientation error calculation
    orientation_error = pin.log(desired_rotation_matrix @ ee_rotation.T)
    pose_err = np.zeros(6)
    pose_err[:3] = position_error
    pose_err[3:] = orientation_error

    # Calculate desired accelerations using the outer loop control equation
    a_x = kp * position_error + kd * (0 - twist.linear)
    a_w = kp * orientation_error + kd * (0 - twist.angular)
    
    desired_acceleration = np.concatenate((a_x, a_w))

    if np.linalg.det(J) == 0:
        J_inv = np.linalg.pinv(J)
    else:
        J_inv = np.linalg.inv(J)

    ddq = J_inv @ (desired_acceleration - dJ @ dq)
    
    tau = nle + M @ ddq

    return tau

def plot_results(times: np.ndarray, positions: np.ndarray, velocities: np.ndarray):
    """Plot and save simulation results."""
    # Joint positions plot
    plt.figure(figsize=(10, 6))
    for i in range(positions.shape[1]):
        plt.plot(times, positions[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Positions [rad]')
    plt.title('Joint Positions over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/05_positions.png')
    plt.close()
    
    # Joint velocities plot
    plt.figure(figsize=(10, 6))
    for i in range(velocities.shape[1]):
        plt.plot(times, velocities[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Velocities [rad/s]')
    plt.title('Joint Velocities over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/05_velocities.png')
    plt.close()

def main():
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    Path("logs/plots").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning task space controller...")
    
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=True,
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/05_tsid.mp4",
        fps=30,
        width=1920,
        height=1080
    )
      
    sim.set_controller(tsid_controller)
    sim.run(time_limit=10.0)
    
    # Simulation parameters
    t = 0
    dt = sim.dt
    time_limit = 10.0
    
    # Data collection
    times = []
    positions = []
    velocities = []

    desired = {
        'pos': np.array([0.5, 0, 0.5]),
        'quat': np.array([1.0, 0.0, 0.0, 0.0])
    }
    
    while t < time_limit:
        state = sim.get_state()
        times.append(t)
        positions.append(state['q'])
        velocities.append(state['dq'])
        
        tau = tsid_controller(q=state['q'], dq=state['dq'], t=t, desired=desired)
        sim.step(tau)
        
        if sim.record_video and len(sim.frames) < sim.fps * t:
            sim.frames.append(sim._capture_frame())
        t += dt
    
    # Process and save results
    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    print(f"Simulation completed: {len(times)} steps")
    print(f"Final joint positions: {positions[-1]}")
    
    plot_results(times, positions, velocities)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
    
    model = pin.buildModelFromMJCF(xml_path)
    data = model.createData()
    
    main()