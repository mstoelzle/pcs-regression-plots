import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
# matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\Users\\Ricardo Valadas\\Downloads\\ffmpeg-7.0.2-essentials_build\\ffmpeg-7.0.2-essentials_build\\bin\\ffmpeg.exe"

# plotting settings
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)


def forward_kinematics(config_data, s, eps, pose_previous_frame):
    T, _ = config_data.shape
    strain_data = config_data.copy()
    strain_data[:,2] = strain_data[:,2] + 1

    k_be = strain_data[:,0]
    sigma_sh = strain_data[:,1]
    sigma_ax = strain_data[:,2]
    # add small eps for numerical stability in bending
    k_be_sign = np.sign(k_be)
    # set zero sign to 1 (i.e. positive)
    k_be_sign = np.where(k_be_sign == 0, 1, k_be_sign)
    # add eps to bending
    # k_be_eps = k_be + k_be_sign * eps
    k_be_eps = np.select(
        [np.abs(k_be) < eps, np.abs(k_be) >= eps],
        [k_be_sign*eps, k_be]
    )

    # Compute the pose from strains through closed-form FK
    
    px = sigma_sh * (np.sin(k_be_eps * s))/k_be_eps + \
        sigma_ax * (np.cos(k_be_eps * s) - 1)/k_be_eps
    py = sigma_sh * (1 - np.cos(k_be_eps * s))/k_be_eps + \
        sigma_ax * (np.sin(k_be_eps * s))/k_be_eps
    theta = k_be_eps * s

    # Pose w.r.t the frame of the previous segment
    pose = np.array([px, py, theta]).T

    # Change pose to be w.r.t the base frame
    pose_base_frame = np.zeros((T,3))
    # Compute the angle w.r.t the base frame
    pose_base_frame[:,2] = pose_previous_frame[:,2] + pose[:,2]
    # Compute the position w.r.t the base frame
    rot_mat = np.transpose((np.array([
        [np.cos(pose_previous_frame[:,2]), -np.sin(pose_previous_frame[:,2])],
        [np.sin(pose_previous_frame[:,2]), np.cos(pose_previous_frame[:,2])]
    ])), (2,0,1))
    pose_base_frame[:,:2] = pose_previous_frame[:,:2] + np.einsum('BNi,Bi ->BN', rot_mat, pose[:,:2])
    
    # # Change pose to be w.r.t the base frame
    # # Compute the angles w.r.t the base frame
    # pose_base_frame = np.cumsum(pose, axis=2)
    # for i in range(T):
    #     for j in range(1, N):
    #         theta = pose_base_frame[i,j,2]
    #         # Compute the position w.r.t the base frame
    #         pose_base_frame[i,j,:2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ pose[i,j,:2]


    return pose_base_frame

def compute_task_error(pose_data, config_data, seg_length_itrs, eps, config_data_2 = None):
    # point coordinates of the points where the error will be calculated
    s_image_cum = np.cumsum(seg_length_itrs[0])

    T, _, _ = config_data.shape
    N = 20
    pose_data_iterations = []
    if config_data_2 is not None:
        pose_data_iterations_2 = []
    error_metric_iterations = []
    for itr in range(1, len(seg_length_itrs)):
        # cumsum of the segment lengths
        s_itr_cum = np.cumsum(seg_length_itrs[itr])
        # add zero to the beginning of the array
        s_itr_cum_padded = np.concatenate([np.array([0.0]), s_itr_cum], axis=0, dtype=np.float32)

        # pose of the segment frame to which the FK are being computed w.r.t
        # for the first segment, it's the base frame, which is always the same at every frame
        pose_previous_frame = np.zeros((T,3))
        pose_previous_frame_2 = np.zeros((T,3))
        prev_segment_idx = 0

        pose_itr = np.zeros((T,N,3))
        if config_data_2 is not None:
            pose_itr_2 = np.zeros((T,N,3))

        for id_seg, s_point in enumerate(s_image_cum):
            # determine in which segment the point is located
            # use argmax to find the last index where the condition is true
            s_point = np.float32(s_point)
            segment_idx = (
                s_itr_cum.shape[0] - 1 - np.argmax((s_point > s_itr_cum_padded[:-1])[::-1]).astype(int)
            )
            
            if segment_idx != prev_segment_idx:
                pose_previous_frame = pose_itr[:, id_seg - 1, :]
                if config_data_2 is not None:
                    pose_previous_frame_2 = pose_itr_2[:, id_seg - 1, :]
                prev_segment_idx = segment_idx

            # point coordinate along the segment in the interval [0, l_segment]
            s_segment = s_point - s_itr_cum_padded[segment_idx]

            pose = forward_kinematics(config_data[:,segment_idx,:], s_segment, eps, pose_previous_frame)
            pose_itr[:,id_seg,:] = pose

            if config_data_2 is not None:
                pose_2 = forward_kinematics(config_data_2[:,segment_idx,:], s_segment, eps, pose_previous_frame_2)
                pose_itr_2[:,id_seg,:] = pose_2

        if high_shear_stiffness == True:
            np.save(f'./data/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_pred_poses.npy', pose_itr)
        else:
            np.save(f'./data/ns-{num_segments}/{validation_type}/ns-{num_segments}_pred_poses.npy', pose_itr)
        
        pose_data_iterations.append(pose_itr)
        if config_data_2 is not None:
            pose_data_iterations_2.append(pose_itr_2)

        # Create the figure and axis
        dt = 1e-3
        time_arr = np.arange(0.0, 7.0, dt)[::5]
        fig, ax = plt.subplots()
        ax.set_xlim(-0.11, 0.11)
        ax.set_ylim(-0.03, 0.17)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.grid(True)

        # Initialize the scatter plots for A and B
        plot_pose_data, = ax.plot([], [], 'b-o', label='Original image')
        plot_time = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14)
        if high_shear_stiffness == True:
            plot_pose_itr, = ax.plot([], [], 'r-o', label=f'Prediction by {num_segments} segment model (wo/ shear)')
            if config_data_2 is not None:
                plot_pose_itr_2, = ax.plot([], [], '-o', color='C1', label=f'Prediction by {num_segments} segment model (all strains)')
        elif high_stiffness == True:
            plot_pose_itr, = ax.plot([], [], 'r-o', label= f'Prediction by {num_segments} segment model (sparsified strains)')
        else:
            plot_pose_itr, = ax.plot([], [], 'r-o', label= f'Prediction by {num_segments} segment model (wo/ noise)')
            if config_data_2 is not None:
                plot_pose_itr_2, = ax.plot([], [], '-o', color='C1', label= f'Prediction by {num_segments} segment model (w/ noise)')

        # Initialize the legend
        ax.legend(loc='upper right')

        # Initialization function
        def init():
            plot_pose_data.set_data([], [])
            plot_pose_itr.set_data([], [])
            plot_time.set_text('')
            if config_data_2 is not None:
                plot_pose_itr_2.set_data([], [])
                return plot_pose_data, plot_pose_itr, plot_pose_itr_2, plot_time
            else:
                return plot_pose_data, plot_pose_itr, plot_time, plot_time

        # Update function
        def update(frame):
            plot_pose_data.set_data(pose_data[frame,:,0], pose_data[frame,:,1])
            plot_pose_itr.set_data(pose_itr[frame,:,0], pose_itr[frame,:,1])
            plot_time.set_text(f'T={time_arr[frame]:.3f} s')
            if config_data_2 is not None:
                plot_pose_itr_2.set_data(pose_itr_2[frame,:,0], pose_itr_2[frame,:,1])
                return plot_pose_data, plot_pose_itr, plot_pose_itr_2, plot_time
            else:
                return plot_pose_data, plot_pose_itr, plot_time

        # Create the animation
        ani = FuncAnimation(fig, update, frames=T, init_func=init, blit=True, repeat=True, interval=5)
        plt.show()

        print('Iteration ' + str(itr) + ':')
        error_position = np.mean(np.linalg.norm(pose_data[:,1:,:2] - pose_itr[:,:,:2], axis=2))
        print('\tmean position error: ' + str(error_position) + ' [m]')
        error_angle = np.mean(np.abs(pose_data[:,1:,2] - pose_itr[:,:,2]))*180/np.pi
        print('\tmean angle error: ' + str(error_angle) + ' [deg]')
        error_metric_iterations.append(np.array([error_position, error_angle]))

        if config_data_2 is not None:
            error_position_2 = np.mean(np.linalg.norm(pose_data[:,1:,:2] - pose_itr_2[:,:,:2], axis=2))
            print('\tmean position error: ' + str(error_position_2) + ' [m]')
            error_angle_2 = np.mean(np.abs(pose_data[:,1:,2] - pose_itr_2[:,:,2]))*180/np.pi
            print('\tmean angle error: ' + str(error_angle_2) + ' [deg]')

        # if itr == len(config_data_itrs) - 1:
        if high_shear_stiffness == True:
            if config_data_2 is not None:
                ani.save(filename = f"results/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_task_space_animation_comparison.mp4", writer=matplotlib.animation.FFMpegWriter(fps=60) )
            else:
                ani.save(filename = f"results/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_task_space_animation.mp4", writer=matplotlib.animation.FFMpegWriter(fps=60) )
        elif high_stiffness == True:
            ani.save(filename = f"results/ns-{num_segments}_high_stiffness/{validation_type}/ns-{num_segments}_task_space_animation.mp4", writer=matplotlib.animation.FFMpegWriter(fps=60) )
        else:
            if config_data_2 is not None:
                ani.save(filename = f"results/ns-{num_segments}/{validation_type}/ns-{num_segments}_task_space_animation_noise_comp.mp4", writer=matplotlib.animation.FFMpegWriter(fps=60) )
            else:
                ani.save(filename = f"results/ns-{num_segments}/{validation_type}/ns-{num_segments}_task_space_animation.mp4", writer=matplotlib.animation.FFMpegWriter(fps=60) )

        

    return pose_data_iterations, error_metric_iterations

validation_type = 'sinusoidal_actuation' # sinusoidal_actuation or step_actuation or training
high_shear_stiffness = False
high_stiffness = False
num_segments = 2
# params = {"l": 0.1 * np.ones((num_segments,))}
params = {"l": np.array([0.07, 0.1])}
params["total_length"] = np.sum(params["l"])
eps = 1e-7

if high_shear_stiffness == True:
    true_poses = np.load(f'./data/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_true_poses.npy')
    true_poses = np.transpose(true_poses, (0,2,1))

    q_pred = np.load(f'./data/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_q_pred.npy').T
    q_pred = np.concatenate((q_pred[:,0].reshape((q_pred.shape[0],1)), np.zeros((q_pred.shape[0],1)), q_pred[:,1].reshape((q_pred.shape[0],1))), axis=1)
    config_data = np.zeros((true_poses.shape[0], num_segments, 3))
    config_data[:,0,:] = q_pred[::5,:]

    q_pred_2 = np.load(f'./data/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_q_pred_all_strains.npy').T
    config_data_2 = np.zeros((true_poses.shape[0],1,3))
    config_data_2[:,0,:] = q_pred_2[::5,:]
elif high_stiffness == True:
    true_poses = np.load(f'./data/ns-{num_segments}/{validation_type}/ns-{num_segments}_true_poses.npy')
    if validation_type == 'training':
        true_poses = np.reshape(true_poses, (true_poses.shape[0], -1, 3))[:500]
    else:
        true_poses = np.transpose(true_poses, (0,2,1))

    q_pred = np.load(f'./data/ns-{num_segments}_high_stiffness/{validation_type}/ns-{num_segments}_q_pred.npy').T
    q_pred = np.concatenate((q_pred[:,0].reshape((q_pred.shape[0],1)), 
                             q_pred[:,1].reshape((q_pred.shape[0],1)), 
                             np.zeros((q_pred.shape[0],1)), 
                             q_pred[:,2].reshape((q_pred.shape[0],1)),
                             np.zeros((q_pred.shape[0],1)),
                             q_pred[:,3].reshape((q_pred.shape[0],1))), axis=1)
    config_data = np.zeros((true_poses.shape[0], num_segments, 3))
    for i in range(num_segments):
        config_data[:,i,:] = q_pred[::5,(3*i):(3*i+3)]
else:
    true_poses = np.load(f'./data/ns-{num_segments}/{validation_type}/ns-{num_segments}_true_poses.npy')
    if validation_type == 'training':
        true_poses = np.reshape(true_poses, (true_poses.shape[0], -1, 3))[:500]
    else:
        true_poses = np.transpose(true_poses, (0,2,1))

    q_pred = np.load(f'./data/ns-{num_segments}/{validation_type}/ns-{num_segments}_q_pred.npy').T
    config_data = np.zeros((true_poses.shape[0], num_segments, 3))
    for i in range(num_segments):
        if validation_type == 'training':
            config_data[:,i,:] = q_pred[:,(3*i):(3*i+3)]
        else:
            config_data[:,i,:] = q_pred[:,(3*i):(3*i+3)]

    q_pred_noise = np.load(f'./data/ns-{num_segments}_noise/{validation_type}/ns-{num_segments}_q_pred.npy').T
    config_data_noise = np.zeros((true_poses.shape[0], num_segments, 3))
    for i in range(num_segments):
        config_data_noise[:,i,:] = q_pred_noise[:,(3*i):(3*i+3)]

    q_pred_noise = np.load(f'./data/ns-{num_segments}_noise/{validation_type}/ns-{num_segments}_q_pred.npy').T
    config_data_noise = np.zeros((true_poses.shape[0], num_segments, 3))
    for i in range(num_segments):
        config_data_noise[:,i,:] = q_pred_noise[:,(3*i):(3*i+3)]

num_cs = 21
s = params["total_length"] / (num_cs - 1)
seg_length = s*np.ones((num_cs - 1))
seg_length_itrs = [seg_length, params['l']]

if high_shear_stiffness == True:
    pose_data_iterations, error_metric_iterations = compute_task_error(true_poses, config_data, seg_length_itrs, eps)
    pose_data_iterations, error_metric_iterations = compute_task_error(true_poses, config_data, seg_length_itrs, eps, config_data_2=config_data_2)
elif high_stiffness == True:
    pose_data_iterations, error_metric_iterations = compute_task_error(true_poses, config_data, seg_length_itrs, eps)
else:
    # pose_data_iterations, error_metric_iterations = compute_task_error(true_poses, config_data, seg_length_itrs, eps)
    pose_data_iterations, error_metric_iterations = compute_task_error(true_poses, config_data, seg_length_itrs, eps, config_data_2=config_data_noise)
    



