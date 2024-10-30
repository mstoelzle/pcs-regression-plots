import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
# Plotting settings
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=10)

color_palette = ['tab:blue', 'tab:orange', 'tab:green']

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

def compute_end_effector_poses(config_data, seg_length, eps):
    # point coordinates of the end_effector
    s_end_segments = np.cumsum(seg_length)

    T, N, _ = config_data.shape
    
    # for itr in range(1, len(seg_length_itrs)):

    # add zero to the beginning of the array
    s_end_segments_padded = np.concatenate([np.array([0.0]), s_end_segments], axis=0, dtype=np.float32)

    # pose of the segment frame to which the FK are being computed w.r.t
    # for the first segment, it's the base frame, which is always the same at every frame
    pose_previous_frame = np.zeros((T,3))
    prev_segment_idx = 0

    pose_end_segments = np.zeros((T,N,3))

    for id_seg, s_point in enumerate(s_end_segments):
        s_point = np.float32(s_point)
        # determine in which segment the point is located
        # use argmax to find the last index where the condition is true
        segment_idx = (
            s_end_segments.shape[0] - 1 - np.argmax((s_point > s_end_segments_padded[:-1])[::-1]).astype(int)
        )
        
        if segment_idx != prev_segment_idx:
            pose_previous_frame = pose_end_segments[:, id_seg - 1, :]
            prev_segment_idx = segment_idx

        # point coordinate along the segment in the interval [0, l_segment]
        s_segment = s_point - s_end_segments_padded[segment_idx]

        pose = forward_kinematics(config_data[:,segment_idx,:], s_segment, eps, pose_previous_frame)
        pose_end_segments[:,id_seg,:] = pose

    # for id_seg, s_point in enumerate(s_image_cum):
        
    pose_end_effector = pose_end_segments[:,-1,:]

    return pose_end_effector

validation_type = 'sinusoidal_actuation' # sinusoidal_actuation or step_actuation
high_shear_stiffness = False
high_stiffness = True
num_segments = 2
# params = {"l": 0.1 * np.ones((num_segments,))}
params = {"l": np.array([0.07, 0.1])}
params["total_length"] = np.sum(params["l"])
eps = 1e-7

if high_shear_stiffness == True:
    q_true = np.load(f'./data/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_q_true_all_strains.npy').T
    config_data_true = np.zeros((q_true.shape[0], num_segments, 3))
    config_data_true[:,0,:] = q_true
    q_pred = np.load(f'./data/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_q_pred.npy').T
    q_pred = np.concatenate((q_pred[:,0].reshape((q_pred.shape[0],1)), np.zeros((q_pred.shape[0],1)), q_pred[:,1].reshape((q_pred.shape[0],1))), axis=1)
    config_data_pred = np.zeros((q_pred.shape[0], num_segments, 3))
    config_data_pred[:,0,:] = q_pred

    q_pred_2 = np.load(f'./data/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_q_pred_all_strains.npy').T
    config_data_pred_2 = np.zeros((q_pred_2.shape[0], num_segments, 3))
    config_data_pred_2[:,0,:] = q_pred_2

    true_poses = np.load(f'./data/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_true_poses.npy')
    true_poses = np.transpose(true_poses, (0,2,1))

    pose_end_effector_true = compute_end_effector_poses(config_data_true, params['l'], eps)
    pose_end_effector_pred = compute_end_effector_poses(config_data_pred, params['l'], eps)
    pose_end_effector_pred_2 = compute_end_effector_poses(config_data_pred_2, params['l'], eps)

    error_position = np.linalg.norm(pose_end_effector_true[:,:2] - pose_end_effector_pred[:,:2], axis=1)
    print('End-effector errors (without shear)')
    print('\tmean position error: ' + str(np.mean(error_position)) + ' [m]')
    error_orientation = np.abs(pose_end_effector_true[:,2] - pose_end_effector_pred[:,2])
    print('\tmean angle error: ' + str(np.mean(error_orientation)*180/np.pi) + ' [deg]')

    print('End-effector errors (all strains)')
    error_position_2 = np.linalg.norm(pose_end_effector_true[:,:2] - pose_end_effector_pred_2[:,:2], axis=1)
    print('\tmean position error: ' + str(np.mean(error_position_2)) + ' [m]')
    error_orientation_2 = np.abs(pose_end_effector_true[:,2] - pose_end_effector_pred_2[:,2])
    print('\tmean angle error: ' + str(np.mean(error_orientation_2)*180/np.pi) + ' [deg]')

    t = np.arange(0.0, 7.0, 5e-3)
    t_fine = np.arange(0.0, 7.0, 1e-3)

    # fig, ax = plt.subplots(2, 1, sharex=True)
    # ax[0].plot(t_fine, error_position, label='Obtained Model (wo/ shear)')
    # ax[0].plot(t_fine, error_position_2, label='Obtained Model (all strains)')
    # ax[0].set_ylabel(r'Position error $[m]$')
    # ax[0].set_xlim([0,7.0])
    # ax[0].grid(True)
    # # ax[0].legend(loc='upper right')

    # ax[1].plot(t_fine, error_orientation, label='Obtained Model (wo/ shear)')
    # ax[1].plot(t_fine, error_orientation_2, label='Obtained Model (all strains)')
    # ax[1].set_ylabel(r'Orientation error $[rad]$')
    # ax[1].set_xlim([0,7.0])
    # ax[1].grid(True)
    # ax[1].legend(loc='upper right')
    # plt.xlabel('Time [s]')
    # plt.show()
    # plt.savefig(f'results/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_ee_error_plots_shear_vs_no_shear.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(3, 1, sharex=True)
    
    ax[0].plot(t, pose_end_effector_pred[::5,0], label='Model (wo/ shear)', linewidth=2.0, marker='s', mec='k', markevery=150, color=color_palette[1])
    ax[0].plot(t, pose_end_effector_pred_2[::5,0], label='Model (all strains)', linewidth=2.0, marker='^', mec='k', markevery=100, color=color_palette[2])
    ax[0].plot(t, true_poses[:,-1,0], label='GT', linewidth=3.5, linestyle='dotted', color='k', alpha=0.55)
    ax[0].set_ylabel(r'Position $(x)$ $[m]$')
    ax[0].set_xlim([0,7.0])
    ax[0].grid(True)
    # ax[0].legend(loc='upper left')
    # ax[0].legend(bbox_to_anchor=(0.7,0.6), loc='lower left')
    ax[0].legend(bbox_to_anchor=(-0.1,1.02), loc='lower left', ncol=3)

    ax[1].plot(t, pose_end_effector_pred[::5,1], label='Model (wo/ shear)', linewidth=2, marker='s', mec='k', markevery=150, color=color_palette[1])
    ax[1].plot(t, pose_end_effector_pred_2[::5,1], label='Model (all strains)', linewidth=2, marker='^', mec='k', markevery=100, color=color_palette[2])
    ax[1].plot(t, true_poses[:,-1,1], label='GT', linewidth=3.5, linestyle='dotted', color='k', alpha=0.55)
    ax[1].set_ylabel(r'Position $(y)$ $[m]$')
    ax[1].set_xlim([0,7.0])
    ax[1].grid(True)
    # ax[1].legend(loc='upper left')

    ax[2].plot(t, pose_end_effector_pred[::5,2], label='Model (wo/ shear)', linewidth=2, marker='s', mec='k', markevery=150, color=color_palette[1])
    ax[2].plot(t, pose_end_effector_pred_2[::5,2], label='Model (all strains)', linewidth=2, marker='^', mec='k', markevery=100, color=color_palette[2])
    ax[2].plot(t, true_poses[:,-1,2], label='GT', linewidth=3.5, linestyle='dotted', color='k', alpha=0.55)
    ax[2].set_ylabel(r'Orientation $(\theta)$ $[rad]$')
    ax[2].set_xlim([0,7.0])
    ax[2].grid(True)
    # ax[2].legend(loc='lower right')
    # plt.legend(bbox_to_anchor=(0,1.02), loc='lower left')

    plt.xlabel('Time [s]')
    plt.show()
    plt.savefig(f'results/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_ee_comparison_plots_shear_vs_no_shear.pdf', bbox_inches='tight')
elif high_stiffness == True:
    q_true = np.load(f'./data/ns-{num_segments}_high_stiffness/{validation_type}/ns-{num_segments}_q_true_all_strains.npy').T
    q_pred = np.load(f'./data/ns-{num_segments}_high_stiffness/{validation_type}/ns-{num_segments}_q_pred.npy').T
    q_pred = np.concatenate((q_pred[:,0].reshape((q_pred.shape[0],1)), 
                             q_pred[:,1].reshape((q_pred.shape[0],1)), 
                             np.zeros((q_pred.shape[0],1)), 
                             q_pred[:,2].reshape((q_pred.shape[0],1)),
                             np.zeros((q_pred.shape[0],1)),
                             q_pred[:,3].reshape((q_pred.shape[0],1))), axis=1)
    config_data_true = np.zeros((q_true.shape[0], num_segments, 3))
    config_data_pred = np.zeros((q_pred.shape[0], num_segments, 3))
    for i in range(num_segments):
        config_data_pred[:,i,:] = q_pred[:,(3*i):(3*i+3)]
        config_data_true[:,i,:] = q_true[:,(3*i):(3*i+3)]

    true_poses = np.load(f'./data/ns-{num_segments}_high_stiffness/{validation_type}/ns-{num_segments}_true_poses.npy')
    true_poses = np.transpose(true_poses, (0,2,1))

    pose_end_effector_true = compute_end_effector_poses(config_data_true, params['l'], eps)
    pose_end_effector_pred = compute_end_effector_poses(config_data_pred, params['l'], eps)

    error_position = np.linalg.norm(pose_end_effector_true[:,:2] - pose_end_effector_pred[:,:2], axis=1)
    print('End-effector errors')
    print('\tmean position error: ' + str(np.mean(error_position)) + ' [m]')
    error_orientation = np.abs(pose_end_effector_true[:,2] - pose_end_effector_pred[:,2])
    print('\tmean angle error: ' + str(np.mean(error_orientation)*180/np.pi) + ' [deg]')

    t = np.arange(0.0, 7.0, 5e-3)
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(t, pose_end_effector_pred[::5,0], label='Model (sparsified strains)', linewidth=2, color=color_palette[1], marker='s', mec='k', markevery=100)
    ax[0].plot(t, true_poses[:,-1,0], label='GT', linewidth=3.5, linestyle='dotted', color='k', alpha=0.55)
    ax[0].set_ylabel(r'Position $(x)$ $[m]$')
    ax[0].set_xlim([0,7.0])
    ax[0].grid(True)
    ax[0].legend(bbox_to_anchor=(-0.1,1.02), loc='lower left', ncol=3)
    # ax[0].legend(loc='upper left')
    # ax[0].legend(bbox_to_anchor=(0.7,0.6), loc='lower left')

    ax[1].plot(t, pose_end_effector_pred[::5,1], label='Model (sparsified strains)', linewidth=2, color=color_palette[1], marker='s', mec='k', markevery=100)
    ax[1].plot(t, true_poses[:,-1,1], label='GT', linewidth=3.5, linestyle='dotted', color='k', alpha=0.55)
    ax[1].set_ylabel(r'Position $(y)$ $[m]$')
    ax[1].set_xlim([0,7.0])
    ax[1].grid(True)
    # ax[1].legend(loc='upper left')

    ax[2].plot(t, pose_end_effector_pred[::5,2], label='Model (sparsified strains)', linewidth=2, color=color_palette[1], marker='s', mec='k', markevery=100)
    ax[2].plot(t, true_poses[:,-1,2], label='GT', linewidth=3, linestyle='dotted', color='k', alpha=0.55)
    ax[2].set_ylabel(r'Orientation $(\theta)$ $[rad]$')
    ax[2].set_xlim([0,7.0])
    ax[2].grid(True)

    plt.xlabel('Time [s]')
    plt.show()
    plt.savefig(f'results/ns-{num_segments}_high_stiffness/{validation_type}/ns-{num_segments}_ee_comparison_plots.pdf', bbox_inches='tight')

else:
    q_true = np.load(f'./data/ns-{num_segments}/{validation_type}/ns-{num_segments}_q_true.npy').T
    q_pred = np.load(f'./data/ns-{num_segments}/{validation_type}/ns-{num_segments}_q_pred.npy').T
    q_pred_noise = np.load(f'./data/ns-{num_segments}_noise/{validation_type}/ns-{num_segments}_q_pred.npy').T
    config_data_true = np.zeros((q_true.shape[0], num_segments, 3))
    config_data_pred = np.zeros((q_pred.shape[0], num_segments, 3))
    config_data_pred_noise = np.zeros((q_pred.shape[0], num_segments, 3))
    for i in range(num_segments):
        config_data_pred[:,i,:] = q_pred[:,(3*i):(3*i+3)]
        config_data_true[:,i,:] = q_true[:,(3*i):(3*i+3)]
        config_data_pred_noise[:,i,:] = q_pred_noise[:,(3*i):(3*i+3)]

    true_poses = np.load(f'./data/ns-{num_segments}/{validation_type}/ns-{num_segments}_true_poses.npy')
    true_poses = np.transpose(true_poses, (0,2,1))

    pose_end_effector_true = compute_end_effector_poses(config_data_true, params['l'], eps)
    pose_end_effector_pred = compute_end_effector_poses(config_data_pred, params['l'], eps)
    pose_end_effector_pred_noise = compute_end_effector_poses(config_data_pred_noise, params['l'], eps)

    error_position = np.linalg.norm(pose_end_effector_true[:,:2] - pose_end_effector_pred[:,:2], axis=1)
    print('End-effector errors - no noise')
    print('\tmean position error: ' + str(np.mean(error_position)) + ' [m]')
    error_orientation = np.abs(pose_end_effector_true[:,2] - pose_end_effector_pred[:,2])
    print('\tmean angle error: ' + str(np.mean(error_orientation)*180/np.pi) + ' [deg]')

    error_position_noise = np.linalg.norm(pose_end_effector_true[:,:2] - pose_end_effector_pred_noise[:,:2], axis=1)
    print('End-effector errors - with noise')
    print('\tmean position error: ' + str(np.mean(error_position_noise)) + ' [m]')
    error_orientation_noise = np.abs(pose_end_effector_true[:,2] - pose_end_effector_pred_noise[:,2])
    print('\tmean angle error: ' + str(np.mean(error_orientation_noise)*180/np.pi) + ' [deg]')

    t = np.arange(0.0, 7.0, 5e-3)
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(t, pose_end_effector_pred[::5,0], label='Model (no noise)', linewidth=2, color=color_palette[1], marker='s', mec='k', markevery=100)
    ax[0].plot(t, pose_end_effector_pred_noise[::5,0], label='Model (with noise)', linewidth=2, color=color_palette[2], marker='^', mec='k', markevery=100)
    ax[0].plot(t, true_poses[:,-1,0], label='GT', linewidth=3.5, linestyle='dotted', color='k', alpha=0.55)
    ax[0].set_ylabel(r'Position $(x)$ $[m]$')
    ax[0].set_xlim([0,7.0])
    ax[0].grid(True)
    ax[0].legend(bbox_to_anchor=(-0.1,1.02), loc='lower left', ncol=3)
    # ax[0].legend(loc='upper left')
    # ax[0].legend(bbox_to_anchor=(0.7,0.6), loc='lower left')

    ax[1].plot(t, pose_end_effector_pred[::5,1], label='Model (no noise)', linewidth=2, color=color_palette[1], marker='s', mec='k', markevery=100)
    ax[1].plot(t, pose_end_effector_pred_noise[::5,1], label='Model (with noise)', linewidth=2, color=color_palette[2], marker='^', mec='k', markevery=100)
    ax[1].plot(t, true_poses[:,-1,1], label='GT', linewidth=3.5, linestyle='dotted', color='k', alpha=0.55)
    ax[1].set_ylabel(r'Position $(y)$ $[m]$')
    ax[1].set_xlim([0,7.0])
    ax[1].grid(True)
    # ax[1].legend(loc='upper left')

    ax[2].plot(t, pose_end_effector_pred[::5,2], label='Model (no noise)', linewidth=2, color=color_palette[1], marker='s', mec='k', markevery=100)
    ax[2].plot(t, pose_end_effector_pred_noise[::5,2], label='Model (with noise)', linewidth=2, color=color_palette[2], marker='^', mec='k', markevery=100)
    ax[2].plot(t, true_poses[:,-1,2], label='GT', linewidth=3, linestyle='dotted', color='k', alpha=0.55)
    ax[2].set_ylabel(r'Orientation $(\theta)$ $[rad]$')
    ax[2].set_xlim([0,7.0])
    ax[2].grid(True)
    # ax[2].legend(loc='upper right')
    # plt.legend(bbox_to_anchor=(0,1.02), loc='lower left')

    plt.xlabel('Time [s]')
    plt.show()
    plt.savefig(f'results/ns-{num_segments}/{validation_type}/ns-{num_segments}_ee_comparison_plots.pdf', bbox_inches='tight')

    t_fine = np.arange(0.0, 7.0, 1e-3)
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t_fine, error_position, label='Model (no noise)', linewidth=2, color=color_palette[1], marker='s', mec='k', markevery=400)
    # ax[0].plot(t_fine, error_position_noise, label='Model (with noise)', linewidth=2, color=color_palette[2], marker='^', mec='k', markevery=400)
    ax[0].set_ylabel(r'Position error $[m]$')
    ax[0].set_xlim([0,7.0])
    ax[0].grid(True)
    # ax[0].legend(loc='upper right')
    ax[0].legend(bbox_to_anchor=(-0.1,1.02), loc='lower left', ncol=2)

    ax[1].plot(t_fine, error_orientation, label='Model (no noise)', linewidth=2, color=color_palette[1], marker='s', mec='k', markevery=400)
    # ax[1].plot(t_fine, error_orientation_noise, label='Model (with noise)', linewidth=2, color=color_palette[2], marker='^', mec='k', markevery=400)
    ax[1].set_ylabel(r'Orientation error $[rad]$')
    ax[1].set_xlim([0,7.0])
    ax[1].grid(True)
    # ax[1].legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.show()
    plt.savefig(f'results/ns-{num_segments}/{validation_type}/ns-{num_segments}_ee_error_plots.pdf', bbox_inches='tight')
