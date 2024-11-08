import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Plotting settings
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=10)

color_palette = ['tab:blue', 'tab:orange', 'tab:green']
validation_type = 'sinusoidal_actuation' # sinusoidal_actuation or step_actuation
high_shear_stiffness = True
high_stiffness = False
num_segments = 1
if high_shear_stiffness == True:
    q_true = np.load(f'./data/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_q_true.npy')
    q_pred = np.load(f'./data/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_q_pred.npy')
elif high_stiffness == True:
    q_true = np.load(f'./data/ns-{num_segments}_high_stiffness/{validation_type}/ns-{num_segments}_q_true.npy')
    q_pred = np.load(f'./data/ns-{num_segments}_high_stiffness/{validation_type}/ns-{num_segments}_q_pred.npy')
else:
    q_true = np.load(f'./data/ns-{num_segments}/{validation_type}/ns-{num_segments}_q_true.npy')
    q_pred = np.load(f'./data/ns-{num_segments}/{validation_type}/ns-{num_segments}_q_pred.npy')
    q_pred_noise = np.load(f'./data/ns-{num_segments}_noise/{validation_type}/ns-{num_segments}_q_pred.npy')

t = np.arange(0.0, 7.0, 1e-3)

if high_shear_stiffness == True:
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(3.8,3.8))
    ax[0].plot(t, q_pred[0,:], label='Model (wo/ shear)', linewidth=2, color=color_palette[1])#, marker='s', mec='k', markevery=400)
    ax[0].plot(t, q_true[0,:], label='GT', linewidth=3.5, linestyle='dotted', color='k', alpha=0.6)
    ax[0].set_ylabel(r'$\kappa_{be}$ $[m^{-1}]$', fontsize=12)
    ax[0].set_xlim([0,7.0])
    ax[0].grid(True)
    # ax[0].legend(loc='lower left', ncol=2)
    # ax[0].legend(bbox_to_anchor=(-0.1,1.02), loc='lower left', ncol=2)
    ax[0].legend(loc='upper right', fontsize=10, framealpha=0.6)

    ax[1].plot(t, q_pred[1,:], label='Model (wo/ shear)', linewidth=2, color=color_palette[1])#, marker='s', mec='k', markevery=400)
    ax[1].plot(t, q_true[1,:], label='GT', linewidth=3.5, linestyle='dotted', color='k', alpha=0.6)
    ax[1].set_ylabel(r'$\sigma_{ax}$ [-]', fontsize=12)
    ax[1].set_xlim([0,7.0])
    ax[1].grid(True)
    # ax[1].legend(loc='upper left', ncol=2)

    plt.xlabel('Time [s]', fontsize=12)
    # fig.set_size_inches(3.5, 3.5 / 1.618 )
    plt.show()
    plt.savefig(f'results/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_configuration_plots.pdf', bbox_inches='tight')
elif high_stiffness == True:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(3.8,3.8))
    ax[0].plot(t, q_pred[0,:], label='Seg. 1 Model', linewidth=2, color=color_palette[0], alpha=0.7)#, marker='s', mec='k', markevery=400, alpha=0.7)
    ax[0].plot(t, q_true[0,:], label='Seg. 1 GT', linewidth=3.5, linestyle='dotted', color=color_palette[0])
    ax[0].plot(t, q_pred[2,:], label='Seg. 2 Model', linewidth=2, color=color_palette[1], alpha=0.7)#, marker='s', mec='k', markevery=400, alpha=0.7)
    ax[0].plot(t, q_true[2,:], label='Seg. 2 GT', linewidth=3.5, linestyle='dotted', color=color_palette[1])
    ax[0].set_ylabel(r'$\kappa_{be}$ $[m^{-1}]$', fontsize=12)
    ax[0].set_xlim([0,7.0])
    ax[0].grid(True)
    # ax[0].legend(loc='lower left', ncol=2)
    # ax[0].legend(bbox_to_anchor=(-0.1,1.02), loc='lower left', ncol=2, fontsize=12)
    ax[0].legend(loc='upper right', fontsize=8, framealpha=0.6)

    ax[1].plot(t, q_pred[1,:], label='Seg. 1 Model', linewidth=2, color=color_palette[0], alpha=0.7)#, marker='s', mec='k', markevery=400)
    ax[1].plot(t, q_true[1,:], label='GT', linewidth=3.5, linestyle='dotted', color=color_palette[0])
    ax[1].set_ylabel(r'$\sigma_{sh}$ [-]', fontsize=12)
    ax[1].set_xlim([0,7.0])
    ax[1].grid(True)
    # ax[1].legend(loc='upper left', ncol=2)

    ax[2].plot(t, q_pred[3,:], label='Seg. 2 Model', linewidth=2, color=color_palette[1], alpha=0.7)#, marker='s', mec='k', markevery=400)
    ax[2].plot(t, q_true[3,:], label='GT', linewidth=3.5, linestyle='dotted', color=color_palette[1])
    ax[2].set_ylabel(r'$\sigma_{ax}$ [-]', fontsize=12)
    ax[2].set_xlim([0,7.0])
    ax[2].grid(True)

    plt.xlabel('Time [s]', fontsize=12)
    # fig.set_size_inches(3.5, 3.5 / 1.618 )
    plt.show()
    plt.savefig(f'results/ns-{num_segments}_high_stiffness/{validation_type}/ns-{num_segments}_configuration_plots.pdf', bbox_inches='tight')
else:
    fig, ax = plt.subplots(3, 1, sharex=True)
    for i in range(num_segments):
        if num_segments <= 1:
            ax[0].plot(t, q_pred[3*i,:], label='Model (no noise)', linewidth=2, color=color_palette[1], marker='s', mec='k', markevery=400)
            ax[0].plot(t, q_pred_noise[3*i,:], label='Model (with noise)', linewidth=2, color=color_palette[2], marker='^', mec='k', markevery=400)
            ax[0].plot(t, q_true[3*i,:], label='GT', linewidth=3.5, linestyle='dotted', color='k', alpha=0.6)
        else:
            ax[0].plot(t, q_pred[3*i,:], label=f'Seg. {i+1} Model (no noise)', linewidth=2, color=color_palette[i], marker='s', mec='k', markevery=400, alpha=0.5)
            ax[0].plot(t, q_pred_noise[3*i,:], label=f'Seg. {i+1} Model (w/ noise)', linewidth=2, color=color_palette[i], marker='^', mec='k', markevery=400, alpha=0.8)
            ax[0].plot(t, q_true[3*i,:], label=f'Seg. {i+1} GT', linewidth=4, linestyle='dotted', color=color_palette[i])
        ax[0].set_ylabel(r'$\kappa_{be}$ $[m^{-1}]$')
        ax[0].set_xlim([0,7.0])
        ax[0].grid(True)
        # ax[0].legend(loc='lower left', ncol=2)
        ax[0].legend(bbox_to_anchor=(-0.15,1.02), loc='lower left', ncol=3)

        if num_segments <= 1:
            ax[1].plot(t, q_pred[3*i+1,:], label='Model (no noise)', linewidth=2, color=color_palette[1], marker='s', mec='k', markevery=400)
            ax[1].plot(t, q_pred_noise[3*i+1,:], label='Model (with noise)', linewidth=2, color=color_palette[2], marker='^', mec='k', markevery=400)
            ax[1].plot(t, q_true[3*i+1,:], label='GT', linewidth=3.5, linestyle='dotted', color='k', alpha=0.6)
        else:
            ax[1].plot(t, q_pred[3*i+1,:], label=f'Seg. {i+1} Model (no noise)', linewidth=2, color=color_palette[i], marker='s', mec='k', markevery=400, alpha=0.5)
            ax[1].plot(t, q_pred_noise[3*i+1,:], label=f'Seg. {i+1} Model (w/ noise)', linewidth=2, color=color_palette[i], marker='^', mec='k', markevery=400, alpha=0.8)
            ax[1].plot(t, q_true[3*i+1,:], label=f'GT (Seg. {i+1})', linewidth=4, linestyle='dotted', color=color_palette[i])
        ax[1].set_ylabel(r'$\sigma_{sh}$ [-]')
        ax[1].set_xlim([0,7.0])
        ax[1].grid(True)
        # ax[1].legend(loc='lower left', ncol=2)

        if num_segments <= 1:
            ax[2].plot(t, q_pred[3*i+2,:], label='Model (no noise)', linewidth=2.0, color=color_palette[1], marker='s', mec='k', markevery=400)
            ax[2].plot(t, q_pred_noise[3*i+2,:], label='Model (with noise)', linewidth=2.0, color=color_palette[2], marker='^', mec='k', markevery=400)
            ax[2].plot(t, q_true[3*i+2,:], label='GT', linewidth=3.5, linestyle='dotted', color='k', alpha=0.6)
        else:
            ax[2].plot(t, q_pred[3*i+2,:], label=f'Seg. {i+1} Model (no noise)', linewidth=2, color=color_palette[i], marker='s', mec='k', markevery=400, alpha=0.5)
            ax[2].plot(t, q_pred_noise[3*i+2,:], label=f'Seg. {i+1} Model (w/ noise)', linewidth=2, color=color_palette[i], marker='^', mec='k', markevery=400, alpha=0.8)
            ax[2].plot(t, q_true[3*i+2,:], label=f'GT (Seg. {i+1})', linewidth=4, linestyle='dotted', color=color_palette[i])
        ax[2].set_ylabel(r'$\sigma_{ax}$ [-]')
        ax[2].set_xlim([0,7.0])
        ax[2].grid(True)
        # ax[2].legend(loc='lower left', ncol=2)

    plt.xlabel('Time [s]')
    # fig.set_size_inches(3.5, 3.5 / 1.618 )
    plt.show()
    plt.savefig(f'results/ns-{num_segments}/{validation_type}/ns-{num_segments}_configuration_plots.pdf', bbox_inches='tight')
