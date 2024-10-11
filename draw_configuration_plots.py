import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Plotting settings
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=10)

validation_type = 'sinusoidal_actuation' # sinusoidal_actuation or step_actuation
high_shear_stiffness = False
num_segments = 1
if high_shear_stiffness == True:
    q_true = np.load(f'./data/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_q_true.npy')
    q_pred = np.load(f'./data/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_q_pred.npy')
else:
    q_true = np.load(f'./data/ns-{num_segments}/{validation_type}/ns-{num_segments}_q_true.npy')
    q_pred = np.load(f'./data/ns-{num_segments}/{validation_type}/ns-{num_segments}_q_pred.npy')

t = np.arange(0.0, 7.0, 1e-3)

if high_shear_stiffness == True:
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, q_true[0,:], label='GT', linewidth=3, linestyle='dotted')
    ax[0].plot(t, q_pred[0,:], label='Obtained Model', linewidth=2)
    ax[0].set_ylabel(r'$\kappa_{be}$ $[m^{-1}]$')
    ax[0].set_xlim([0,7.0])
    ax[0].grid(True)
    ax[0].legend(loc='lower left', ncol=2)

    ax[1].plot(t, q_true[1,:]*100, label='GT', linewidth=3, linestyle='dotted')
    ax[1].plot(t, q_pred[1,:]*100, label='Obtained Model', linewidth=2)
    ax[1].set_ylabel(r'$\sigma_{ax}$ [\%]')
    ax[1].set_xlim([0,7.0])
    ax[1].grid(True)
    ax[1].legend(loc='upper left', ncol=2)

    plt.xlabel('Time [s]')
    # fig.set_size_inches(3.5, 3.5 / 1.618 )
    plt.show()
else:
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(t, q_true[0,:], label='GT', linewidth=3, linestyle='dotted')
    ax[0].plot(t, q_pred[0,:], label='Obtained Model', linewidth=2)
    ax[0].set_ylabel(r'$\kappa_{be}$ $[m^{-1}]$')
    ax[0].set_xlim([0,7.0])
    ax[0].grid(True)
    ax[0].legend(loc='lower left', ncol=2)

    ax[1].plot(t, q_true[1,:]*100, label='GT', linewidth=3, linestyle='dotted')
    ax[1].plot(t, q_pred[1,:]*100, label='Obtained Model', linewidth=2)
    ax[1].set_ylabel(r'$\sigma_{sh}$ [\%]')
    ax[1].set_xlim([0,7.0])
    ax[1].grid(True)
    ax[1].legend(loc='lower left', ncol=2)

    ax[2].plot(t, q_true[2,:]*100, label='GT', linewidth=3, linestyle='dotted')
    ax[2].plot(t, q_pred[2,:]*100, label='Obtained Model', linewidth=2)
    ax[2].set_ylabel(r'$\sigma_{ax}$ [\%]')
    ax[2].set_xlim([0,7.0])
    ax[2].grid(True)
    ax[2].legend(loc='upper left', ncol=2)

    plt.xlabel('Time [s]')
    # fig.set_size_inches(3.5, 3.5 / 1.618 )
    plt.show()

if high_shear_stiffness == True:
    plt.savefig(f'results/ns-{num_segments}_high_shear_stiffness/{validation_type}/ns-{num_segments}_configuration_plots.pdf', bbox_inches='tight')
else:
    plt.savefig(f'results/ns-{num_segments}/{validation_type}/ns-{num_segments}_configuration_plots.pdf', bbox_inches='tight')
