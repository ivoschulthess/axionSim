import numpy as np
from multiprocessing import Pool

from scipy.constants import physical_constants

from IPython.display import clear_output
from time import sleep


##################################
## PHYSICAL CONSTANTS
##################################

# just the normal pi = 3.1415
pi = np.pi
# gyromagnetic ratio of the neutron in [rad/s/T]
gamma_n = physical_constants['neutron gyromag. ratio'][0]
# Plack constant in [m**2*kg/s]
h = physical_constants['Planck constant'][0]
# neutron mass in [kg]
mass_n = physical_constants['neutron mass'][0]

# spin up representation vector
spinUp = np.array([[1],[0]])
# spin down representation vector
spinDn = np.array([[0],[1]])


##################################
## HELPER FUNCTIONS
##################################

class dotdict(dict):
    
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def track_job(job, update_interval=2):
    # total number of jobs
    jobLength = job._number_left
    while job._number_left > 0:
        print('\rTasks (Chunks) remaining = {0}({1}) / {2}({3})     '.format(
        job._number_left*job._chunksize,job._number_left,jobLength*job._chunksize,jobLength), end='')
        sleep(update_interval)
    clear_output()
    print('Task finished')

def sinFct(t, frequency, amplitude, phase, offset):
    '''
    simple sinusoidal function
    
    PARAMETERS
    -----------------
    
    t:          time array in seconds
    frequency:  frequency in Hz
    amplitude:  amplitude of sinusoidal function
    phase:      phase in degrees
    offset:     offset of the sinusoidal function
    
    RETURN
    -----------------
    
    sin : 1d-array
        sinusoidal signal
        amplitude * np.sin(2*np.pi*frequency*t - phase*np.pi/180) + offset
    
    '''
    return amplitude * np.sin(2*np.pi*frequency*t - phase*np.pi/180) + offset

def resonantCancelation(f, B, t_int, gamma=gamma_n): 
    '''
    resonant cancelation of the signal if an oscillating B0 field is added in a Ramsey setup
    
    PARAMETERS
    ----------------------------
    f : 1d-array
        frequency (x-axis)
    B : float
        amplitude
    t_int : float
        interaction time
    gamma : float
        gyromagnetic ratio of the particle in [rad/s/T]
        
    RETURN
    ----------------------------
    gamma * B / np.pi / f * np.sin(np.pi*f*t_int)
        
    '''
    return abs(gamma * B / np.pi / f * np.sin(np.pi*f*t_int))
    
    
##################################
## MATRIX FUNCTIONS
##################################

def matSpinflip(t0, t1, B0, B1, omega, theta=0.0, N=1000):
    '''
    calculates the interaction matrix of the SPIN-FLIPPER
    
    PARAMETERS
    ----------------------------
    t0 : float
        start of interaction in [s]
    t1 : float
        end of interaction in [s]
    B0 : float
        amplitude of the static magnetic field B0 in [T]
    B1 : float
        amplitude of the oscillating magnetic field B1 in [T]
    omega : float
        angular frequency of oscillating field in [rad/s]
    theta : float
        phase of oscillating field in [rad]
    N : int
        number of steps to calculate
        
    RETURN
    ----------------------------
    U : 2x2 matrix
        propagation matrix through spin-flipper
        
    '''
    
    # calculate the time step
    dt = (t1 - t0) / N
    
    # create the U1 matrix as the identity matrix
    U = np.identity(2, dtype=complex)
    
    # iterate over the time steps
    for i in np.arange(N):
        
        # create the time evolution matrix of this step
        U_step = np.zeros((2,2), dtype=complex)
        
        # basic calculation, assuming omega_y = 0
        omega_x = -gamma_n*B1*(np.cos(omega*(t0+i*dt)+theta)+np.cos(omega*(t0+(i+1)*dt)+theta))
        omega_y = 0
        omega_z = -gamma_n*B0
        
        Omega = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        
        # calculate the propagation matrix of this step
        U_step[0,0] = np.cos(Omega*dt/2) + 1.j*omega_z/Omega*np.sin(Omega*dt/2)
        U_step[0,1] = -1.j*omega_x/Omega*np.sin(Omega*dt/2)
        U_step[1,0] = -1.j*omega_x/Omega*np.sin(Omega*dt/2)
        U_step[1,1] = np.cos(Omega*dt/2) - 1.j*omega_z/Omega*np.sin(Omega*dt/2)

        U = U_step @ U
    
    return U

def matPrecession(t0, t1, B0, N=1000):
    '''
    calculates the interaction matrix of the LARMOR PRECESSION
    
    PARAMETERS
    ----------------------------
    t0 : float
        start of interaction in [s]
    t1 : float
        end of interaction in [s]
    B0 : float
        amplitude of the static magnetic field B0 [T]
    N : int
        number of steps to calculate
        
    RETURN
    ----------------------------
    U : 2x2 matrix
        propagation matrix through Larmor precession area
        
    '''
    
    # calculate the time step
    dt = (t1 - t0) / N
    
    # create the U1 matrix as the identity matrix
    U = np.identity(2, dtype=complex)
    
    # iterate over the time steps
    for i in np.arange(N):
        
        # create the time evolution matrix of this step
        U_step = np.zeros((2,2), dtype=complex)
        
        # basic calculation, assuming omega_y = 0, omega_x = 0
        omega_x = 0
        omega_y = 0
        omega_z = -gamma_n*B0
        
        Omega = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        
        # calculate the propagation matrix of this step
        U_step[0,0] = np.cos(Omega*dt/2) + 1.j*omega_z/Omega*np.sin(Omega*dt/2)
        U_step[1,1] = np.cos(Omega*dt/2) - 1.j*omega_z/Omega*np.sin(Omega*dt/2)

        U = U_step @ U
    
    return U


def matAxionPrecession(t0, t1, B0, BA, omegaA, thetaA, N=1000):
    '''
    calculates the interaction matrix of the LARMOR PRECESSION
    
    PARAMETERS
    ----------------------------
    t0 : float
        start of interaction in [s]
    t1 : float
        end of interaction in [s]
    B0 : float
        amplitude of the static magnetic field B0 [T]
    BA : float
        amplitude of the axionic magnetic field Ba in [T]
    omegaA : float
        angular frequency of the axionic field in [rad/s]
    thetaA : float
        phase of axionic field in [rad]
    N : int
        number of steps to calculate
        
    RETURN
    ----------------------------
    U : 2x2 matrix
        propagation matrix through Larmor precession area
        
    '''
    
    # calculate the time step
    dt = (t1 - t0) / N
    
    # create the U1 matrix as the identity matrix
    U = np.identity(2, dtype=complex)
    
    # iterate over the time steps
    for i in np.arange(N):
        
        # create the time evolution matrix of this step
        U_step = np.zeros((2,2), dtype=complex)
        
        # basic calculation, assuming omega_y = 0, omega_x = 0
        omega_x = 0
        omega_y = 0
        omega_z = -gamma_n*B0 -gamma_n*BA*(np.sin(omegaA*(t0+i*dt)+thetaA)+np.sin(omegaA*(t0+(i+1)*dt)+thetaA))/2.
        
        Omega = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        
        # calculate the propagation matrix of this step
        U_step[0,0] = np.cos(Omega*dt/2) + 1.j*omega_z/Omega*np.sin(Omega*dt/2)
        U_step[1,1] = np.cos(Omega*dt/2) - 1.j*omega_z/Omega*np.sin(Omega*dt/2)

        U = U_step @ U
    
    return U