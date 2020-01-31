from sympy.physics.mechanics import *
from sympy import symbols, cos, sin, expand, Matrix, ImmutableDenseMatrix
from sympy import Dummy, lambdify, diff, pprint
from numpy import array, hstack, zeros, linspace, pi, ones, arange, around, ndarray
import numpy as np
from numpy.linalg import solve
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import math
from matplotlib import animation
import time
from sympy.utilities.iterables import flatten

class Swimmer:

    ##################################################
    #################  Constructor  ##################
    ##################################################
    def __init__(self, n, link_length = 0.3, k_val = 1):
        
        l = symbols('l')  #the link length
        k = symbols('k')  #the liquid vicousness
        q = symbols('q:' + str(n - 1)) #the joint angle


        #l and k are useless now

        assert(n==3)


        rho=1
        a=4
        b=1

        demonimator = (np.pi**3*rho**3*(-25*a**8 - 186*a**6*b**2 - 218*a**4*b**4 - 42*a**2*b**6 - 9*b**8 + (a - b)*(a + b)*(19*a**6 + 63*a**4*b**2 + a**2*b**4 - 3*b**6)*cos(2*q[1]) - 
               (a - b)*(a + b)*cos(2*q[0])*(-19*a**6 - 63*a**4*b**2 - a**2*b**4 + 3*b**6 + (13*a**6 - 31*a**4*b**2 - a**2*b**4 + 3*b**6)*cos(2*q[1])) + 
               32*b**2*cos(q[1])*(-9*a**4*b**2*cos(q[0])**2 - (2*a**3 + a*b**2)**2*sin(q[0])**2) + 32*a**2*b**4*(2*a**2 + b**2)*sin(q[0])*sin(q[1]) + 16*a**2*b**2*(-2*a**4 + a**2*b**2 + b**4)*sin(q[0])*sin(2*q[1]) + 
               2*cos(q[0])*(-8*a**2*b**2*(4*a**4 + 13*a**2*b**2 + b**4) - 48*a**4*b**4*cos(q[1]) + 
                  (a - b)*(a + b)*(8*a**2*b**2*((4*a**2 - b**2)*cos(2*q[1]) - 2*(2*a**2 + b**2)*sin(q[0])*sin(q[1])) + (-3*a**6 + a**4*b**2 - 17*a**2*b**4 + 3*b**6)*sin(q[0])*sin(2*q[1])))))/8.


        nominator = ImmutableDenseMatrix(
            np.array([
                [(a*np.pi**3*rho**3*(2*b**2*(19*a**6 + 37*a**4*b**2 + 9*a**2*b**4 + 3*b**6)*sin(q[0]) + (a**2 + b**2)*(a**2 + 2*b**2)*(a**4 + 4*a**2*b**2 - b**4)*sin(2*q[0]) - 16*a**6*b**2*sin(q[0] - 2*q[1]) + 
                   8*a**4*b**4*sin(q[0] - 2*q[1]) + 8*a**2*b**6*sin(q[0] - 2*q[1]) + 48*a**4*b**4*sin(q[0] - q[1]) + 24*a**2*b**6*sin(q[0] - q[1]) - 3*a**6*b**2*sin(q[1]) - 21*a**4*b**4*sin(q[1]) - 
                   13*a**2*b**6*sin(q[1]) - 3*b**8*sin(q[1]) + a**8*sin(2*q[1]) + 7*a**6*b**2*sin(2*q[1]) - 3*a**4*b**4*sin(2*q[1]) - 3*a**2*b**6*sin(2*q[1]) - 2*b**8*sin(2*q[1]) + 16*a**4*b**4*sin(q[0] + q[1]) + 
                   8*a**2*b**6*sin(q[0] + q[1]) - a**8*sin(2*(q[0] + q[1])) - 4*a**6*b**2*sin(2*(q[0] + q[1])) + 2*a**4*b**4*sin(2*(q[0] + q[1])) + 4*a**2*b**6*sin(2*(q[0] + q[1])) - b**8*sin(2*(q[0] + q[1])) + 
                   a**6*b**2*sin(2*q[0] + q[1]) + 5*a**4*b**4*sin(2*q[0] + q[1]) + 3*a**2*b**6*sin(2*q[0] + q[1]) - b**8*sin(2*q[0] + q[1]) + 2*b**2*(-9*a**6 + 7*a**4*b**2 - 3*a**2*b**4 + b**6)*sin(q[0] + 2*q[1])))/8.,
                  (a*np.pi**3*rho**3*(b**2*(3*a**6 + 21*a**4*b**2 + 13*a**2*b**4 + 3*b**6 + 16*a**2*b**2*(2*a**2 + b**2)*cos(q[1]) - (a**2 + b**2)*(a**4 + 4*a**2*b**2 - b**4)*cos(2*q[1]))*sin(q[0]) + 
                       (-a**8 - 7*a**6*b**2 + 3*a**4*b**4 + 3*a**2*b**6 + 2*b**8 + 2*b**2*(a**6 - 3*a**4*b**2 + 7*a**2*b**4 - b**6)*cos(q[1]) + (a**8 + 4*a**6*b**2 - 2*a**4*b**4 - 4*a**2*b**6 + b**8)*cos(2*q[1]))*sin(2*q[0]) - 
                       2*b**2*(19*a**6 + 37*a**4*b**2 + 9*a**2*b**4 + 3*b**6 + 16*a**2*b**2*(2*a**2 + b**2)*cos(q[0]) + (-17*a**6 + 11*a**4*b**2 + a**2*b**4 + b**6)*cos(2*q[0]))*sin(q[1]) + 
                       (a**2 + b**2)*(-a**4 - 4*a**2*b**2 + b**4)*(a**2 + 2*b**2 + b**2*cos(q[0]) + (-a**2 + b**2)*cos(2*q[0]))*sin(2*q[1])))/8.],
                [(a*np.pi**3*rho**3*(-24*a**4*b**4 - (a**2 + b**2)*(2*a**2 + b**2)*(a**4 + 4*a**2*b**2 - b**4)*cos(2*q[0]) + 
                        b**2*(-5*a**6 - 31*a**4*b**2 - 3*a**2*b**4 - b**6 - (a**2 + b**2)*(a**4 + 4*a**2*b**2 - b**4)*cos(2*q[0]))*cos(q[1]) + (a - b)*(a + b)*(2*a**6 + 13*a**4*b**2 + b**6)*cos(2*q[1]) + 
                        2*b**2*cos(q[0])*(-21*a**6 - 43*a**4*b**2 - 3*a**2*b**4 - b**6 - 72*a**4*b**2*cos(q[1]) + (15*a**6 - 17*a**4*b**2 - 3*a**2*b**4 + b**6)*cos(2*q[1])) + 
                        b**2*(a**2 + b**2)*(a**4 + 4*a**2*b**2 - b**4)*sin(2*q[0])*sin(q[1]) - 2*b**2*(-9*a**6 + 7*a**4*b**2 - 3*a**2*b**4 + b**6)*sin(q[0])*sin(2*q[1])))/8.,
                   (a*np.pi**3*rho**3*(-24*a**4*b**4 + (a - b)*(a + b)*(2*a**6 + 13*a**4*b**2 + b**6)*cos(2*q[0]) + 
                        2*b**2*(-21*a**6 - 43*a**4*b**2 - 3*a**2*b**4 - b**6 + (15*a**6 - 17*a**4*b**2 - 3*a**2*b**4 + b**6)*cos(2*q[0]))*cos(q[1]) - (a**2 + b**2)*(2*a**2 + b**2)*(a**4 + 4*a**2*b**2 - b**4)*cos(2*q[1]) + 
                        b**2*cos(q[0])*(-5*a**6 - 31*a**4*b**2 - 3*a**2*b**4 - b**6 - 144*a**4*b**2*cos(q[1]) - (a**2 + b**2)*(a**4 + 4*a**2*b**2 - b**4)*cos(2*q[1])) - 
                        2*b**2*(-9*a**6 + 7*a**4*b**2 - 3*a**2*b**4 + b**6)*sin(2*q[0])*sin(q[1]) + b**2*(a**2 + b**2)*(a**4 + 4*a**2*b**2 - b**4)*sin(q[0])*sin(2*q[1])))/8.],
                [(np.pi**3*rho**3*(-3*a**8 - 30*a**6*b**2 - 46*a**4*b**4 - 14*a**2*b**6 - 3*b**8 + 2*(a**8 + 4*a**6*b**2 - 2*a**4*b**4 - 4*a**2*b**6 + b**8)*cos(2*q[0])*cos(q[1])**2 + 
                        (a**2 - b**2)**2*(a**4 + 6*a**2*b**2 + b**4)*cos(2*q[1]) - 16*a**2*b**2*(2*a**2 + b**2)*(-b**2 + (a - b)*(a + b)*cos(q[1]))*sin(q[0])*sin(q[1]) + 
                        2*cos(q[0])*(-24*a**4*b**4*cos(q[1])*(1 + 3*cos(q[1])) - 8*a**2*b**2*(2*a**2 + b**2)**2*sin(q[1])**2 - (a**8 + 4*a**6*b**2 - 2*a**4*b**4 - 4*a**2*b**6 + b**8)*sin(q[0])*sin(2*q[1]))))/8.,
                   (np.pi**3*rho**3*(3*a**8 + 30*a**6*b**2 + 46*a**4*b**4 + 14*a**2*b**6 + 3*b**8 - (a**2 - b**2)**2*(a**4 + 6*a**2*b**2 + b**4)*cos(2*q[0]) + 
                        8*a**2*b**2*(4*a**4 + 13*a**2*b**2 + b**4 + 6*a**2*b**2*cos(q[0]) - (4*a**4 - 5*a**2*b**2 + b**4)*cos(2*q[0]))*cos(q[1]) - 
                        2*(a**8 + 4*a**6*b**2 - 2*a**4*b**4 - 4*a**2*b**6 + b**8)*cos(q[0])**2*cos(2*q[1]) + 16*a**2*b**2*(2*a**2 + b**2)*(-b**2 + (a - b)*(a + b)*cos(q[0]))*sin(q[0])*sin(q[1]) + 
                        (a**8 + 4*a**6*b**2 - 2*a**4*b**4 - 4*a**2*b**6 + b**8)*sin(2*q[0])*sin(2*q[1])))/8.]
           ])
        )

        
        A = nominator / demonimator



        #initialize self objects
        self.l = l
        self.k = k
        self.q = q
        self.n = n              #the number of links


        self.link_length = link_length
        self.k_val = k_val

        #set the value for all the variables
        parameters = (self.k, self.l)
        self.parameter_vals = [self.k_val, self.link_length]
        #this can potentially be improved, probably through substituting in this function
        

        self.A_func = lambdify(q, A)




    ####################################################################
    ######################  get the actualy motion  ####################
    ####################################################################
    #return the generated y and corresponding t
    def generate_trajectories(self, init, dq1, dq2, t_val = [0, 50, 1000]):
        '''generate the trajectories of a snake robot's motion given an initial canodition and joints input

        Parameters
        ----------
        a list of initial conditions,
        q1 amplitude,
        q1 frequency,
        time interval

        x0: x, y, t, q1, q2

        
        Returns
        ----------
        y, t
        '''
        self.dq1 = dq1
        self.dq2 = dq2

        x0 = init                                                               # Initial conditions, q and u
        t = linspace(t_val[0], t_val[1], t_val[2])                              # Time vector
        y = odeint(self.right_hand_side, x0, t)                                 # Actual integration

        return y, t



    def generate_init_value(self, q1_init = 0, q2_init = 0):
        return hstack([0 , 0 , 0, q1_init, q2_init])
    

    '''
    Helper Function
    '''
    def right_hand_side(self, x, t):
        """Returns the derivatives of the states.

        Parameters
        ----------
        x : ndarray, shape(2 * (n + 1))
            The current state vector.
        t : float
            The current time.
        args : ndarray
            The constants.

        Returns
        -------
        dx : ndarray, shape(2 * (n + 1))
            The derivative of the state.
        
        """
        arguments = x[3:]    # States, input, and parameters


        A = self.A_func(*arguments) # Solving for the derivatives

        #print(f"A is {A}")
        #we should probably also calculate the acceleration of q here
        # dq1 = self.q1_amp*self.q1_freq*np.sin(self.q1_freq*t)
        # dq2 = self.q2_amp*self.q2_freq*np.cos(self.q2_freq*t)

        dx = np.matmul(-A,[self.dq1,self.dq2]) 

        #transform it to global frame
        theta = x[2]
        rotation_matrix = [[np.cos(theta), -np.sin(theta), 0], 
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]]
        dx = np.matmul(rotation_matrix,dx)

        return np.concatenate([dx, [self.dq1,self.dq2]])

    def plot_image(self, y,t, show_pos = True, show_vel = True):
        '''visualize the trajectory'''
        num = int(y.shape[1] / 2)

        plt.gca().set_aspect('equal', adjustable='box')

        lines = plt.plot(y[:, 0], y[:, 1])
        lab = plt.xlabel('x')
        leg = plt.legend('y')
        plt.show()

        if(show_pos):
            #for x,y,t
            lines = plt.plot(t, y[:, :3])
            lab = plt.xlabel('Time [sec]')
            leg = plt.legend(["x", "y", "theta"])
            plt.show()
        
            #for all the q
            lines = plt.plot(t, y[:, 3:self.n+2])
            lab = plt.xlabel('Time [sec]')
            leg = plt.legend(['q1', 'q2'])
            plt.show()

        #currently we are not capable of keeping track of the acceleration, as this may not be continuous
        # if(show_vel):
        #     #for x' y' t'
        #     lines = plt.plot(t, y[:, self.n+2:self.n+5])
        #     lab = plt.xlabel('Time [sec]')
        #     leg = plt.legend(self.dynamic[self.n+2:self.n+5])
        #     plt.show()

        #     #for all the u
        #     lines = plt.plot(t, y[:, self.n+5:])
        #     lab = plt.xlabel('Time [sec]')
        #     leg = plt.legend(self.dynamic[self.n+5:])
        #     plt.show()

    #switch the physical parameter of our snake
    #don't switch for fields = None
    #this will need to be changed if the numpy array is making the performance weird
    def switch_param(self, mass = None, k_val = None, link_length = None):

        #first, process the mass, k_val, and link_length
        if mass is not None:
            if(type(mass)!=list and type(mass)!=np.ndarray):
                mass = [mass]*self.n
            self.mass = mass

        if link_length is not None:
            if(type(link_length)!=list and type(mass)!=np.ndarray):
                link_length = [link_length]*self.n
            self.link_length = link_length
        if k_val is not None:
            if(type(k_val)!=list and type(mass)!=np.ndarray):
                k_val = [k_val]*(self.n-1)
            self.k_val = k_val

        self.parameter_vals = [self.mass[0], self.link_length[0]]
        for i in range(self.n-1):
            self.parameter_vals+=[self.mass[i+1],self.link_length[i+1],self.k_val[i]]

        #print(f"our current parameters are {self.parameter_vals}")

    ####################################################################
    #################### currently not functional  #####################
    ####################################################################
    def animate_snake(self, t, states, length, filename=None):
        """Animates the n-pendulum and optionally saves it to file.

        Parameters
        ----------
        t : ndarray, shape(m)
            Time array.
        states: ndarray, shape(m,p)
            State time history.
        length: float
            The length of the links.
        filename: string or None, optional
            If true a movie file will be saved of the animation. This may take some time.

        Returns
        -------
        fig : matplotlib.Figure
            The figure.
        anim : matplotlib.FuncAnimation
            The animation.

            """
        # the number of pendulum bobs
        # n = 3

        numpoints = int(states.shape[1] / 2 - 2)

        # first set up the figure, the axis, and the plot elements we want to animate
        fig = plt.figure()
        
        # set the limits based on the motion
        xmin = around(states[:, 2].min()-0.5, 1)
        xmax = around(states[:, 2].max()+0.5, 1)
        ymin = around(states[:, 3].min()-0.5, 1)
        ymax = around(states[:, 3].max()+0.5, 1)
        
        # create the axes
        ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect='equal')
        
        # display the current time
        time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)

        # blank line for the pendulum
        line, = ax.plot([], [], lw=2, marker='o', markersize=6)

        # initialization function: plot the background of each frame
        def init():
            time_text.set_text('')
            line.set_data([], [])
            return time_text, line,

        # animation function: update the objects
        def animate(i):
            time_text.set_text('time = {:2.2f}'.format(t[i]))
            x = zeros((numpoints+1))
            y = zeros((numpoints+1))
            theta = states[i, 4]
            x[0] = states[i, 2] - math.cos(theta)*length / 2.0
            y[0] = states[i, 3] - math.sin(theta)*length / 2.0
            x[1] = states[i, 2] + math.cos(theta)*length / 2.0
            y[1] = states[i, 3] + math.sin(theta)*length / 2.0

            for j in arange(numpoints-1):
                theta = theta + states[i, j]
                x[j+2] = x[j + 1] + length * math.cos(theta)
                y[j+2] = y[j + 1] + length * math.sin(theta)
                line.set_data(x, y)
            return time_text, line,


        # call the animator function
        anim = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
            interval=t[-1] / len(t) * 1000, blit=True, repeat=False)
        
        # save the animation if a filename is given
        if filename is not None:
            anim.save(filename, fps=30, codec='libx264')


####################################################################
################### calling the functions above ####################
####################################################################


if __name__ == "__main__":
    swimmer_test = Swimmer(3, link_length = 0.3, k_val = 1)
    #generate initial condition
    init_cond = swimmer_test.generate_init_value(q1_init = 0.2, q2_init = -0.2)

    #generate trajectory
    y,t = swimmer_test.generate_trajectories(init_cond, 0.2, -0.2, t_val = [0, 2, 1000])

    #plot the trajectory
    swimmer_test.plot_image(y, t, show_pos = True, show_vel = False)


    # set a joint limit, convert the action
    # add prev_action to observation