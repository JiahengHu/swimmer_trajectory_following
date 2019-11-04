from sympy.physics.mechanics import *
from sympy import symbols, cos, sin, expand
from sympy import Dummy, lambdify
from numpy import array, hstack, zeros, linspace, pi, ones, arange, around, ndarray
import numpy as np
from numpy.linalg import solve
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import math
from matplotlib import animation
import time

debug_snake_time = False

class Snake_Robot:

    ##################################################
    #################  Constructor  ##################
    ##################################################
    def __init__(self, n, link_length = 0.3, mass = 0.01, k_val = 0.01):
    # link_length, mass and k_val can either be list or value
    # but eventually they will be turned into list of correct length
    # if they are list, you have to make sure they are of correct length
        l = symbols('l:' + str(n))  #the link length
        m = symbols('m:' + str(n))  #the mass of the object
        k = symbols('k:' + str(n-1))  #the spring factor of each joint
        q = dynamicsymbols('q:' + str(n - 1)) #the joint angle
        u = dynamicsymbols('q:' + str(n - 1),1) #the joint angle
        x0 = dynamicsymbols("x0")   #the position of the first link
        y0 = dynamicsymbols("y0")
        theta0 = dynamicsymbols("theta0")
        t = symbols("t")  #time
        dx0 = dynamicsymbols("x0",1)#derivative
        dy0 = dynamicsymbols("y0",1)
        dtheta0 = dynamicsymbols("theta0",1)
        q1_amp = symbols("q1_amp")  #the controlling parameters of the joint
        q1_freq = symbols("q1_freq")
        x_pos = [x0]                #list of x, y, t
        y_pos = [y0]
        theta = [theta0]
        dx_pos = [dx0]              #the derivative of x, y, t
        dy_pos = [dy0]
        dtheta = [dtheta0]
        #constraint the wheel to be moving horizontally only 
        constraints = [dy0*cos(theta0) - dx0*sin(theta0)]  
        #the lagrangian function
        L = 0.5*m[0]*(dx0*dx0+dy0*dy0+dtheta0*dtheta0)
        #loop to get the value of other parameters
        for i in range(n-1):
            #calculate the position
            ttemp = q[i]+theta[-1]
            xtemp = x_pos[-1]+l[i]/2.0*cos(theta[-1])+l[i+1]/2.0*cos(ttemp)
            ytemp = y_pos[-1]+l[i]/2.0*sin(theta[-1])+l[i+1]/2.0*sin(ttemp)

            
            #calculate the derivative
            dttemp = u[i]+dtheta[-1]
            dxtemp = dx_pos[-1] - l[i] / 2.0 * sin(theta[-1]) * dtheta[-1] - l[i+1] / 2.0 * sin(ttemp) * dttemp
            dytemp = dy_pos[-1] + l[i] / 2.0 * cos(theta[-1]) * dtheta[-1] + l[i+1] / 2.0 * cos(ttemp) * dttemp 

            #add the position
            x_pos.append(xtemp)
            y_pos.append(ytemp)
            theta.append(ttemp)

            #add the derivative
            dx_pos.append(dxtemp)
            dy_pos.append(dytemp)
            dtheta.append(dttemp)

            L += 0.5*m[i+1]*(dxtemp*dxtemp+dytemp*dytemp+dttemp*dttemp) - 0.5*k[i]*q[i]*q[i]
            constraints.append(dytemp*cos(ttemp) - dxtemp*sin(ttemp))

        #initialize self objects
        self.l = l
        self.m = m
        self.k = k
        self.q = q
        self.u = u  
        self.x0 = x0            #the position of the first link
        self.y0 = y0
        self.theta0 = theta0
        self.t = t  
        self.dx0 = dx0          #derivative
        self.dy0 = dy0
        self.dtheta0 = dtheta0
        self.x_pos = x_pos      #list of x, y, t
        self.y_pos = y_pos
        self.theta = theta
        self.dx_pos = dx_pos    #the derivative of x, y, t
        self.dy_pos = dy_pos
        self.dtheta = dtheta
        self.n = n              #the number of links
        self.L = L              #the lagrangian function
        self.q1_amp = q1_amp
        self.q1_freq = q1_freq
        constraints.append(self.u[0] - self.q1_amp*self.q1_freq*sin(self.q1_freq*self.t))

        #constraint the wheel to be moving horizontally only 
        self.constraints = constraints  

        if(debug_snake_time):
            start_time = time.time()

        LM = LagrangesMethod(self.L, [self.x0,self.y0,self.theta0]+self.q, nonhol_coneqs = self.constraints)
        LM.form_lagranges_equations()

        #first, process the mass, k_val, and link_length
        if(type(mass)!=list):
            mass = [mass]*self.n
        if(type(link_length)!=list):
            link_length = [link_length]*self.n
        if(type(k_val)!=list):
            k_val = [k_val]*(self.n-1)

        self.mass = mass
        self.link_length = link_length
        self.k_val = k_val
        #set the value for all the variables
        parameters = [self.m[0],self.l[0]]
        self.parameter_vals = [mass[0], link_length[0]]
        #this can potentially be improved, probably through substituting in this function

        #for now, lets just assume that mass, k and link_length are same across different links
        #iterate to set the value of all variables
        for i in range(self.n-1):
            parameters+=[self.m[i+1],self.l[i+1],self.k[i]]
            self.parameter_vals+=[mass[i+1],link_length[i+1],k_val[i]]

        parameters += [self.q1_amp,self.q1_freq]
        #create dummy function
        dynamic = [self.x0,self.y0,self.theta0]+self.q+[self.dx0,self.dy0,self.dtheta0]+self.u
        dummy_symbols = [Dummy() for i in dynamic] 
        dummy_dict = dict(zip(dynamic, dummy_symbols))

        M = LM.mass_matrix_full.simplify().subs(dummy_dict)                                  # Substitute into the mass matrix 
        F = LM.forcing_full.simplify().subs(dummy_dict)                                      # Substitute into the forcing vector
        #print(M)
        # print(F)
        self.M = M
        self.F = F
        self.dummy_symbols = dummy_symbols
        self.parameters = parameters


        # print(LM.mass_matrix_full)
        # self.final_mat = LM.mass_matrix_full.LUsolve(LM.forcing_full)
        self.M_func = lambdify(dummy_symbols + parameters + [self.t], M, "numpy")               # Create a callable function to evaluate the mass matrix 
        self.F_func = lambdify(dummy_symbols + parameters + [self.t], F, "numpy")               # Create a callable function to evaluate the forcing vector 
        self.dynamic = dynamic



    ####################################################################
    ######################  get the actualy motion  ####################
    ####################################################################
    #return the generated y and corresponding t
    def generate_trajectories(self, init, amplitude, frequency, t_val = [0, 50, 1000]):
        '''generate the trajectories of a snake robot's motion given an initial canodition and joints input

        Parameters
        ----------
        a list of initial conditions,
        q1 amplitude,
        q1 frequency,
        time interval

        
        Returns
        ----------
        y, t
        '''
        if(debug_snake_time):
            start_time = time.time()

        self.frequency_val = frequency
        self.amplitude_val = amplitude

        # param_dict = dict(zip(self.parameters, self.parameter_vals + [frequency, amplitude]))

        # #we can just substitute in things here, which would reduce the time a bit
        # M = self.M.subs(param_dict)
        # F = self.F.subs(param_dict)
        # self.M_func = lambdify(self.dummy_symbols + [self.t], M)               # Create a callable function to evaluate the mass matrix 
        # self.F_func = lambdify(self.dummy_symbols + [self.t], F)               # Create a callable function to evaluate the forcing vector 

        x0 = init                                                               # Initial conditions, q and u
        t = linspace(t_val[0], t_val[1], t_val[2])
        if(debug_snake_time):                              # Time vector
            y, info = odeint(self.right_hand_side, x0, t,full_output=True)                                 # Actual integration

        else:
            y = odeint(self.right_hand_side, x0, t) 

        if(debug_snake_time):
            print("odeint %s seconds ---" % (time.time() - start_time))
            print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))
        return y, t
  


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
        if(debug_snake_time):
            start_time = time.time()

        arguments = hstack((x, self.parameter_vals+[self.amplitude_val, self.frequency_val], t))     # States, input, and parameters
        #arguments = hstack((x, t))     # States, input, and parameters


        dx = array(solve(self.M_func(*arguments), # Solving for the derivatives
            self.F_func(*arguments))).T[0]        # The star here is for passing in an array

        if(debug_snake_time):
            print("hstack takes %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

        M = self.M_func(*arguments)

        if(debug_snake_time):
            print("replace M symbols %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

        # TEST_DICT = dict(zip(self.test_input, list(x) + self.parameter_vals+[self.amplitude_val, self.frequency_val] + [t]))
        # print(TEST_DICT)
        # testthings = msubs(self.test_mat, TEST_DICT)
        # print(testthings)

        # if(debug_snake_time):
        #     print("test msub %s seconds ---" % (time.time() - start_time))
        #     start_time = time.time()

        F = self.F_func(*arguments)

        if(debug_snake_time):
            print("replace F symbols %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

        dx = array(solve(M, F)).T[0]        # The star here is for passing in an array
        
        if(debug_snake_time):
            print("solve to get a matrix solution %s seconds ---" % (time.time() - start_time))

        
        return dx[:2*(self.n+2)]

    def plot_image(self, y,t, show_pos = True, show_vel = True):
        '''visualize the trajectory'''
        num = int(y.shape[1] / 2)

        lines = plt.plot(y[:, 0], y[:, 1])
        lab = plt.xlabel('x')
        leg = plt.legend('y')
        plt.show()

        if(show_pos):
            #for x,y,t
            lines = plt.plot(t, y[:, :3])
            lab = plt.xlabel('Time [sec]')
            leg = plt.legend(self.dynamic[:3])
            plt.show()
        
            #for all the q
            lines = plt.plot(t, y[:, 3:self.n+2])
            lab = plt.xlabel('Time [sec]')
            leg = plt.legend(self.dynamic[3:self.n+2])
            plt.show()

        if(show_vel):
            #for x' y' t'
            lines = plt.plot(t, y[:, self.n+2:self.n+5])
            lab = plt.xlabel('Time [sec]')
            leg = plt.legend(self.dynamic[self.n+2:self.n+5])
            plt.show()

            #for all the u
            lines = plt.plot(t, y[:, self.n+5:])
            lab = plt.xlabel('Time [sec]')
            leg = plt.legend(self.dynamic[self.n+5:])
            plt.show()


    def generate_init_value(self, q1_init = 0, q2_init = 0):
        return hstack([0 , 0 , 0, q1_init, q2_init] + [0]*(self.n - 3) +[0, 0, 0, 0, 0] + [0]*(self.n - 3))
    
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
    # #create the snake model
    # snake_test = Snake_Robot(4, mass = 0.01, k_val = 0.01)

    # #generate initial condition
    # init_cond = snake_test.generate_init_value(q1_init = -0.2, q2_init = 0.2)
    
    # #generate trajectory
    # y,t = snake_test.generate_trajectories(init_cond, 
    #     amplitude = 0.2, frequency = 1, t_val = [0, 50, 1000])

    # #plot the trajectory
    # snake_test.plot_image(y, t)

    #create the snake model
    snake_test = Snake_Robot(3, mass = [0.01, 0.01, 0.02], k_val = 0.01)

    #generate initial condition
    init_cond = snake_test.generate_init_value(q1_init = -0.2, q2_init = 0.2)
    
    #generate trajectory
    y,t = snake_test.generate_trajectories(init_cond, 
        amplitude = 0.2, frequency = 1, t_val = [0, 50, 1000])

    #plot the trajectory
    snake_test.plot_image(y, t, show_pos = False, show_vel = False)

    #  #create the snake model
    # snake_test = Snake_Robot(3, mass = [0.01, 0.01, 0.02], k_val = 0.01)

    # #generate initial condition
    # init_cond = snake_test.generate_init_value(q1_init = -0.2, q2_init = 0.0)
    
    # #generate trajectory
    # y,t = snake_test.generate_trajectories(init_cond, 
    #     amplitude = 0.2, frequency = 1, t_val = [0, 50, 1000])

    # #plot the trajectory
    # snake_test.plot_image(y, t)



    # #create the snake model
    # snake_test = Snake_Robot(5, mass = [0.01, 0.01, 0.01, 0.005, 0.005], k_val = 0.01)

    # #generate initial condition
    # init_cond = snake_test.generate_init_value(q1_init = -0.2, q2_init = 0.2)
    
    # #generate trajectory
    # y,t = snake_test.generate_trajectories(init_cond, 
    #     amplitude = 0.2, frequency = 1, t_val = [0, 50, 1000])

    # #plot the trajectory
    # snake_test.plot_image(y, t)




#   some of the other test data I've used
#   init_cond = snake_test.generate_init_value(q1_init = pi/6, q2_init = -pi/6)
#   init_cond = snake_test.generate_init_value(u1_init = 0, q1_init = 0)
#   %time y,t = snake_test.generate_trajectories(init_cond, amplitude = 0.2, frequency = 1, t_val = [0, pi, 1000])
#   init_cond = snake_test.generate_init_value(u1_init = 0, q1_init = 0)
#   %time y,t = snake_test.generate_trajectories(init_cond, amplitude = 0.2, frequency = 1, t_val = [0, pi, 1000])
#   init_cond = snake_test.generate_init_value(u1_init = 0, q1_init = 0.5408495338787851)
#   %time y,t = snake_test.generate_trajectories(init_cond, amplitude = 0.2, frequency = 1, t_val = [0, pi, 1000])
#   init_cond = snake_test.generate_init_value(u1_init = 0, q1_init = 0)
#   %time y,t = snake_test.generate_trajectories(init_cond, amplitude = 0.2, frequency = 1, t_val = [0, pi, 1000])
    
#animate_snake(t, y, 1.0/n, filename="open-loop.html")