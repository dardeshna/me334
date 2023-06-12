import os
import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate
from matplotlib import pyplot as plt

from jax import config
config.update("jax_enable_x64", True)

pi = jnp.pi
cos = jnp.cos
sin = jnp.sin

g = 9.81

a = distFromCgToFrontAxle = 1.6  # dist CG to front axle [m]
b = distFromCgToRearAxle = 1.4  # dist CG to rear axle [m]
vehicleCGz = 0.5 # height of CG [m]
trackWidth = 1.6 # track width [m]
rollCenterZ = 0.1 # height of roll center [m]

r_chassis_rc = jnp.array([0,0,rollCenterZ])
r_rc_cg = jnp.array([0,0,vehicleCGz-rollCenterZ])

m_body = 1600 # vehicle unsprung mass [kg]
radiusOfGyration = 1.2 # radius of gyration [m]
Izz = m_body * radiusOfGyration * radiusOfGyration # yaw moment of inertia [kg*m^2]
Ixx = Izz * (0.5)**2 # roll moment of inertia [kg*m^2] 
Iyy = Izz # pitch moment of inertia [kg*m^2]

m_corner = 70 # individual corner sprung mass [kg]

f_ride = 1.3 # ride frequency [Hz]
damping_ratio = 0.3 # damping ratio

k_spring = (2*pi*f_ride)**2 * (m_body/4) # wheel rate [N/m]
b_damping = damping_ratio * 2 * np.sqrt(k_spring * m_body/4) # damping coefficient [N*s/m]

R_wheel = 0.25 # wheel radius [m]

frontCorneringStiffness = 155000 # front cornering stiffness [N/rad]
rearCorneringStiffness = 225000 # rear cornering stiffness [N/rad]

def rot_z(a):
    R = jnp.array([
        [cos(a), -sin(a), 0],
        [sin(a), cos(a), 0],
        [0, 0, 1],
    ])
    return R

def rot_y(a):
    R = jnp.array([
        [cos(a), 0, sin(a)],
        [0,1,0],
        [-sin(a),0, cos(a)],
    ])
    return R

def rot_x(a):
    R = jnp.array([
        [1, 0, 0],
        [0, cos(a), -sin(a)],
        [0, sin(a), cos(a)],
    ])
    return R

def r_world_cg(q):

    x, y, z, psi, theta, phi = q

    return jnp.array([x,y,0]) + rot_z(psi) @ (r_chassis_rc + jnp.array([0,0,z]) + rot_y(theta) @ rot_x(phi) @ r_rc_cg)

def r_world_corner(q):

    x, y, z, psi, theta, phi = q

    cg = r_world_cg(q)

    R_world_body = rot_z(psi) @ rot_y(theta) @ rot_x(phi)

    return jnp.array([
        cg + R_world_body @ jnp.array([a, trackWidth/2, 0]),
        cg + R_world_body @ jnp.array([a, -trackWidth/2, 0]),
        cg + R_world_body @ jnp.array([-b, trackWidth/2, 0]),
        cg + R_world_body @ jnp.array([-b, -trackWidth/2, 0]),
    ])

r_world_corner_static = r_world_corner(jnp.zeros(6))

def w_world_body(q, dq):

    x, y, z, psi, theta, phi = q
    dx, dy, dz, dpsi, dtheta, dphi = dq

    return jnp.array([dphi, 0, 0]) + rot_x(-phi) @ jnp.array([0, dtheta, 0]) + rot_x(-phi) @ rot_y(-theta) @ jnp.array([0, 0, dpsi])

def r_world_tire(q):

    x, y, z, psi, theta, phi = q

    R_world_chassis = rot_z(psi)

    return jnp.array([
        jnp.array([x,y,0]) + R_world_chassis @ jnp.array([a, trackWidth/2, 0]),
        jnp.array([x,y,0]) + R_world_chassis @ jnp.array([a, -trackWidth/2, 0]),
        jnp.array([x,y,0]) + R_world_chassis @ jnp.array([-b, trackWidth/2, 0]),
        jnp.array([x,y,0]) + R_world_chassis @ jnp.array([-b, -trackWidth/2, 0]),
    ])


@jax.jit
def mass_matrix(q):

    Jb_v = jax.jacfwd(r_world_cg)(q)
    Jb_w = jax.jacfwd(w_world_body, 1)(q, jnp.zeros_like(q))

    Jb = jnp.vstack((Jb_v, Jb_w))
    Gb = jnp.diag(jnp.array([m_body, m_body, m_body, Ixx, Iyy, Izz]))
    
    Jc = jax.jacfwd(r_world_tire)(q)
    Gc = jnp.diag(jnp.array([m_corner, m_corner, m_corner]))

    M = Jb.T @ Gb @ Jb + jnp.sum(Jc.transpose((0,2,1)) @ Gc @ Jc, axis=0)

    return M


def christoffel_matrix(q):
    dMdq = jax.jacfwd(mass_matrix)(q)
    Γ = dMdq - 0.5 * dMdq.T
    return Γ

@jax.jit
def coriolis_matrix(q, dq):
    Γ = christoffel_matrix(q)
    return dq.T @ Γ


def potential_energy(q):
    z_corner = r_world_corner(q)[:,2] - r_world_corner_static[:,2]
    return m_body*g*r_world_cg(q)[2] + 1/2 * k_spring * jnp.sum(z_corner**2)

@jax.jit
def gravity_vector(q):
    return jax.jacfwd(potential_energy)(q)


def dissipation(q, dq):
    z_corner = lambda q: r_world_corner(q)[:,2] - r_world_corner_static[:,2]
    return 1/2 * b_damping * jnp.sum((jax.jacfwd(z_corner)(q)@dq)**2)

@jax.jit
def damping_vector(q, dq):
    return jax.jacfwd(dissipation, argnums=1)(q, dq)


def fiala_lateral_tire_model(a, Fz, C, u):
    
    tan_a = jnp.tan(a)
    a_sl = jnp.arctan(3*u*Fz/C)
    
    return jnp.where(
        jnp.abs(a) >= a_sl,
        -u*Fz*jnp.sign(a),
        -C*tan_a + C**2/(3*u*Fz)*jnp.abs(tan_a)*tan_a - C**3/(27*u**2*Fz**2)*tan_a**3
    )

@jax.jit
def genralized_forces(q, dq, rwa, tau):

    x, y, z, psi, theta, phi = q

    R_world_chassis = rot_z(psi)

    Fz = -k_spring * (r_world_corner(q)[:,2] - r_world_corner_static[:,2]) + m_corner*g

    Fx = tau/R_wheel

    u = jnp.sqrt(Fz**2 - Fx**2) / Fz

    dr_dq = jax.jacfwd(r_world_tire)(q)

    v_world_tire_chassis_frame = dr_dq @ dq @ R_world_chassis

    a_Fr = jnp.arctan2(v_world_tire_chassis_frame[0:2,1], v_world_tire_chassis_frame[0:2,0]) - rwa
    a_Re = jnp.arctan2(v_world_tire_chassis_frame[2:4,1], v_world_tire_chassis_frame[2:4,0])
    
    Fy = fiala_lateral_tire_model(
        jnp.concatenate((a_Fr, a_Re)),
        Fz,
        jnp.array([frontCorneringStiffness, frontCorneringStiffness, rearCorneringStiffness, rearCorneringStiffness]),
        u
    )

    F_Fr = R_world_chassis @ rot_z(rwa) @ jnp.array([jnp.array([Fx, Fx]), Fy[0:2], jnp.array([0,0])])
    F_Re = R_world_chassis @ jnp.array([jnp.array([Fx, Fx]), Fy[2:4], jnp.array([0,0])])

    F = jnp.hstack((F_Fr, F_Re)).T

    Q = jnp.sum(F[...,None] * dr_dq, axis=(0,1))

    return Q

def dynamics(x, t, u):

    dq = x[6:12]
    q = x[0:6]

    rwa, tau = u(x, t)
    Q = genralized_forces(q, dq, rwa, tau)

    M = mass_matrix(q)
    C = coriolis_matrix(q, dq)
    g = gravity_vector(q)
    d = damping_vector(q, dq)

    return np.concatenate((dq, np.linalg.solve(M,-C@dq-g-d+Q)))

def get_x0(v_0):

    x0 = np.zeros(12)
    x0[2] = -(m_body/4)*g/k_spring # static deflection [m]
    x0[6] = v_0 # initial speed [m/s]

    return x0

# CONTROLLERS

def gen_sine_steer(T, M):
    
    def u(x, t):

        tau = 0
        if t > 1:
            rwa = M*np.sin(2*pi/T*(t-1))
        else:
            rwa = 0

        return rwa, tau
    
    return u

def gen_step_steer(rwa, tau):

    def u(x, t):

        if t < 1:
            return 0, 0
        elif t < 2:
            return (t-1)*rwa, tau
        else:
            return rwa, tau
    
    return u

def gen_accel_decel(tau_accel, t_accel, tau_decel, t_decel):
    
    def u(x, t):

        rwa = 0
        tau = 0
        if t > 1 + t_accel + t_decel:
            tau = 0
        elif t > 1 + t_accel:
            tau = tau_decel
        elif t > 1:
            tau = tau_accel
        
        return rwa, tau
    
    return u

def plot(ts, res, u, path='', show=False):

    if not os.path.exists(path):
        os.makedirs(path)

    q = res[:,:6]
    dq = res[:,6:]
    ddq = np.zeros_like(q)
    us = np.zeros((len(q), 2))

    for i,t in enumerate(ts):
        ddq[i] = dynamics(res[i], t, u)[6:]
        us[i] = u(res[i], t)

    x, y, z, psi, theta, phi = q.T
    dx, dy, dz, dpsi, dtheta, dphi = dq.T
    ddx, ddy, ddz, ddpsi, ddtheta, ddphi = ddq.T
    rwa, tau = us.T

    # PATH

    plt.figure()
    plt.plot(x, y)
    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig(os.path.join(path, "path.pdf"))

    # HANDLING

    rot_x = lambda a : np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    u, v = np.array([rot_x(-psi[i]) @ np.array([dx[i], dy[i]]) for i in range(len(ts))]).T
    beta = np.arctan2(v, u)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(ts, u)
    plt.ylabel('v [m/s]')
    plt.subplot(3, 1, 2)
    plt.plot(ts, np.rad2deg(beta))
    plt.ylabel('beta [deg]')
    plt.subplot(3, 1, 3)
    plt.plot(ts, np.rad2deg(dpsi))
    plt.ylabel('dpsi [deg/s]')
    plt.xlabel('t [s]')
    plt.savefig(os.path.join(path, "handling.pdf"))

    # GRIP

    ax, ay = np.array([rot_x(-psi[i]) @ np.array([ddx[i], ddy[i]]) for i in range(len(ts))]).T

    plt.figure()
    plt.plot(ax, ay)
    plt.axis('equal')
    th = np.linspace(0,2*np.pi,100)
    plt.plot(g*np.cos(th), g*np.sin(th), '--r')
    plt.xlabel('ax [m]')
    plt.ylabel('ay [m]')
    plt.savefig(os.path.join(path, "grip.pdf"))

    # RIDE

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(ts, np.rad2deg(theta))
    plt.ylabel('theta [deg]')
    plt.subplot(3, 1, 2)
    plt.plot(ts, np.rad2deg(phi))
    plt.ylabel('phi [deg]')
    plt.subplot(3, 1, 3)
    plt.plot(ts, z)
    plt.ylabel('z [m]')
    plt.xlabel('t [s]')
    plt.savefig(os.path.join(path, "ride.pdf"))

    # INPUT

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(ts, np.rad2deg(rwa))
    plt.ylabel('delta [deg]')
    plt.subplot(2, 1, 2)
    plt.plot(ts, tau)
    plt.ylabel('tau [N*m]')
    plt.xlabel('t [s]')
    plt.savefig(os.path.join(path, "inputs.pdf"))

    if show:
        plt.show()


accel_decel = gen_accel_decel(300, 5, -600, 5)
sine_steer_sub_limit = gen_sine_steer(5, np.deg2rad(3))
sine_steer_at_limit = gen_sine_steer(5, np.deg2rad(7))
step_steer = gen_step_steer(np.deg2rad(3), 250)

ts = np.linspace(0, 12, 100)
res = scipy.integrate.odeint(dynamics, get_x0(30), ts, (accel_decel,))
plot(ts, res, accel_decel, path='figures/accel_decel')

ts = np.linspace(0, 16, 100)
res = scipy.integrate.odeint(dynamics, get_x0(20), ts, (sine_steer_sub_limit,))
plot(ts, res, sine_steer_sub_limit, path='figures/sine_steer_sub_limit')

ts = np.linspace(0, 16, 100)
res = scipy.integrate.odeint(dynamics, get_x0(20), ts, (sine_steer_at_limit,))
plot(ts, res, sine_steer_at_limit, path='figures/sine_steer_at_limit')

ts = np.linspace(0, 25, 100)
res = scipy.integrate.odeint(dynamics, get_x0(20), ts, (step_steer,))
plot(ts, res, step_steer, path='figures/step_steer')

