#WESEF2026
import pybullet as p
import math
import pybullet_data
import time
import random
import csv
import numpy as np

BASE_SEED=1337  #Reproducibility seed

class WindEngine:
    """Outputs wind *velocity* (m/s) each step: mean drift + OU turbulence + smooth gusts."""
    def __init__(self, dt,
                 mean_xy=(-3, 3),           # per-trial mean wind range in x,y (m/s)
                 tau=3.0,                    # correlation time (s)
                 sigma=2.0,                  # turbulence strength (m/s)
                 clip=12.0,                  # cap each component (m/s)
                 gust_rate=0.5,              # gusts per second (Poisson)
                 gust_peak=(10, 16),         # gust peak speed (m/s)
                 gust_dur=(0.25, 0.8),       # gust duration (s)
                 allow_vertical=True,       # ALL CAN BE EDITED
                 rng_py=None,
                 rng_np=None):
        self.dt = dt
        self.t = 0.0

        # RNGs (fallbacks if not provided)
        self.rng_py = rng_py if rng_py is not None else random.Random(0)
        self.rng_np = rng_np if rng_np is not None else np.random.default_rng(0)

        # per-trial mean drift
        self.mean = np.array([
            self.rng_py.uniform(*mean_xy), # Random Wind Gusts in X direction
            self.rng_py.uniform(*mean_xy), # Random Wind Gusts in Y direction
            0.0 if not allow_vertical else self.rng_py.uniform(-1, 1) # Random Wind Gusts in Z direction
        ], dtype=float)
        # OU turbulence state
        self.state = np.zeros(3, dtype=float)
        self.theta = 1.0 / max(tau, 1e-6)
        self.sigma = sigma
        self.clip = clip

        self.gust_rate = gust_rate
        self.gust_peak = gust_peak
        self.gust_dur = gust_dur
        self.events = []

    def _maybe_spawn_gust(self):
        if self.rng_py.random() < (1.0 - math.exp(-self.gust_rate * self.dt)):
            ang = self.rng_py.uniform(0, 2*math.pi)
            direction = np.array([math.cos(ang), math.sin(ang), 0.0])
            peak = self.rng_py.uniform(*self.gust_peak)
            T = self.rng_py.uniform(*self.gust_dur)
            self.events.append({"dir": direction, "peak": peak, "t0": self.t, "T": T})

    def _gust_sum(self):
        # Smooth raised-cosine envelope per gust
        g = np.zeros(3)
        keep = []
        for ev in self.events:
            tau = (self.t - ev["t0"]) / ev["T"]
            if 0.0 <= tau <= 1.0:
                env = 0.5 * (1 - math.cos(math.pi * tau))
                g += ev["dir"] * (ev["peak"] * env)
                keep.append(ev)
        self.events = keep
        return g

    def step(self):
        self.t += self.dt
        noise = self.rng_np.normal(0.0, 1.0, size=3)
        self.state += (-self.theta * self.state) * self.dt + self.sigma * math.sqrt(self.dt) * noise

        self._maybe_spawn_gust()
        gust = self._gust_sum()

        wind = self.mean + self.state + gust
        return np.clip(wind, -self.clip, self.clip)

#Features
USE_GUI = True
dt = 1.0/480.0
SIM_DURATION_S = 20.0       # Maximum of 20 seconds
SIMULATION_STEPS = int(SIM_DURATION_S / dt)
SAMPLE_RATE_HZ = 30 
DATA_INTERVAL = int((1.0/dt) / SAMPLE_RATE_HZ) 
NUM_TRIALS = 7        #50 Trials for this simulation
CSV_PATH = "Simulation.csv"     #Edit as needed

#Environmental Factors
Cd = 0.47  # Drag coefficient for sphere
air_density = 1.225  # kg/m^3 (standard at sea level)
radius = 0.5  # sphere2.urdf (meters)
area = math.pi * radius**2  # cross-sectional area (meters^2)

total_data_points = 0   
# ===== Setup CSV =====
with open(CSV_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['trial', 'trial_seed', 'time', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                      'initial_speed', 'launch_angle_deg', 'wind_x', 'wind_y', 'wind_z'])
    # ===== Initialization =====
    p.connect(p.GUI if USE_GUI else p.DIRECT)  #Connect to PyBullet
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)       # Gravity
    p.setRealTimeSimulation(0)
    p.setTimeStep(dt)
    if USE_GUI:
        p.resetDebugVisualizerCamera(cameraDistance=12, cameraYaw=20, cameraPitch=-30, cameraTargetPosition=[15, -5, 10])
    
    for trial in range(1, NUM_TRIALS + 1):
        #Randomized parameters
        trial_seed = BASE_SEED + trial
        rng_py = random.Random(trial_seed)
        rng_np = np.random.default_rng(trial_seed)
        initial_speed = rng_py.uniform(30, 50)
        launch_angle_deg = rng_py.uniform(35, 55)
        launch_angle_rad = math.radians(launch_angle_deg)
        vx = initial_speed * math.cos(launch_angle_rad)
        vz = initial_speed * math.sin(launch_angle_rad)
        vy = rng_py.uniform(-0.5, 0.5)  # Small lateral nudge to rotate direction 

        INITIAL_VELOCITY = [vx, vy, vz]
        PROJECTILE_START_POS = [0,0,0.5] #Start Position (0.5 because of sphere.2urdf radius)

        # Load environment and projectile
        p.resetSimulation() # clear world but keep the same GUI window
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(dt)
        plane_id = p.loadURDF("plane.urdf")
        projectile_id = p.loadURDF("sphere2.urdf", basePosition=PROJECTILE_START_POS, useFixedBase=False)
        p.changeDynamics(projectile_id, -1, restitution=0.5) #Bounce
        p.changeDynamics(plane_id, -1, restitution=0.3, lateralFriction=0.8)
        p.resetBaseVelocity(projectile_id, linearVelocity=INITIAL_VELOCITY)

        # ---- per trial ---- SIMULATE INTENSE CONDITIONS
        wind_engine = WindEngine(
            dt=dt,
            mean_xy=(-10, 10),
            tau=0.5,                 # very choppy
            sigma=6.0,
            clip=16.0,
            gust_rate=1.2,
            gust_peak=(14, 20),
            gust_dur=(0.4, 0.9),
            allow_vertical=False,
            rng_py=rng_py,
            rng_np=rng_np
        )
        
        #Stop detection
        counter = 0       
        STOP_FRAME_COUNT = 30 # Must be grounded for 30 consecutive frames to terminate

        for step in range(SIMULATION_STEPS):
            # 1) advance wind (m/s)
            wind = wind_engine.step()   # world-frame wind velocity

            # 2) get projectile velocity (m/s)
            lin_vel, _ = p.getBaseVelocity(projectile_id)
            v = np.array(lin_vel, dtype=float)

            # 3) relative air velocity
            v_rel = v - wind
            speed_rel = np.linalg.norm(v_rel)

            # 4) aerodynamic force (single, unified drag+wind force)
            if speed_rel > 1e-6:
                F_aero = -0.5 * air_density * Cd * area * speed_rel * v_rel   # N, opposes v_rel
                p.applyExternalForce(projectile_id, -1, F_aero.tolist(), [0,0,0], p.WORLD_FRAME)

            # Step the simulation forward
            p.stepSimulation()
            if USE_GUI:
                time.sleep(dt)

            #DETERMINE STATE
            pos, _ = p.getBasePositionAndOrientation(projectile_id)
            vel, _ = p.getBaseVelocity(projectile_id)

            # Collect data every ~0.033s
            if step % DATA_INTERVAL == 0 and pos[2] > radius + 0.02: #stop logging to csv when the sphere is near the ground
                time_sec = step * dt  # convert step to time in seconds
                writer.writerow([ trial, trial_seed, time_sec,
                                pos[0], pos[1], pos[2],
                                vel[0], vel[1], vel[2],
                                initial_speed, launch_angle_deg,
                                wind[0], wind[1], wind[2]
                                ])
                total_data_points +=1

            has_contact = len(p.getContactPoints(bodyA=projectile_id, bodyB=plane_id)) > 0
            if has_contact:
                counter += 1

            else:
                counter = 0
    
            if counter >= STOP_FRAME_COUNT:
                print(f"Simulation ended early at frame {step} (object stopped).")
                print(f"Total flight time: {step * dt:.2f} seconds")
                print(f"Final position: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
                break
    
    # Terminate
    p.disconnect()

print(f"Data points collected: {total_data_points}")