import os
import opensim as osim
import casadi as ca
import numpy as np
from utilities import C3D, get_coordinate_indexes, get_ipopt_options

label_map = {
    'pelvis': '/bodyset/pelvis/lower_pelvis',
    'torso': '/bodyset/torso/upper_torso',
    'l_thigh': '/jointset/hip_l/femur_l_offset/l_thigh',
    'l_shank': '/jointset/walker_knee_l/tibia_l_offset/l_shank',
    'l_foot': '/jointset/ankle_l/talus_l_offset/l_foot',
    'l_toes': '/jointset/mtp_l/toes_l_offset/l_toes',
    'r_thigh': '/jointset/hip_r/femur_r_offset/r_thigh',
    'r_shank': '/jointset/walker_knee_r/tibia_r_offset/r_shank',
    'r_foot': '/jointset/ankle_r/talus_r_offset/r_foot',
    'r_toes': '/jointset/mtp_r/toes_r_offset/r_toes',
    'l_uarm': '/jointset/acromial_l/humerus_l_offset/l_uarm',
    'l_larm': '/jointset/elbow_l/ulna_l_offset/l_larm',
    'l_hand': '/jointset/radius_hand_l/hand_l_offset/l_hand',
    'r_uarm': '/jointset/acromial_r/humerus_r_offset/r_uarm',
    'r_larm': '/jointset/elbow_r/ulna_r_offset/r_larm',
    'r_hand': '/jointset/radius_hand_r/hand_r_offset/r_hand',
}

class TrackingCost(ca.Callback):
  def __init__(self, name, model, coordinate_indexes, positions, quaternions,
               time_index, weights, opts={}):
    ca.Callback.__init__(self)
    self.model = model
    self.state = self.model.initSystem()
    self.coordinate_indexes = coordinate_indexes
    self.weights = weights

    # Get the frame paths.
    self.frame_paths = positions.getColumnLabels()

    # Load in positions
    positions_vec3 = positions.getRowAtIndex(time_index)
    self.positions = np.zeros((3, positions_vec3.size()))
    for i in range(positions_vec3.size()):
        self.positions[:, i] = positions_vec3[i].to_numpy()

    # Load in rotations
    self.quaternions = quaternions.getRowAtIndex(time_index)
    self.R_DG = []
    for i in range(self.quaternions.size()):
        rotation = osim.Rotation(self.quaternions.getElt(0, i))
        self.R_DG.append(osim.Rotation(rotation.transpose()))

    self.construct(name, opts)

  # Number of inputs and outputs
  def get_n_in(self): return 1
  def get_n_out(self): return 1

  def get_sparsity_in(self,i):
    return ca.Sparsity.dense(len(self.coordinate_indexes),1)

  def get_sparsity_out(self,i):
    return ca.Sparsity.dense(1,1)

  def calc_errors(self):
    position_errors = []
    rotation_errors = []
    for i, frame_path in enumerate(self.frame_paths):
       frame = osim.Frame.safeDownCast(self.model.getComponent(frame_path))

       # Position error
       position = frame.getPositionInGround(self.state).to_numpy()
       position_errors.append(np.linalg.norm(position - self.positions[:,i]))

       # Rotation error
       R_GF = frame.getRotationInGround(self.state)
       R_DF = self.R_DG[i].multiply(R_GF)
       AA = R_DF.convertRotationToAngleAxis()
       rotation_errors.append(AA.get(0))

    return position_errors, rotation_errors

  def eval(self, arg):
    q = np.zeros(self.state.getNQ())
    q[self.coordinate_indexes] = np.squeeze(arg[0].full())
    self.state.setQ(osim.Vector.createFromMat(q))
    self.model.realizePosition(self.state)
    pos_errors, rot_errors = self.calc_errors()
    return [
            self.weights['position'] * np.sum(np.square(pos_errors)) +
            self.weights['rotation'] * np.sum(np.square(rot_errors))
            ]

# Load model
# ----------
model_fpath = osim.Model(os.path.join('data', 'acl', 'jump_1', 'jump_1_scaled.osim'))
modelProcessor = osim.ModelProcessor(model_fpath)
modelProcessor.append(osim.ModOpRemoveMuscles())
model = modelProcessor.process()
state = model.initSystem()

# For now, disallow models with joints where qdot != u.
assert(state.getNQ() == state.getNU())

# Load tracking data
# ------------------
columns_to_ignore = ['worldbody', 'head', 'pelvis_shifted', 'l_clavicle', 'r_clavicle']
c3d = C3D(os.path.join('data', 'acl', 'jump_1', 'pose_0.c3d'),
          columns_to_ignore=columns_to_ignore, label_map=label_map)
positions = c3d.get_positions_table()
quaternions = c3d.get_quaternions_table()
times = positions.getIndependentColumn()

# Construct problem
# -----------------
# This utility retrieves a mapping between coordinate paths and their indexes in the
# state vector. This only includes independent coordinates, not coupled
# coordinates (e.g., `knee_angle_r_beta`).
coordinates_map = get_coordinate_indexes(model)
coordinate_indexes = list(coordinates_map.values())

# Declare optimization variables.
x = ca.SX.sym('x', len(coordinates_map))

# Define initial guess and bounds.
x0 = []
lbx = []
ubx = []
for coord_path, ix in coordinates_map.items():
    coord = osim.Coordinate.safeDownCast(model.getComponent(coord_path))
    x0.append(coord.getDefaultValue())
    lbx.append(coord.getRangeMin())
    ubx.append(coord.getRangeMax())


N = len(times)
statesTraj = osim.StatesTrajectory()
for itime in range(N):

    # Construct the callback function defining the tracking cost.
    weights = {'position': 10.0, 'rotation': 0.1}
    cost = TrackingCost('tracking_cost', model, coordinate_indexes, positions,
                        quaternions, itime, weights, {'enable_fd': True})
    obj = ca.Function('f', [x], [cost(x)])
    f = obj(x)

    # Form the non-linear program (NLP).
    nlp = {'x':x, 'f':f}

    # Allocate a solver.
    opts = {}
    opts['ipopt'] = get_ipopt_options(1e-2)
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Solve the NLP.
    sol = solver(x0=x0, lbx=lbx, ubx=ubx)

    # Save solution
    state = model.initSystem()
    state.setTime(times[itime])

    q = np.zeros(state.getNQ())
    q[coordinate_indexes] = np.squeeze(sol['x'].full())
    state.setQ(osim.Vector.createFromMat(q))
    statesTraj.append(state)

    x0 = sol['x']

statesTable = statesTraj.exportToTable(model)

osim.STOFileAdapter.write(statesTable, 'ik_solution.sto')
