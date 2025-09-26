import os
import opensim as osim
import casadi as ca
import numpy as np
from theia2opensim.utilities import C3D, get_coordinate_indexes, get_ipopt_options

# This custom CasADi callback function computes the tracking cost for a single time
# step. This will allow us to embed the tracking cost into a CasADi expression graph,
# which enables us to construct the NLP below.
class TrackingCost(ca.Callback):
    def __init__(self, name, model, coordinate_indexes, frame_paths, positions,
                 quaternions, weights, opts={}):
        ca.Callback.__init__(self)
        self.model = model
        self.state = self.model.initSystem()
        self.coordinate_indexes = coordinate_indexes
        self.weights = weights
        self.frame_paths = frame_paths

        # Convert the position data to a numpy array.
        self.positions = np.zeros((3, positions.size()))
        for i in range(positions.size()):
            self.positions[:, i] = positions[i].to_numpy()

        # Convert the quaternion data to a list of rotations. We precompute the
        # transpose of the rotations of the data frames, because we need to compute the
        # rotation of the data frame relative to the model frame, R_DF = R_DG * R_GF.
        self.q_GD = quaternions
        self.R_DG = []
        for i in range(self.q_GD.size()):
            rotation = osim.Rotation(self.q_GD.getElt(0, i))
            self.R_DG.append(osim.Rotation(rotation.transpose()))

        # Construct the callback.
        self.construct(name, opts)

    # Number of inputs.
    def get_n_in(self): return 1

    # Number of outputs.
    def get_n_out(self): return 1

    # Input sparsity: [number of coordinates x 1].
    def get_sparsity_in(self,i):
        return ca.Sparsity.dense(len(self.coordinate_indexes),1)

    # Output sparsity: [total error x 1].
    def get_sparsity_out(self,i):
        return ca.Sparsity.dense(1,1)

    def calc_errors(self):
        position_errors = []
        rotation_errors = []
        for i, frame_path in enumerate(self.frame_paths):
            frame = osim.Frame.safeDownCast(self.model.getComponent(frame_path))

            # Compute the position error as the norm of the difference between model and
            # data positions.
            position = frame.getPositionInGround(self.state).to_numpy()
            position_errors.append(np.linalg.norm(position - self.positions[:,i]))

            # Compute the rotation error. To do this, we first compute the rotation
            # between the model frame and the data frame, R_DF = R_DG * R_GF. Then,
            # we convert this rotation to an angle-axis representation and use the angle
            # as the rotation error.
            R_GF = frame.getRotationInGround(self.state)

            # R_DF = self.R_DG[i].multiply(R_GF)
            # # The first element of the returned Vec4 is the angle of rotation.
            # AA = R_DF.convertRotationToAngleAxis()
            # rotation_errors.append(AA.get(0))

            q_GF = R_GF.convertRotationToQuaternion()
            q_GD = self.q_GD.getElt(0, i)

            # Compute the quaternion distance.
            # https://math.stackexchange.com/questions/90081/quaternion-distance
            inner_product = (q_GD.get(0)*q_GF.get(0) + q_GD.get(1)*q_GF.get(1) +
                             q_GD.get(2)*q_GF.get(2) + q_GD.get(3)*q_GF.get(3))
            error = 1 - inner_product*inner_product

            rotation_errors.append(error)

        return position_errors, rotation_errors

    def eval(self, arg):
        # Apply the input coordinates to the model state. We use the coordinate index
        # vector to skip over the state indexes that are not independent coordinates
        # (e.g., `knee_angle_r_beta`).
        q = np.zeros(self.state.getNQ())
        q[self.coordinate_indexes] = np.squeeze(arg[0].full())
        self.state.setQ(osim.Vector.createFromMat(q))

        # Realize the system to the position stage and compute the errors.
        self.model.realizePosition(self.state)

        # Compute the position and rotation errors, and return the weighted sum of
        # squared errors.
        pos_errors, rot_errors = self.calc_errors()
        return [
                self.weights['position']    * np.sum(np.square(pos_errors)) +
                self.weights['orientation'] * np.sum(np.square(rot_errors))
                ]

def run_inverse_kinematics(scaled_model_path, trial_path, c3d_filename,
                           offset_frame_map, weights, convergence_tolerance):

    # Load model
    # ----------
    modelProcessor = osim.ModelProcessor(scaled_model_path)
    modelProcessor.append(osim.ModOpRemoveMuscles())
    model = modelProcessor.process()
    state = model.initSystem()
    # For now, disallow models with joints where qdot != u.
    assert(state.getNQ() == state.getNU())

    # Load tracking data
    # ------------------
    columns_to_ignore = ['worldbody', 'head', 'pelvis_shifted', 'l_clavicle',
                         'r_clavicle']
    label_map = {k: f'{v}/{k}' for k, v in offset_frame_map.items()}
    c3d_filepath = os.path.join(trial_path, c3d_filename)
    c3d = C3D(c3d_filepath, columns_to_ignore=columns_to_ignore, label_map=label_map)
    positions_table = c3d.get_positions_table()
    quaternions_table = c3d.get_quaternions_table()
    frame_paths = positions_table.getColumnLabels()
    times = positions_table.getIndependentColumn()

    # Inverse kinematics
    # ------------------
    # This utility retrieves a mapping between coordinate paths and their indexes in the
    # state vector. This only includes independent coordinates, not coupled coordinates
    # (e.g., `knee_angle_r_beta`).
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

    # Solve position-only optimization to create an inital guess for the full IK
    # problem.
    init_weights = {'position': 10.0,
                    'orientation': 1.0}
    positions = positions_table.getRowAtIndex(0)
    quaternions = quaternions_table.getRowAtIndex(0)
    cost = TrackingCost('tracking_cost', model, coordinate_indexes, frame_paths,
                        positions, quaternions, init_weights, {'enable_fd': True})
    obj = ca.Function('f', [x], [cost(x)])
    f = obj(x)

    # Form the non-linear program (NLP).
    nlp = {'x': x, 'f': f}

    # Allocate a solver.
    opts = {}
    opts['ipopt'] = get_ipopt_options(convergence_tolerance)
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Solve the NLP.
    sol = solver(x0=x0, lbx=lbx, ubx=ubx)
    x0 = sol['x']

    # Iterate over all of the time steps in the tracking data and solve the
    statesTraj = osim.StatesTrajectory()
    for itime, time in enumerate(times):

        # Construct the callback function defining the tracking cost.
        positions = positions_table.getRowAtIndex(itime)
        quaternions = quaternions_table.getRowAtIndex(itime)
        cost = TrackingCost('tracking_cost', model, coordinate_indexes, frame_paths,
                            positions, quaternions, weights, {'enable_fd': True})
        obj = ca.Function('f', [x], [cost(x)])
        f = obj(x)

        # Form the non-linear program (NLP).
        nlp = {'x': x, 'f': f}

        # Allocate a solver.
        opts = {}
        opts['ipopt'] = get_ipopt_options(convergence_tolerance)
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

        # Use the solution for the current time step as the initial guess for the next
        # time step.
        x0 = sol['x']

    # Export the solution to a .sto file.
    statesTable = statesTraj.exportToTable(model)
    trial_name = c3d_filename.replace('.c3d', '')
    solution_path = os.path.join(trial_path, f'{trial_name}_ik_solution.sto')
    osim.STOFileAdapter.write(statesTable, solution_path)
