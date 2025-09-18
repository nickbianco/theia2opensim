import os
import opensim as osim

data_fpath = os.path.join('data', 'acl', 'theia', 'jump_1')
c3d_fpath = os.path.join(data_fpath, 'pose_0.c3d')
unscaled = osim.Model('unscaled_generic.osim')

import pdb; pdb.set_trace()