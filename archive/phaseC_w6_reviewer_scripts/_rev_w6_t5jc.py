import sys
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")
from _rev_w6_t5 import run_joint

run_joint(312, "V2V prior WITH curvature cap (= W6 target), 312 upd",
          v2v=True, cap=True)
