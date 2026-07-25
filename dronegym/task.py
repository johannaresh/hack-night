"""Task constants shared by env.py and rewards.py.

env.py imports rewards.py, so anything both need lives here instead — that keeps
the dependency a straight line (task <- rewards <- env) with no cycle.
Values are justified in EXECUTION_PLAN.md section 9.
"""

TARGET_RADIUS = 0.5           # sphere radius [m]; matches camera.py's examples
CAPTURE_RADIUS = 1.0          # TARGET_RADIUS + 0.5 m slack
ARENA_RADIUS = 50.0           # [m] from origin; max spawn is 30 m + overshoot room
FOV_DEG = 120.0               # camera field of view, edge to edge

MAX_EPISODE_STEPS = 500       # 10 s at 50 Hz -> truncation
ACTION_REPEAT = 2             # physics steps per policy step: 100 Hz -> 50 Hz policy

LOST_GRACE_STEPS = 25         # consecutive invisible steps tolerated after first sighting
LOST_GRACE_INITIAL = 100      # ...before the target has ever been seen (level 3 search)

OBS_DIM = 12
ACT_DIM = 4
N_STACK = 4                   # VecFrameStack depth; policy sees OBS_DIM * N_STACK
