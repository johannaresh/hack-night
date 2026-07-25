"""Fake CV: privileged-info bounding box, pure geometry.

No rendering, no pixels. Given the drone's pose, the camera uptilt angle,
and the target sphere's position, compute where the target would sit in the
image and how big it would look.

Conventions (shared with physics.py — don't change one without the other):
  World frame:  Z up.
  Body frame:   x forward, y left, z up.
  Quaternion:   [w, x, y, z], rotates body-frame vectors into world frame.
  Camera:       looks along body +x, tilted UP by cam_angle_deg (FPV uptilt),
                square image with field of view fov_deg.
  Image coords: normalized to [-1, 1] on both axes at the FOV edges,
                (0, 0) = image center, +x = right, +y = up.

Output: np.array([bbox_x, bbox_y, bbox_size, visible])
  bbox_x, bbox_y : bbox center in normalized image coords
  bbox_size      : projected diameter of the sphere, same units
  visible        : 1.0 if the target center is in front of the camera and
                   inside the frame, else 0.0 (other fields zeroed)
"""

import numpy as np


def _quat_to_rot(q):
    """Unit quaternion [w, x, y, z] -> 3x3 rotation matrix (body -> world)."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def get_bbox(drone_pos, drone_quat, target_pos, target_radius,
             cam_angle_deg, fov_deg=120.0):
    """Bounding box of the target sphere as seen by the drone's camera.

    Args:
        drone_pos:     (3,) drone position, world frame [m]
        drone_quat:    (4,) drone attitude [w, x, y, z]
        target_pos:    (3,) target sphere center, world frame [m]
        target_radius: sphere radius [m]
        cam_angle_deg: camera uptilt from body x axis [deg]
        fov_deg:       field of view edge-to-edge [deg]

    Returns:
        np.array([bbox_x, bbox_y, bbox_size, visible])
    """
    R_bw = _quat_to_rot(np.asarray(drone_quat, dtype=float))
    rel_world = np.asarray(target_pos, dtype=float) - np.asarray(drone_pos, dtype=float)
    rel_body = R_bw.T @ rel_world

    # Camera basis in body frame: body axes pitched up by the uptilt angle.
    a = np.radians(cam_angle_deg)
    fwd = np.array([np.cos(a), 0.0, np.sin(a)])
    right = np.array([0.0, -1.0, 0.0])
    up = np.array([-np.sin(a), 0.0, np.cos(a)])

    depth = rel_body @ fwd
    if depth <= 1e-6:  # behind the camera
        return np.zeros(4)

    half_tan = np.tan(np.radians(fov_deg) / 2.0)
    bbox_x = (rel_body @ right) / (depth * half_tan)
    bbox_y = (rel_body @ up) / (depth * half_tan)
    bbox_size = 2.0 * target_radius / (depth * half_tan)

    if abs(bbox_x) > 1.0 or abs(bbox_y) > 1.0:  # center out of frame
        return np.zeros(4)

    return np.array([bbox_x, bbox_y, bbox_size, 1.0])


if __name__ == "__main__":
    level = np.array([1.0, 0.0, 0.0, 0.0])  # identity: level, facing world +x

    # Target dead ahead -> centered bbox
    b = get_bbox([0, 0, 0], level, [10, 0, 0], 0.5, cam_angle_deg=0)
    assert np.allclose(b[:2], 0) and b[3] == 1, b
    print("dead ahead:      ", b)

    # Closer target -> bigger bbox
    far = get_bbox([0, 0, 0], level, [20, 0, 0], 0.5, 0)
    assert b[2] > far[2] > 0
    print("2x distance:     ", far)

    # Target above -> bbox in upper half, to the right -> right half
    assert get_bbox([0, 0, 0], level, [10, 0, 3], 0.5, 0)[1] > 0
    assert get_bbox([0, 0, 0], level, [10, -3, 0], 0.5, 0)[0] > 0

    # Camera uptilt pushes a level target DOWN in the frame
    tilted = get_bbox([0, 0, 0], level, [10, 0, 0], 0.5, cam_angle_deg=25)
    assert tilted[1] < 0, tilted
    print("25 deg uptilt:   ", tilted)

    # Target behind -> not visible, all zeros
    assert np.all(get_bbox([0, 0, 0], level, [-10, 0, 0], 0.5, 0) == 0)

    print("all camera sanity checks passed")
