"""Decide whether a mesh is inside out by testing visibility from inside the room.

For a face that is visible from an interior viewpoint, the normal must point towards
that viewpoint. A mesh whose visible faces mostly point away is inverted.

usage: obj_visibility_orientation.py <model dir> [mesh names...]
"""
import glob
import os
import sys

import numpy as np

CAMERAS = [
    (-0.079, 1.094, 3.815),   # scene camera
    (-0.802, 3.497, -0.035),  # scene bounding box centre (stairwell)
]


def load_all(root):
    tris = []
    owner = []
    for path in sorted(glob.glob(os.path.join(root, '*.obj'))):
        verts = []
        name = os.path.basename(path)
        for line in open(path, 'rb'):
            if line[:2] == b'v ':
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line[:2] == b'f ':
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split(b'/')[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                if len(idx) < 3:
                    continue
                for k in range(1, len(idx) - 1):
                    tri = (idx[0], idx[k], idx[k + 1])
                    if any(i < 0 or i >= len(verts) for i in tri):
                        continue
                    tris.append((verts[tri[0]], verts[tri[1]], verts[tri[2]]))
                    owner.append(name)
    return np.array(tris, dtype=np.float64), owner


def main():
    root = sys.argv[1]
    only = set(sys.argv[2:])
    tri, owner = load_all(root)
    print('triangles: %d' % len(tri))
    v0 = tri[:, 0]
    e1 = tri[:, 1] - v0
    e2 = tri[:, 2] - v0
    geo = np.cross(e1, e2)
    norm = np.linalg.norm(geo, axis=1)
    ok = norm > 1e-12
    v0, e1, e2, geo, norm = v0[ok], e1[ok], e2[ok], geo[ok], norm[ok]
    owner = [o for o, keep in zip(owner, ok) if keep]
    centroid = (tri[ok][:, 0] + tri[ok][:, 1] + tri[ok][:, 2]) / 3.0
    owner_arr = np.array(owner)
    unit = geo / norm[:, None]
    area = norm * 0.5

    names = sorted(set(owner))
    if only:
        names = [n for n in names if n in only]

    rows = []
    for name in names:
        sel = np.nonzero(owner_arr == name)[0]
        if len(sel) == 0:
            continue
        # sample the largest faces of this mesh
        order = sel[np.argsort(-area[sel])]
        sample = order[:24]
        total_visible = 0
        total_facing = 0
        for idx in sample:
            c = centroid[idx]
            n = unit[idx]
            for cam in CAMERAS:
                cam = np.array(cam, dtype=np.float64)
                d = c - cam
                dist = np.linalg.norm(d)
                if dist < 1e-4:
                    continue
                d /= dist
                # any triangle before the sample face?
                pvec = np.cross(d, e2)
                det = np.einsum('ij,ij->i', e1, pvec)
                valid = np.abs(det) > 1e-12
                inv = np.zeros_like(det)
                inv[valid] = 1.0 / det[valid]
                tvec = cam - v0
                u = np.einsum('ij,ij->i', tvec, pvec) * inv
                qvec = np.cross(tvec, e1)
                v = np.einsum('j,ij->i', d, qvec) * inv
                t = np.einsum('ij,ij->i', e2, qvec) * inv
                hit = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-4) & (t < dist * 0.999)
                hit[idx] = False
                if hit.any():
                    continue
                total_visible += 1
                if np.dot(n, -d) > 0.0:
                    total_facing += 1
                break
        if total_visible < 3:
            continue
        rows.append((total_facing / total_visible, name, total_visible, float(area[sel].sum())))

    rows.sort()
    print('\n%-16s %8s %9s %8s' % ('mesh', 'visible', 'facing', 'area'))
    for frac, name, vis, ar in rows:
        flag = '  <-- INVERTED' if frac < 0.4 else ''
        print('%-16s %8d %8.1f%% %8.3f%s' % (name, vis, frac * 100.0, ar, flag))


if __name__ == '__main__':
    main()
