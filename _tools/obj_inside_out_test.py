"""Decide which side of a mesh is the room interior, using ray parity.

For each sampled face we step a short distance to both sides of the surface and count
ray/shell intersections in several directions. The side with an odd parity is inside
the room, so a correct surface normal points to that side. A mesh whose normals point
outside on most of its area is inverted.

usage: obj_inside_out_test.py <model dir> [--apply] [--min-area 0.2]
"""
import glob
import math
import os
import sys

import numpy as np

DIRECTIONS = [
    (0.577350, 0.577350, 0.577350),
    (-0.707107, 0.408248, 0.577350),
    (0.267261, -0.801784, 0.534522),
    (0.816497, -0.408248, -0.408248),
    (-0.408248, 0.816497, -0.408248),
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
    return np.array(tris, dtype=np.float64), np.array(owner)


def main():
    root = sys.argv[1]
    apply = '--apply' in sys.argv
    min_area = 0.2
    if '--min-area' in sys.argv:
        min_area = float(sys.argv[sys.argv.index('--min-area') + 1])

    tri, owner = load_all(root)
    v0 = tri[:, 0]
    e1 = tri[:, 1] - v0
    e2 = tri[:, 2] - v0
    geo = np.cross(e1, e2)
    norm = np.linalg.norm(geo, axis=1)
    keep = norm > 1e-12
    v0, e1, e2 = v0[keep], e1[keep], e2[keep]
    geo, norm = geo[keep], norm[keep]
    owner = owner[keep]
    centroid = ((tri[keep][:, 0] + tri[keep][:, 1] + tri[keep][:, 2]) / 3.0)
    unit = geo / norm[:, None]
    area = norm * 0.5
    print('triangles: %d  meshes: %d' % (len(v0), len(set(owner))))

    names = sorted(set(owner))
    rows = []
    for name in names:
        sel = np.nonzero(owner == name)[0]
        total_area = float(area[sel].sum())
        if total_area < min_area:
            continue
        order = sel[np.argsort(-area[sel])]
        sample = order[:6]
        inside_votes = 0
        outside_votes = 0
        for idx in sample:
            c = centroid[idx]
            n = unit[idx]
            for side in (+1.0, -1.0):
                p = c + n * side * (0.01 + 0.02 * 0)
                parities = []
                for d in DIRECTIONS:
                    d = np.array(d, dtype=np.float64)
                    d /= np.linalg.norm(d)
                    pvec = np.cross(d, e2)
                    det = np.einsum('ij,ij->i', e1, pvec)
                    valid = np.abs(det) > 1e-12
                    inv = np.zeros_like(det)
                    inv[valid] = 1.0 / det[valid]
                    tvec = p - v0
                    u = np.einsum('ij,ij->i', tvec, pvec) * inv
                    qvec = np.cross(tvec, e1)
                    v = np.einsum('j,ij->i', d, qvec) * inv
                    t = np.einsum('ij,ij->i', e2, qvec) * inv
                    hit = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-6)
                    parities.append(int(hit.sum()) & 1)
                # odd parity => inside
                if sum(parities) * 2 > len(parities):
                    inside = True
                elif sum(parities) * 2 < len(parities):
                    inside = False
                else:
                    continue
                if side > 0:
                    # the +n side is inside => normal points inward => correct
                    if inside:
                        inside_votes += 1
                    else:
                        outside_votes += 1
                else:
                    # the -n side is inside => normal points outward => inverted
                    if inside:
                        outside_votes += 1
                    else:
                        inside_votes += 1
        total = inside_votes + outside_votes
        if total == 0:
            continue
        rows.append((outside_votes / total, name, total, total_area))

    rows.sort(reverse=True)
    print('\n%-16s %7s %8s %8s' % ('mesh', 'area', 'outside', 'votes'))
    flagged = []
    for frac, name, votes, ar in rows:
        if frac > 0.6:
            flagged.append((name, ar, frac))
        print('%-16s %7.3f %7.1f%% %8d%s'
              % (name, ar, frac * 100.0, votes, '  <-- INVERTED' if frac > 0.6 else ''))

    print('\ninverted candidates: %d  (area sum %.2f)' % (len(flagged), sum(f[1] for f in flagged)))
    if apply and flagged:
        for name, ar, frac in flagged:
            flip_mesh(os.path.join(root, name))
        print('applied %d flips' % len(flagged))


def flip_mesh(path):
    lines = open(path, 'rb').read().split(b'\n')
    out = []
    for raw in lines:
        stripped = raw.strip()
        if stripped[:2] == b'f ':
            toks = [p for p in stripped.split(b' ') if p]
            head, body = toks[0], toks[1:]
            body.reverse()
            out.append(b' '.join([head] + body))
        elif stripped[:2] == b'vn':
            p = stripped.split()
            out.append(b'vn %.9g %.9g %.9g' % (-float(p[1]), -float(p[2]), -float(p[3])))
        else:
            out.append(raw)
    open(path, 'wb').write(b'\n'.join(out))


if __name__ == '__main__':
    main()
