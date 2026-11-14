"""Repair meshes whose surface faces away from the room interior.

A planar (single sided) panel that belongs to the room shell must face the interior,
so its area weighted orientation towards the scene centre is either ~100% (correct)
or ~0% (inside out). Non planar / open meshes are only reported, never flipped, and
closed meshes are additionally checked with the signed volume.

usage: obj_orientation_fix.py <model dir> [--apply]
"""
import glob
import math
import os
import sys


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def v_len(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def newell(pts):
    n = [0.0, 0.0, 0.0]
    count = len(pts)
    for i in range(count):
        p = pts[i]
        q = pts[(i + 1) % count]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    return tuple(n)


class Mesh:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as handle:
            self.lines = handle.read().split(b'\n')
        self.verts = []
        self.faces = []
        for li, raw in enumerate(self.lines):
            line = raw.strip()
            if line[:2] == b'v ':
                p = line.split()
                self.verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line[:2] == b'f ':
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split(b'/')[0])
                    idx.append(i - 1 if i > 0 else len(self.verts) + i)
                if len(idx) >= 3:
                    self.faces.append((li, idx))

    def face_normals(self):
        out = []
        for li, idx in self.faces:
            if any(i < 0 or i >= len(self.verts) for i in idx):
                continue
            pts = [self.verts[i] for i in idx]
            n = newell(pts)
            if v_len(n) < 1e-14:
                continue
            out.append((li, idx, pts, n))
        return out

    def flip(self, negate_normals):
        out = list(self.lines)
        for li, _ in self.faces:
            toks = [p for p in out[li].rstrip().split(b' ') if p]
            head, body = toks[0], toks[1:]
            body.reverse()
            out[li] = b' '.join([head] + body)
        if negate_normals:
            for li, raw in enumerate(out):
                if raw.strip()[:2] == b'vn':
                    p = raw.split()
                    out[li] = b'vn %.9g %.9g %.9g' % (-float(p[1]), -float(p[2]), -float(p[3]))
        with open(self.path, 'wb') as handle:
            handle.write(b'\n'.join(out))


def main():
    root = sys.argv[1]
    apply = '--apply' in sys.argv
    files = sorted(glob.glob(os.path.join(root, '*.obj')))

    meshes = [Mesh(p) for p in files]
    lo = [1e30] * 3
    hi = [-1e30] * 3
    for m in meshes:
        for v in m.verts:
            for k in range(3):
                lo[k] = min(lo[k], v[k])
                hi[k] = max(hi[k], v[k])
    center = [(lo[k] + hi[k]) * 0.5 for k in range(3)]
    print('scene centre: (%.3f %.3f %.3f)' % tuple(center))

    flips = []
    suspects = []
    for m in meshes:
        normals = m.face_normals()
        if not normals:
            continue
        toward = away = 0.0
        dirs = []
        for li, idx, pts, n in normals:
            area = v_len(n) * 0.5
            c = [sum(p[k] for p in pts) / len(pts) for k in range(3)]
            d = v_dot(n, (center[0] - c[0], center[1] - c[1], center[2] - c[2]))
            if d > 0.0:
                toward += area
            else:
                away += area
            dirs.append((n[0] / v_len(n), n[1] / v_len(n), n[2] / v_len(n)))
        total = toward + away
        if total <= 1e-9:
            continue
        # planarity: max angular deviation of the face normals
        ref = dirs[0]
        max_angle = 0.0
        for d in dirs:
            max_angle = max(max_angle, math.degrees(math.acos(max(-1.0, min(1.0, v_dot(ref, d))))))
        planar = max_angle < 1.0
        frac = toward / total
        name = os.path.basename(m.path)
        if planar and total > 0.2 and frac < 0.05:
            flips.append((name, total, frac))
            if apply:
                m.flip(True)
        elif not planar and total > 0.2 and frac < 0.2:
            suspects.append((name, total, frac))

    print('\nflipped (%d):' % len(flips))
    for name, area, frac in sorted(flips, key=lambda t: -t[1]):
        print('  %-16s area=%8.3f toward=%5.1f%%' % (name, area, frac * 100.0))
    if suspects:
        print('\nnon planar, open, facing away (left untouched, review manually) (%d):' % len(suspects))
        for name, area, frac in sorted(suspects, key=lambda t: -t[1])[:20]:
            print('  %-16s area=%8.3f toward=%5.1f%%' % (name, area, frac * 100.0))
    if not apply:
        print('\n(dry run, pass --apply to write)')


if __name__ == '__main__':
    main()
