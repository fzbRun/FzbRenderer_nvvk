"""Area weighted orientation report: how much of each mesh faces the scene centre.

Interior shells (walls, floor, ceiling) should almost entirely face the centre; a
mesh whose area is almost entirely facing away is probably inside-out even though
its winding matches its own vertex normals.
"""
import os
import glob
import sys


def newell(pts):
    n = [0.0, 0.0, 0.0]
    count = len(pts)
    for i in range(count):
        p = pts[i]
        q = pts[(i + 1) % count]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    return n


def load_verts(path):
    verts = []
    faces = []
    for line in open(path, 'rb'):
        if line[:2] == b'v ':
            p = line.split()
            verts.append((float(p[1]), float(p[2]), float(p[3])))
        elif line[:2] == b'f ':
            faces.append([int(t.split(b'/')[0]) for t in line.split()[1:]])
    return verts, faces


def main():
    root = sys.argv[1]
    files = sorted(glob.glob(os.path.join(root, '*.obj')))

    # scene bounds first
    lo = [1e30] * 3
    hi = [-1e30] * 3
    cache = {}
    for path in files:
        verts, faces = load_verts(path)
        cache[path] = (verts, faces)
        for v in verts:
            for k in range(3):
                lo[k] = min(lo[k], v[k])
                hi[k] = max(hi[k], v[k])
    center = [(lo[k] + hi[k]) * 0.5 for k in range(3)]
    print('scene bbox min=(%.3f %.3f %.3f) max=(%.3f %.3f %.3f) center=(%.3f %.3f %.3f)'
          % (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], center[0], center[1], center[2]))

    rows = []
    for path in files:
        verts, faces = cache[path]
        toward = 0.0
        away = 0.0
        for idx in faces:
            if len(idx) < 3:
                continue
            pts = []
            bad = False
            for i in idx:
                j = i - 1 if i > 0 else len(verts) + i
                if j < 0 or j >= len(verts):
                    bad = True
                    break
                pts.append(verts[j])
            if bad or len(pts) < 3:
                continue
            n = newell(pts)
            area = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5 * 0.5
            if area <= 0.0:
                continue
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            cz = sum(p[2] for p in pts) / len(pts)
            d = (center[0] - cx, center[1] - cy, center[2] - cz)
            if n[0] * d[0] + n[1] * d[1] + n[2] * d[2] > 0.0:
                toward += area
            else:
                away += area
        total = toward + away
        if total <= 0.0:
            continue
        rows.append((toward / total, os.path.basename(path), total))

    rows.sort()
    print('\n-- least centre facing meshes (likely inside-out) --')
    for frac, name, area in rows[:30]:
        print('  %-16s toward=%5.1f%%  area=%.4f' % (name, frac * 100.0, area))
    print('\n-- most centre facing meshes --')
    for frac, name, area in rows[-10:]:
        print('  %-16s toward=%5.1f%%' % (name, frac * 100.0))
    low = sum(1 for f, _, _ in rows if f < 0.2)
    mid = sum(1 for f, _, _ in rows if 0.2 <= f <= 0.8)
    high = sum(1 for f, _, _ in rows if f > 0.8)
    print('\nsummary: <20%%=%d  20..80%%=%d  >80%%=%d' % (low, mid, high))


if __name__ == '__main__':
    main()
