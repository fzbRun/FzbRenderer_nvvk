"""Find double sided geometry: the same triangle present twice with opposite winding.

Such pairs cancel inside the VGB normal accumulation (sumNormal_G), which makes the
per voxel mean normal collapse to ~0 and the normal partition of that voxel arbitrary.
"""
import glob
import os
import sys


def key_of(pts):
    return tuple(sorted((round(p[0], 6), round(p[1], 6), round(p[2], 6)) for p in pts))


def signed_key(pts):
    return tuple((round(p[0], 6), round(p[1], 6), round(p[2], 6)) for p in pts)


def main():
    root = sys.argv[1]
    files = sorted(glob.glob(os.path.join(root, '*.obj')))
    total_pairs = 0
    rows = []
    for path in files:
        verts = []
        faces = []
        for line in open(path, 'rb'):
            if line[:2] == b'v ':
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line[:2] == b'f ':
                idx = []
                for t in line.split()[1:]:
                    i = int(t.split(b'/')[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                if len(idx) >= 3:
                    faces.append(idx)
        tri = {}
        for idx in faces:
            if any(i < 0 or i >= len(verts) for i in idx):
                continue
            if len(idx) == 3:
                tris = [tuple(idx)]
            else:
                tris = [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
            for t in tris:
                pts = [verts[i] for i in t]
                k = key_of(pts)
                s = signed_key(pts)
                tri.setdefault(k, set()).add(s)
        pairs = 0
        for k, variants in tri.items():
            if len(variants) > 1:
                pairs += 1
        total_pairs += pairs
        if pairs:
            rows.append((pairs, os.path.basename(path), len(faces)))
    rows.sort(reverse=True)
    print('files          : %d' % len(files))
    print('double sided   : %d faces in %d files' % (total_pairs, len(rows)))
    for pairs, name, faces in rows[:25]:
        print('  %-16s dup=%-7d faces=%d' % (name, pairs, faces))


if __name__ == '__main__':
    main()
