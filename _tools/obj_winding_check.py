"""Check / repair OBJ face winding against the file's own vertex normals.

Modes:
    report (default)  print statistics only
    --apply           flip the faces that disagree with their vertex normals and
                      repair closed meshes whose signed volume is negative

Rules:
  * a face is flipped when every usable corner normal sits on the opposite side of
    the winding normal (clear inconsistency, no ambiguity)
  * faces with split normals across corners (some corners agree, some not) are left
    alone and reported as "mixed"
  * a closed mesh with negative signed volume is inside out even if winding and
    normals agree with each other, so the whole mesh is flipped and its vn negated
"""
import glob
import os
import sys


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def v_len(a):
    return (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) ** 0.5


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


def face_vidx(tok):
    return int(tok.split(b'/')[0])


def face_vnidx(tok):
    parts = tok.split(b'/')
    if len(parts) >= 3 and parts[2]:
        return int(parts[2])
    return 0


def resolve(index, count):
    return index - 1 if index > 0 else count + index


class ObjFile:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as handle:
            self.raw = handle.read()
        self.lines = self.raw.split(b'\n')
        self.verts = []
        self.norms = []
        self.faces = []
        for li, raw in enumerate(self.lines):
            line = raw.strip()
            if not line:
                continue
            head = line[:2]
            if head == b'v ':
                p = line.split()
                self.verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif head == b'vn':
                p = line.split()
                self.norms.append((float(p[1]), float(p[2]), float(p[3])))
            elif head == b'f ':
                self.faces.append((li, line.split()[1:]))

    def _face_geo(self, toks):
        idx = [resolve(face_vidx(t), len(self.verts)) for t in toks]
        if any(i < 0 or i >= len(self.verts) for i in idx):
            return None, None
        geo = newell([self.verts[i] for i in idx])
        return idx, geo

    def analyse(self):
        res = dict(faces=len(self.faces), tris=0, flip_lines=[], mixed=0,
                   degenerate=0, no_normal=0, zero_normal=0, volume=0.0,
                   closed=False, area_flip=0.0, area_total=0.0)
        edge_use = {}
        for li, toks in self.faces:
            idx, geo = self._face_geo(toks)
            if idx is None:
                continue
            if v_len(geo) < 1e-14:
                res['degenerate'] += 1
                continue
            res['tris'] += 1
            area = v_len(geo) * 0.5
            res['area_total'] += area
            plus = minus = 0
            for t in toks:
                ni = face_vnidx(t)
                if ni == 0:
                    continue
                ni = resolve(ni, len(self.norms))
                if not (0 <= ni < len(self.norms)):
                    continue
                n = self.norms[ni]
                if v_len(n) < 1e-12:
                    res['zero_normal'] += 1
                    continue
                dot = v_dot(geo, n)
                if dot > 0.0:
                    plus += 1
                elif dot < 0.0:
                    minus += 1
            if plus == 0 and minus == 0:
                res['no_normal'] += 1
            elif plus == 0:
                res['flip_lines'].append(li)
                res['area_flip'] += area
            elif minus > 0:
                res['mixed'] += 1
            pts = [self.verts[i] for i in idx]
            if len(pts) == 3:
                res['volume'] += v_dot(pts[0], v_cross(pts[1], pts[2])) / 6.0
            else:
                for k in range(1, len(pts) - 1):
                    res['volume'] += v_dot(pts[0], v_cross(pts[k], pts[k + 1])) / 6.0
            for k in range(len(idx)):
                a = idx[k]
                b = idx[(k + 1) % len(idx)]
                edge_use[(a, b)] = edge_use.get((a, b), 0) + 1
        closed = bool(edge_use)
        for (a, b), count in edge_use.items():
            if count != 1 or edge_use.get((b, a), 0) != 1:
                closed = False
                break
        res['closed'] = closed
        return res

    def rewrite(self, flip_lines, negate_normals):
        out = list(self.lines)
        for li in flip_lines:
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
    if len(sys.argv) < 2:
        print('usage: obj_winding_check.py <obj dir> [--apply]')
        return
    root = sys.argv[1]
    apply = '--apply' in sys.argv
    files = sorted(glob.glob(os.path.join(root, '*.obj')))
    total_faces = total_flips = total_mixed = total_zero = 0
    touched = []
    inside_out = []
    zero_normals = []
    for path in files:
        obj = ObjFile(path)
        st = obj.analyse()
        total_faces += st['faces']
        total_flips += len(st['flip_lines'])
        total_mixed += st['mixed']
        total_zero += st['zero_normal']
        negate = st['closed'] and st['volume'] < 0.0
        if st['flip_lines'] or negate:
            touched.append((os.path.basename(path), st['faces'], len(st['flip_lines']), negate))
            if negate:
                inside_out.append((os.path.basename(path), st['volume'], len(st['flip_lines'])))
            if apply:
                if negate:
                    # fully inside out: reverse every face and flip its normals
                    all_lines = [li for li, _ in obj.faces]
                    obj.rewrite(all_lines, True)
                else:
                    obj.rewrite(st['flip_lines'], False)
        if st['zero_normal']:
            zero_normals.append((os.path.basename(path), st['zero_normal']))

    print('files                : %d' % len(files))
    print('faces                : %d' % total_faces)
    print('faces with flip      : %d' % total_flips)
    print('faces with mixed vn  : %d' % total_mixed)
    print('zero length normals  : %d' % total_zero)
    print('files changed        : %d' % len(touched))
    print('closed inside-out    : %d' % len(inside_out))
    if touched:
        print('\n-- changed files --')
        for name, faces, flips, negate in sorted(touched, key=lambda t: (-t[3], -t[2])):
            print('  %-16s faces=%-7d flip=%-6d%s' % (name, faces, flips, '  NEGATE-ALL' if negate else ''))
    if inside_out:
        print('\n-- closed inside-out detail --')
        for name, vol, flips in inside_out:
            print('  %-16s volume=%.6f local_flips=%d' % (name, vol, flips))
    if zero_normals:
        print('\n-- zero length vn --')
        for name, cnt in zero_normals[:20]:
            print('  %-16s count=%d' % (name, cnt))


if __name__ == '__main__':
    main()
