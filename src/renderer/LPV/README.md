# LPV renderer

Cascaded Light Propagation Volumes, after

> A. Kaplanyan, C. Dachsbacher, **"Cascaded Light Propagation Volumes for Real-Time Indirect
> Illumination"**, I3D 2010.

One bounce of diffuse global illumination with fuzzy occlusion from a set of camera centred, nested
3D lattices. Everything runs in compute shaders with inline ray queries (`RayQuery`).

The implementation follows the paper section by section and deliberately contains **only** what the
paper describes: there is no calibration constant, no gather/injection bias, no artistic multiplier
and no per-cascade weight blending. A handful of *implementation* choices remain (ray traced RSM
instead of rasterised, the exact dampening formula, the object size metric); they are listed at the
end.

## Files

| File | Contents |
| --- | --- |
| `LPV.h` / `LPV.cpp` | the renderer: cascade placement, resources, descriptors, passes, UI |
| `LPVShaderio.h` | C++ **and** Slang shared header: constants, `LPVPixelData`, `LPVFrameInfo`, `LPVPushConstant`, binding enums |
| `shaders/LPVCommon.slang` | SH helpers, volume addressing, geometry volume, ray query helpers, light helpers |
| `shaders/LPVPasses.slang` | the five entry points (`lpvGBufferMain`, `lpvRsmInjectMain`, `lpvGvMergeMain`, `lpvPropagateMain`, `lpvGatherMain`) |

## Pipeline

```
lpvGBufferMain    camera G-buffer (ray query): position / normal / albedo / specular / direct light
                  + blocking potential of the camera surfels into the geometry volume   (3.1, 3.2)
lpvRsmInjectMain  virtual reflective shadow map, one ray per RSM texel and light:
                  reflected flux -> SH VPL injection (+ blockers into the GV)            (3.1, 3.2)
lpvGvMergeMain    geometry volume = max(camera GV, RSM GV)                               (3.2)
   copy           accumulated volume = injected volume
lpvPropagateMain  N iterations of the propagation, ping pong, accumulated                (3.3)
lpvGatherMain     trilinear SH lookup, irradiance, dampening, glossy lookup, composite    (3.4, 7)
```

## Math, section by section

**SH basis (2 bands, 4 coefficients)** — `(Y00, Y1-1, Y10, Y11)`:

```
basis(w) = ( 0.282095, 0.488603*w.y, 0.488603*w.z, 0.488603*w.x )
lobe(n)  = ( 0.886227, 1.023327*n.y, 1.023327*n.z, 1.023327*n.x )
```

`lobe(n)` is the SH projection of a clamped cosine **and** the cosine convolved evaluation of an SH
vector at the same time (the coefficients `A0 = pi`, `A1 = 2pi/3` coincide), so it must never be
applied twice.

**Injection** (3.1). A VPL of reflected flux `Phi` and normal `n` adds `(Phi/pi) * lobe(n)` at
`cell = floor((p + 0.5*cellSize*n - origin) / cellSize)` — the paper's fixed half cell offset along
the normal, after which only the orientation of the VPL is used. The `1/pi` is the intensity
normalisation of a Lambertian distribution (`I(w) = (Phi/pi) max(0, n.w)`); the paper writes the
injection as `I_p(w) = Phi_p <n_p,w>_+`, where `Phi_p` is the peak intensity a VPL of total emitted
flux `pi*Phi_p` carries.

**RSM flux** (RSM paper). One texel covers a footprint perpendicular to the light ray, `A_proj`
(`(2*halfExtent/rsm)^2` for the orthographic sun, `d^2 * (2*tan(fov/2)/rsm)^2` for a perspective
light). Undoing that projection gives the true surface area and the irradiance of the patch, and the
observation cosine cancels in the flux:

```
A_s = A_proj / |n.l|          E = E_l * |n.l|          Phi = rho * E * A_s = rho * E_l * A_proj
```

**Geometry volume** (3.2). `B(w) = (A_s / cellSize^2) * max(0, n.w)` accumulated as
`(A_s/cellSize^2) * lobe(n)`. Its cell centres sit on the **corners** of the LPV cells (the GV is
"shifted by half a cell"), so sampling uses `(p - origin)/cellSize` without the `-0.5` of the
intensity volume, and injection rounds to the nearest corner. Nothing else is displaced: the surfel
is *not* pushed along its normal. The camera view and the RSM are injected into separate volumes and
merged with a component wise maximum.

**Propagation** (3.3, Fig. 2 center). One iteration gathers from the 6 neighbours. Every
(source -> destination) transfer is split into the **5 faces of the destination cell** the light can
leave through — the face shared with the source is the entry side and gets no VPL:

```
bin 0     far face  : w_c = d,                    dOmega = 0.4006696846    lobe d
bin 1..4  side faces: w_c = normalize(d + 0.5*u), dOmega = 0.4234313545    lobe u   (u = +-t, +-b)
```

`u` runs over the 4 tangential axes; the `0.5` is exact (a side face's centre sits at `(1, 0.5)` cell
sizes from the source cell's centre). For each bin the flux is `Phi_f = dOmega_f * I(w_c)` of the
source cell and it is re-emitted at the destination's centre as a cosine lobe along the normal of the
face it leaves through, scaled by `Phi_l = Phi_f / pi` — the paper's "new point light source at the
destination cell's center pointing towards the face". The 5 solid angles sum to exactly `4pi/6`, the
solid angle of the shared face, so a uniform intensity field is stationary
(`6 * (4pi/6)/pi * 0.886 * 0.282 = 1`).

Occlusion is sampled once per transfer, at the centre of the face the light propagates through (the
shared face), and evaluated **against** the light travel direction: `B(-w_c)`. `B(w)` is the blocking
potential of surfels whose normal is `w`, i.e. the potential for light travelling along `-w`, so the
query direction is `-w_c` (paper 7 uses the same `B'(-w_c)`). Occlusion is skipped on the very first
iteration, otherwise the emitter's own surfel shadows it.

Each iteration replaces the volume and its result is accumulated into a separate grid ("the sum of
all intermediate results is the final light distribution", 3.3); ping pong rendering as in 6.

**Gather** (3.4). The accumulated volume is sampled trilinearly **at the shading point**, then
`E = dot(c, lobe(-n)) * (2/cellSize)^2` and `L = albedo/pi * E`. The half cell distance (`s/2`) of the
paper is the conversion from intensity to irradiance; the lookup itself is not displaced. The
**minus** matters: the stored coefficients are indexed by the light's travel direction, so the cosine
lobe of a surface with normal `n` is evaluated against `-n` (light arriving at that surface travels
along `-w`).

The accumulated volume holds the *sum* of all iterations, which counts the same transport once per
cell it crosses. That over-count is part of the paper's method — there is no calibration constant.

**Dampening** (3.4, no closed form in the paper). Central difference of the SH vector along the
normal, damped where the derivative is large and points along the intensity:

```
leak = saturate(dot(d, c) / (dot(c, c) + eps))       d = c(p + h*n) - c(p - h*n), h = cellSize/2
mag  = saturate(2 * |d| / (|c| + eps))
damp = saturate(1 - smoothstep(0, 1, leak) * mag)
```

**Cascades** (4.1, 4.2). `cascadeCount` **nested** grids of `N^3` cube cells, cell size
`baseCell * 2^i`, all starting at the camera and displaced along the view direction, snapped to whole
cells. VPLs and geometry are injected into **all** grids in the same pass; the propagation runs
independently per grid. Lighting looks up the **finest grid** that contains the point and, at that
grid's border, interpolates between it and the next coarser grid (one cell wide, smoothstep).

Objects whose bounding box is smaller than a grid's cell size are not injected into that grid
(`LPV_ObjectSize`), which is what prevents aliasing from small bright objects (4.1).

**Glossy** (7). March 4 cells along the reflection direction, average `I(-r)` divided by the squared
distance, multiply by the specular reflectance.

## Differences and limitations

* **RSM without rasterisation.** One ray per virtual RSM texel is traced from the light; the VPL set is
  the same (the surfaces visible from the light). No depth peeling, no second RSM layer (paper 3.2
  offers it as an option).
* **Skylight and emissive area lights do not inject VPLs** (paper 3.1 also injects environment maps and
  area lights as low frequency VPLs; this renderer only handles punctual lights).
* **No multiple bounces** and **no participating media** (paper 7 offers both as extensions).
* **Previous frame surface samples** for static geometry (paper 3.2, optional) are not reused.
* **Dampening formula**: the paper only says "whenever the derivative is large, and c and c's
  derivative are deviating, we dampen c"; the formula above is this renderer's reading of that
  sentence.
* **Object size metric**: the paper says "determined using its bounding box"; here it is the largest
  extent of the world space AABB. The culling skips the VPL injection of that instance in that grid
  (it keeps contributing to the geometry volume).
* The propagation reach is `iterations * cellSize` per cascade. The LPV is a low frequency
  approximation: expect blur, some light bleeding through thin geometry and soft, grid shaped
  occlusion.

## Settings

Only the quantities the paper leaves open (paper 3.3/4: grid resolution, cascade layout, iteration
count) are parameters; everything else is fixed by the paper.

```xml
<renderer type = "LPV">
    <LPV>
        <cascades value = "3" />        <!-- 1..4 nested grids -->
        <resolution value = "32" />     <!-- N, N^3 cells, multiple of 8 -->
        <rsmResolution value = "256" /> <!-- virtual RSM per light, 16..2048, multiple of 16 -->
        <iterations value = "8" />      <!-- propagation iterations, reach ~ iterations cells, 0 = injection only -->
        <cellSize value = "0" />        <!-- finest cell in metres, 0 = scene diameter / N -->
        <useOcclusion value = "true" /> <!-- geometry volume on/off (debug) -->
        <useDampening value = "true" /> <!-- section 3.4 dampening on/off (debug) -->
        <debugView value = "0" />       <!-- 0 off, 1 cascades, 2 direct only, 3 indirect only, 4 reference GI, 5 RSM -->
        <saveFrame value = "-1" />      <!-- write <exe dir>/lpv_snapshot.png at this frame index -->
    </LPV>
</renderer>
```

Debug environment variables: `LPV_DEBUG` (same as `debugView`), `LPV_ITERATIONS`, `LPV_SAVE_FRAME`,
`LPV_NO_SHADOW=1` (disables the direct shadow ray, bit 6 of the push constant flags),
`LPV_NO_DIRECT=1` (same as the *Remove direct light* button).

The *Remove direct light* button in the UI (flag `eLPVFlagNoDirect`, bit 2 of the push constant flags)
drops the direct term from the image, so what remains is the LPV contribution plus the glossy lookup —
handy to judge the indirect term on its own without the *Indirect only* debug view (that one also
disables the glossy term).

`debugView = 5` (the *Show RSM* checkbox) draws the virtual RSM of the first light: one screen pixel
per RSM texel, evaluated through the same `LPV_RsmTexel()` as the injection pass, so it shows exactly
the reflected flux that gets injected. It is a normal gather-pass view, so the tonemapper runs and the
image is never stale; the rest of the frame (propagation, gather) still executes.

## Self checks

The `Reference GI` debug view is an unbiased Monte Carlo single bounce estimator of the same frame and
camera (8 cosine weighted rays per pixel); it is a validation tool, not part of the method. Checks that
run without a reference image:

| Check | Expected |
| --- | --- |
| propagation applied to a uniform intensity field | field unchanged (gain 1.0, per cascade cell size) |
| mirror symmetry of one propagation step | `field(+x lobe) == mirror(field(-x lobe))` |
| wall between two cells | `Indirect only` must drop when `useOcclusion` is on |
| one directional VPL, 3 iterations | the DC centroid moves along the field direction |

## Performance

Measured in a **Debug** build: ~1-2 fps at 1280x720 with 3 cascades of 32^3, 8 iterations and a 256^2
RSM per light. The propagation costs 6 neighbours x 5 exit faces SH evaluations per cell per
iteration, as in the paper (a release build is what the method is designed for).
