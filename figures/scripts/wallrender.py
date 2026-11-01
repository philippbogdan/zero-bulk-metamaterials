"""Trace the stored herringbone voxels to make the wall image and camera orbit.

The input is figures/data/wall_geometry.json. Its solid[i][k] section is
extended along x2, with one unit cube per occupied voxel. Grid traversal
follows Amanatides and Woo; the geometry is neither smoothed nor deformed.

Scene coordinates are X=x1, Y=x2 and Z=x3, with Z vertical. Illumination
combines a soft warm light, a cool sky sampled over a cosine-weighted
hemisphere and one reflected contribution inside the channels. A hidden
plane receives the ground shadow. Outputs and temporary arrays remain
within figures/.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data" / "wall_geometry.json"
RENDERED = HERE
TMP = HERE / "tmp"

BIG = 1.0e30


# Read the binary section and extend it into a voxel volume.


def load_mask():
    record = json.loads(DATA.read_text())
    solid = np.asarray(record["solid"], dtype=np.uint8)
    n, p, p2, p3 = record["N"], record["P"], record["p2"], record["p3"]
    assert solid.shape == (n, n)
    return solid, dict(N=n, P=p, p2=p2, p3=p3, source=record["source"])


def build_grid(solid, meta, depth, crop_i=None, crop_k=None, notch=None):
    """Build occupancy and orientation arrays by extending a section along x2.

    Colour labels are 1 for continuous solid layers, 2 for plates where
    i modulo p3 is below p3/2, and 3 for the opposite plate orientation.
    Crops use half-open (start, stop) ranges on the input i and k axes.
    A notch clears [i0, i1) by [0, y1) by [k0, k1) for the cutaway view.
    """
    p2, p3 = meta["p2"], meta["p3"]
    n = solid.shape[0]
    i = np.arange(n)[:, None]
    k = np.arange(n)[None, :]
    label = np.where(k % p2 < p2 // 2, 1, np.where(i % p3 < p3 // 2, 2, 3)).astype(np.uint8)
    label = label * solid
    i0, i1 = crop_i if crop_i else (0, n)
    k0, k1 = crop_k if crop_k else (0, n)
    sec = solid[i0:i1, k0:k1]
    lab = label[i0:i1, k0:k1]
    occ = np.repeat(sec[:, None, :], depth, axis=1).copy()
    mat = np.repeat(lab[:, None, :], depth, axis=1).copy()
    if notch is not None:
        a0, a1, y1, c0, c1 = notch
        occ[a0:a1, :y1, c0:c1] = 0
        mat[a0:a1, :y1, c0:c1] = 0
    return np.ascontiguousarray(occ), np.ascontiguousarray(mat)


# Hash pixel and sample indices to obtain repeatable random samples.


@njit(inline="always")
def _hash(x):
    x = x & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x


@njit(inline="always")
def _rnd(key, dim):
    return _hash(key ^ ((dim * 0x9E3779B9 + 0x7F4A7C15) & 0xFFFFFFFF)) * (1.0 / 4294967296.0)


# Step through grid cells in ray order.


@njit(inline="always")
def _trace(occ, ox, oy, oz, dx, dy, dz, tlim):
    """Find the nearest occupied cell and return (t, axis, sign, i, j, k).

    A negative t denotes a miss. The entry-face normal has the given
    sign on axis; axis=-1 and t=0 denote an origin within the solid.
    """
    nx = occ.shape[0]
    ny = occ.shape[1]
    nz = occ.shape[2]
    t0 = 0.0
    t1 = tlim
    eax = -1
    if dx > 1e-12 or dx < -1e-12:
        inv = 1.0 / dx
        ta = (0.0 - ox) * inv
        tb = (nx - ox) * inv
        if ta > tb:
            ta, tb = tb, ta
        if ta > t0:
            t0 = ta
            eax = 0
        if tb < t1:
            t1 = tb
    elif ox < 0.0 or ox > nx:
        return -1.0, 0, 0, 0, 0, 0
    if dy > 1e-12 or dy < -1e-12:
        inv = 1.0 / dy
        ta = (0.0 - oy) * inv
        tb = (ny - oy) * inv
        if ta > tb:
            ta, tb = tb, ta
        if ta > t0:
            t0 = ta
            eax = 1
        if tb < t1:
            t1 = tb
    elif oy < 0.0 or oy > ny:
        return -1.0, 0, 0, 0, 0, 0
    if dz > 1e-12 or dz < -1e-12:
        inv = 1.0 / dz
        ta = (0.0 - oz) * inv
        tb = (nz - oz) * inv
        if ta > tb:
            ta, tb = tb, ta
        if ta > t0:
            t0 = ta
            eax = 2
        if tb < t1:
            t1 = tb
    elif oz < 0.0 or oz > nz:
        return -1.0, 0, 0, 0, 0, 0
    if t0 >= t1:
        return -1.0, 0, 0, 0, 0, 0
    px = ox + dx * t0
    py = oy + dy * t0
    pz = oz + dz * t0
    ix = int(math.floor(px))
    iy = int(math.floor(py))
    iz = int(math.floor(pz))
    ix = min(max(ix, 0), nx - 1)
    iy = min(max(iy, 0), ny - 1)
    iz = min(max(iz, 0), nz - 1)
    if dx > 0.0:
        sx = 1
        tdx = 1.0 / dx
        tmx = (ix + 1 - ox) / dx
    elif dx < 0.0:
        sx = -1
        tdx = -1.0 / dx
        tmx = (ix - ox) / dx
    else:
        sx = 0
        tdx = BIG
        tmx = BIG
    if dy > 0.0:
        sy = 1
        tdy = 1.0 / dy
        tmy = (iy + 1 - oy) / dy
    elif dy < 0.0:
        sy = -1
        tdy = -1.0 / dy
        tmy = (iy - oy) / dy
    else:
        sy = 0
        tdy = BIG
        tmy = BIG
    if dz > 0.0:
        sz = 1
        tdz = 1.0 / dz
        tmz = (iz + 1 - oz) / dz
    elif dz < 0.0:
        sz = -1
        tdz = -1.0 / dz
        tmz = (iz - oz) / dz
    else:
        sz = 0
        tdz = BIG
        tmz = BIG
    ax = eax
    sg = 0
    if eax == 0:
        sg = -sx
    elif eax == 1:
        sg = -sy
    elif eax == 2:
        sg = -sz
    t = t0
    while True:
        if occ[ix, iy, iz] != 0:
            return t, ax, sg, ix, iy, iz
        if tmx < tmy:
            if tmx < tmz:
                ix += sx
                t = tmx
                tmx += tdx
                ax = 0
                sg = -sx
                if ix < 0 or ix >= nx:
                    break
            else:
                iz += sz
                t = tmz
                tmz += tdz
                ax = 2
                sg = -sz
                if iz < 0 or iz >= nz:
                    break
        else:
            if tmy < tmz:
                iy += sy
                t = tmy
                tmy += tdy
                ax = 1
                sg = -sy
                if iy < 0 or iy >= ny:
                    break
            else:
                iz += sz
                t = tmz
                tmz += tdz
                ax = 2
                sg = -sz
                if iz < 0 or iz >= nz:
                    break
        if t > t1:
            break
    return -1.0, 0, 0, 0, 0, 0


# Evaluate direct illumination and one reflected contribution.

# Lighting array slices and scalar entries:
# 0:3 is the unit vector pointing towards the main light.
# 3:6 holds its irradiance in linear RGB.
# 6 is the tangent of the light radius angle.
# 7:10 stores sky radiance overhead.
# 10:13 stores sky radiance at the horizon.
# 13:16 supplies radiance for downward directions.
# 16:19 approximates additional illumination at reflected-light intersections.
# 19 multiplies exposure before tone mapping.
# 20:23 gives the linear RGB background.
# 23 controls the ground-shadow contribution.
# 24:27 and 27:30 span the plane perpendicular to the main light.
# 30 sets highlight strength; 31 sets its exponent.


@njit(inline="always")
def _sky(prm, wz):
    if wz >= 0.0:
        a = math.sqrt(wz)
        return (prm[10] + (prm[7] - prm[10]) * a,
                prm[11] + (prm[8] - prm[11]) * a,
                prm[12] + (prm[9] - prm[12]) * a)
    return prm[13], prm[14], prm[15]


@njit(inline="always")
def _key_dir(prm, u1, u2):
    r = prm[6] * math.sqrt(u1)
    ph = 6.283185307179586 * u2
    c = r * math.cos(ph)
    s = r * math.sin(ph)
    lx = prm[0] + c * prm[24] + s * prm[27]
    ly = prm[1] + c * prm[25] + s * prm[28]
    lz = prm[2] + c * prm[26] + s * prm[29]
    inv = 1.0 / math.sqrt(lx * lx + ly * ly + lz * lz)
    return lx * inv, ly * inv, lz * inv


@njit(inline="always")
def _hemi(ax, sg, u1, u2):
    r = math.sqrt(u1)
    ph = 6.283185307179586 * u2
    a = r * math.cos(ph)
    b = r * math.sin(ph)
    c = math.sqrt(max(0.0, 1.0 - u1)) * sg
    if ax == 0:
        return c, a, b
    if ax == 1:
        return a, c, b
    return a, b, c


@njit(inline="always")
def _offset(px, py, pz, ax, sg):
    e = 1e-4 * sg
    if ax == 0:
        return px + e, py, pz
    if ax == 1:
        return px, py + e, pz
    return px, py, pz + e


@njit(inline="always")
def _ndot(ax, sg, x, y, z):
    if ax == 0:
        return sg * x
    if ax == 1:
        return sg * y
    return sg * z


@njit(inline="always")
def _aces(x):
    v = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return min(max(v, 0.0), 1.0)


@njit(parallel=True, cache=True)
def render_rows(occ, mat, alb, cam, prm, W, H, spp, seed, row0, row1, out, cov):
    """Fill a horizontal strip of an orthographic image with linear RGB values.

    The image dimensions are W by H; the strip covers [row0, row1).
    The cov array stores the proportion of samples hitting solid per pixel.
    """
    tx, ty, tz = cam[0], cam[1], cam[2]
    fx, fy, fz = cam[3], cam[4], cam[5]
    rx, ry, rz = cam[6], cam[7], cam[8]
    ux, uy, uz = cam[9], cam[10], cam[11]
    scale = cam[12]
    lx0, ly0, lz0 = prm[0], prm[1], prm[2]
    exposure = prm[19]
    bgr, bgg, bgb = prm[20], prm[21], prm[22]
    gstr = prm[23]
    ks = prm[30]
    kexp = prm[31]
    a1 = 0.7548776662466927
    a2 = 0.5698402909980532
    for py in prange(row0, row1):
        for px in range(W):
            pid = py * 7919 + px
            rot = _hash(pid * 0x27D4EB2D + seed * 0x165667B1)
            ox_r = (rot & 0xFFFF) / 65536.0
            oy_r = (rot >> 16) / 65536.0
            sr = 0.0
            sgn = 0.0
            sbn = 0.0
            nobj = 0
            gnum_r = 0.0
            gnum_g = 0.0
            gnum_b = 0.0
            gden_r = 0.0
            gden_g = 0.0
            gden_b = 0.0
            ngr = 0
            for s in range(spp):
                key = _hash(pid * 0x9E3779B1 + s * 0x85EBCA77 + seed * 0xC2B2AE3D)
                jx = (ox_r + s * a1) % 1.0
                jy = (oy_r + s * a2) % 1.0
                sxp = (px + jx) - 0.5 * W
                syp = (py + jy) - 0.5 * H
                ox = tx + sxp * scale * rx - syp * scale * ux - fx * 2000.0
                oy = ty + sxp * scale * ry - syp * scale * uy - fy * 2000.0
                oz = tz + sxp * scale * rz - syp * scale * uz - fz * 2000.0
                t, ax, sg, ix, iy, iz = _trace(occ, ox, oy, oz, fx, fy, fz, 1.0e9)
                if t > 0.0 and ax >= 0:
                    hx = ox + fx * t
                    hy = oy + fy * t
                    hz = oz + fz * t
                    m = mat[ix, iy, iz]
                    ar, ag, ab = alb[m, 0], alb[m, 1], alb[m, 2]
                    sx0, sy0, sz0 = _offset(hx, hy, hz, ax, sg)
                    # Sample the main light and test visibility for a soft shadow.
                    lx, ly, lz = _key_dir(prm, _rnd(key, 1), _rnd(key, 2))
                    cosl = _ndot(ax, sg, lx, ly, lz)
                    er = 0.0
                    eg = 0.0
                    eb = 0.0
                    vis = 0.0
                    if cosl > 0.0:
                        tt, _a, _s, _i, _j, _k = _trace(occ, sx0, sy0, sz0, lx, ly, lz, 1.0e9)
                        if tt < 0.0:
                            vis = 1.0
                            er += prm[3] * cosl
                            eg += prm[4] * cosl
                            eb += prm[5] * cosl
                    # Sample the sky or, on an intersection, one reflection of the main light.
                    wx, wy, wz = _hemi(ax, sg, _rnd(key, 3), _rnd(key, 4))
                    th, bax, bsg, bi, bj, bk = _trace(occ, sx0, sy0, sz0, wx, wy, wz, 1.0e9)
                    if th < 0.0:
                        lr, lg, lb = _sky(prm, wz)
                        er += 3.141592653589793 * lr
                        eg += 3.141592653589793 * lg
                        eb += 3.141592653589793 * lb
                    elif bax >= 0:
                        qx = sx0 + wx * th
                        qy = sy0 + wy * th
                        qz = sz0 + wz * th
                        qx, qy, qz = _offset(qx, qy, qz, bax, bsg)
                        mb = mat[bi, bj, bk]
                        qr = prm[16]
                        qg = prm[17]
                        qb = prm[18]
                        bl = _ndot(bax, bsg, lx0, ly0, lz0)
                        if bl > 0.0:
                            kx, ky, kz = _key_dir(prm, _rnd(key, 5), _rnd(key, 6))
                            tq, _a, _s, _i, _j, _k = _trace(occ, qx, qy, qz, kx, ky, kz, 1.0e9)
                            if tq < 0.0:
                                qr += prm[3] * bl
                                qg += prm[4] * bl
                                qb += prm[5] * bl
                        er += alb[mb, 0] * qr
                        eg += alb[mb, 1] * qg
                        eb += alb[mb, 2] * qb
                    # Combine diffuse reflection with a weak specular highlight.
                    cr = ar * er / 3.141592653589793
                    cg = ag * eg / 3.141592653589793
                    cb = ab * eb / 3.141592653589793
                    if ks > 0.0 and vis > 0.0:
                        hxv = lx - fx
                        hyv = ly - fy
                        hzv = lz - fz
                        hn = math.sqrt(hxv * hxv + hyv * hyv + hzv * hzv)
                        nh = _ndot(ax, sg, hxv, hyv, hzv) / hn
                        if nh > 0.0:
                            sp = ks * (nh ** kexp) * cosl
                            cr += sp * prm[3]
                            cg += sp * prm[4]
                            cb += sp * prm[5]
                    sr += _aces(cr * exposure)
                    sgn += _aces(cg * exposure)
                    sbn += _aces(cb * exposure)
                    nobj += 1
                elif fz < 0.0:
                    # Use the z=0 plane only to shade the background beneath the solid.
                    tg = -oz / fz
                    gx = ox + fx * tg
                    gy = oy + fy * tg
                    gz = 1e-4
                    lx, ly, lz = _key_dir(prm, _rnd(key, 1), _rnd(key, 2))
                    cosl = lz
                    fr = 0.0
                    fg = 0.0
                    fb = 0.0
                    if cosl > 0.0:
                        fr = prm[3] * cosl
                        fg = prm[4] * cosl
                        fb = prm[5] * cosl
                        tt, _a, _s, _i, _j, _k = _trace(occ, gx, gy, gz, lx, ly, lz, 1.0e9)
                        if tt >= 0.0:
                            fr = 0.0
                            fg = 0.0
                            fb = 0.0
                    wx, wy, wz = _hemi(2, 1, _rnd(key, 3), _rnd(key, 4))
                    lr, lg, lb = _sky(prm, wz)
                    th, _a, _s, _i, _j, _k = _trace(occ, gx, gy, gz, wx, wy, wz, 1.0e9)
                    occf = 1.0 if th < 0.0 else 0.0
                    gnum_r += fr + 3.141592653589793 * lr * occf
                    gnum_g += fg + 3.141592653589793 * lg * occf
                    gnum_b += fb + 3.141592653589793 * lb * occf
                    gden_r += prm[3] * max(cosl, 0.0) + 3.141592653589793 * lr
                    gden_g += prm[4] * max(cosl, 0.0) + 3.141592653589793 * lg
                    gden_b += prm[5] * max(cosl, 0.0) + 3.141592653589793 * lb
                    ngr += 1
            nbg = spp - nobj - ngr
            rr = sr + nbg * bgr
            rg = sgn + nbg * bgg
            rb = sbn + nbg * bgb
            if ngr > 0:
                qr = gnum_r / gden_r
                qg = gnum_g / gden_g
                qb = gnum_b / gden_b
                qr = 1.0 - gstr * (1.0 - qr)
                qg = 1.0 - gstr * (1.0 - qg)
                qb = 1.0 - gstr * (1.0 - qb)
                rr += ngr * bgr * qr
                rg += ngr * bgg * qg
                rb += ngr * bgb * qb
            out[py - row0, px, 0] = rr / spp
            out[py - row0, px, 1] = rg / spp
            out[py - row0, px, 2] = rb / spp
            cov[py - row0, px] = nobj / spp


# Define display conversion, camera placement and illumination.


def srgb_to_lin(c):
    c = np.asarray(c, dtype=float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb8(x):
    x = np.clip(x, 0.0, 1.0)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def camera(target, az_deg, el_deg, scale):
    """Set an orthographic view with azimuth from -Y towards +X and elevation above horizontal."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    to_cam = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
    f = -to_cam
    r = unit(np.cross(f, [0.0, 0.0, 1.0]))
    u = np.cross(r, f)
    return np.concatenate([np.asarray(target, float), f, r, u, [scale]])


def projected_bounds(shape, cam):
    """Project the eight box corners and return their horizontal and vertical limits."""
    nx, ny, nz = shape
    corners = np.array([[x, y, z] for x in (0, nx) for y in (0, ny) for z in (0, nz)], float)
    t = cam[0:3]
    r, u = cam[6:9], cam[9:12]
    sx = (corners - t) @ r
    sy = (corners - t) @ u
    return sx.min(), sx.max(), sy.min(), sy.max()


LOOKS = {
    # Uniform albedo lets illumination alone vary the surface colour.
    "mono": dict(albedo=[(0, 0, 0), (222, 214, 202), (222, 214, 202), (222, 214, 202)]),
    # Keep continuous layers neutral and distinguish the two plate directions.
    "tint": dict(albedo=[(0, 0, 0), (226, 221, 212), (120, 146, 186), (214, 150, 104)]),
    # Use light layers with stronger blue and copper orientation colours.
    "tint2": dict(albedo=[(0, 0, 0), (212, 207, 198), (84, 116, 168), (196, 126, 80)]),
    "tint3": dict(albedo=[(0, 0, 0), (214, 210, 202), (58, 86, 130), (178, 112, 80)]),
    # Apply one blue albedo throughout the solid.
    "blue": dict(albedo=[(0, 0, 0), (98, 122, 160), (98, 122, 160), (98, 122, 160)]),
}


def lighting(look="mono", key_az=-38.0, key_el=48.0, key_soft=0.05, exposure=1.0,
             bg=(255, 255, 255), gstrength=0.55, spec=0.0, spec_exp=40.0):
    kaz, kel = math.radians(key_az), math.radians(key_el)
    L = np.array([math.sin(kaz) * math.cos(kel), -math.cos(kaz) * math.cos(kel), math.sin(kel)])
    t1 = unit(np.cross(L, [0.0, 0.0, 1.0]))
    t2 = np.cross(L, t1)
    prm = np.zeros(32)
    prm[0:3] = L
    prm[3:6] = np.array([1.00, 0.80, 0.62]) * 2.9      # Set the warm light spectrum and intensity.
    prm[6] = key_soft
    prm[7:10] = np.array([0.42, 0.55, 0.80]) * 0.55     # Use blue-biased illumination overhead.
    prm[10:13] = np.array([0.80, 0.84, 0.90]) * 0.55    # Brighten the sky near horizontal directions.
    prm[13:16] = np.array([0.55, 0.52, 0.48]) * 0.40    # Approximate light reflected from the ground.
    prm[16:19] = np.array([0.30, 0.34, 0.42]) * 0.25    # Add a weak ambient contribution at secondary intersections.
    prm[19] = exposure
    prm[20:23] = srgb_to_lin(bg)
    prm[23] = gstrength
    prm[24:27] = t1
    prm[27:30] = t2
    prm[30] = spec
    prm[31] = spec_exp
    alb = srgb_to_lin(np.array(LOOKS[look]["albedo"], float))
    return prm, np.ascontiguousarray(alb)


def render_view(occ, mat, alb, prm, cam, W, H, spp, seed=1, rows_per_call=None):
    out = np.zeros((H, W, 3), np.float32)
    cov = np.zeros((H, W), np.float32)
    render_rows(occ, mat, alb, cam, prm, W, H, spp, seed, 0, H, out, cov)
    return out


def frame_camera(shape, az, el, W, H, fill=0.86, centre=None):
    """Choose a camera scale and centre that enclose the box within the requested image."""
    nx, ny, nz = shape
    cam = camera([nx / 2, ny / 2, nz / 2] if centre is None else centre, az, el, 1.0)
    x0, x1, y0, y1 = projected_bounds(shape, cam)
    scale = max((x1 - x0) / (W * fill), (y1 - y0) / (H * fill))
    cam[12] = scale
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    cam[0:3] = cam[0:3] + cx * cam[6:9] + cy * cam[9:12]
    return cam


def sheet(args):
    """Place three views of the full section and its crops side by side."""
    from PIL import Image
    solid, meta = load_mask()
    full_notch = build_grid(solid, meta, 24, notch=(64, 128, 12, 64, 128))
    full_plain = build_grid(solid, meta, 24)
    crop = build_grid(solid, meta, 24, crop_i=(0, 48), crop_k=(0, 40))
    crop32 = build_grid(solid, meta, 24, crop_i=(0, 48), crop_k=(0, 32))
    variants = [
        ("tint2", full_notch, 32, 24, {}),
        ("tint2", crop, 32, 24, {}),
        ("tint2", crop32, 32, 24, {}),
    ]
    W, H = args.W, args.H
    tiles = []
    t = time.time()
    for look, (occ, mat), az, el, kw in variants:
        prm, alb = lighting(look, **kw)
        cam = frame_camera(occ.shape, az, el, W, H)
        tiles.append(lin_to_srgb8(render_view(occ, mat, alb, prm, cam, W, H, args.spp)))
    print(f"sheet: {len(tiles)} tiles {W}x{H} spp={args.spp}: {time.time() - t:.1f}s")
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * H, cols * W, 3), 255, np.uint8)
    for n, tile in enumerate(tiles):
        r, c = divmod(n, cols)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = tile
    Image.fromarray(canvas).save(args.out)


# Use one geometry and composition for the poster and all video frames.


DESIGN = dict(
    look="tint3",
    depth=24,                          # Number of voxel layers along the invariant x2 direction.
    notch=(64, 128, 12, 64, 128),      # Remove the front upper-right box: i=64:128, x2=0:12, k=64:128.
    crop_i=(0, 48), crop_k=(0, 32),    # Extract the close view directly from these i and k ranges.
    az=32.0, el=24.0,                  # Set the poster view, which shares the opening video angle.
    orbit_az=64.0, orbit_el=10.0,      # Maximum decreases in azimuth and elevation during the camera cycle.
    canvas=(2400, 1350), split=0.575,  # Set poster pixels and the fraction of width allocated to the full section.
    fill=0.88,
    frames=144, fps=24, video_scale=1280 / 2400,
    spp_hero=192, spp_video=56,
    light=dict(),
)


def scenes(design=DESIGN):
    solid, meta = load_mask()
    big = build_grid(solid, meta, design["depth"], notch=design["notch"])
    small = build_grid(solid, meta, design["depth"], crop_i=design["crop_i"], crop_k=design["crop_k"])
    return meta, big, small


def orbit_angles(f, design=DESIGN):
    w = 0.5 * (1 - math.cos(2 * math.pi * f / design["frames"]))
    return design["az"] - design["orbit_az"] * w, design["el"] - design["orbit_el"] * w


def fixed_cameras(shape, W, H, design=DESIGN):
    """Fit the whole orbit at one orthographic scale to keep magnification constant."""
    nx, ny, nz = shape
    centre = [nx / 2, ny / 2, nz / 2]
    need = 0.0
    for f in range(design["frames"]):
        az, el = orbit_angles(f, design)
        cam = camera(centre, az, el, 1.0)
        x0, x1, y0, y1 = projected_bounds(shape, cam)
        # Fit both sides of the projected box about its fixed centre.
        need = max(need, 2 * max(abs(x0), abs(x1)) / (W * design["fill"]),
                   2 * max(abs(y0), abs(y1)) / (H * design["fill"]))
    cams = []
    for f in range(design["frames"]):
        az, el = orbit_angles(f, design)
        cam = camera(centre, az, el, need)
        cams.append(cam)
    return cams


def compose(design, scale_px, f, spp, seed, parts=1, part=0):
    """Calculate linear RGB rows for frame f, scaling the image dimensions by scale_px."""
    CW, CH = (int(round(v * scale_px)) for v in design["canvas"])
    WL = int(round(CW * design["split"]))
    WR = CW - WL
    meta, (occ_b, mat_b), (occ_s, mat_s) = scenes(design)
    prm, alb = lighting(design["look"], **design["light"])
    cam_b = fixed_cameras(occ_b.shape, WL, CH, design)[f]
    cam_s = fixed_cameras(occ_s.shape, WR, CH, design)[f]
    r0 = CH * part // parts
    r1 = CH * (part + 1) // parts
    # Trace each object over the complete image, keeping its centre in its own area.
    # This retains shadows that extend across the boundary between the two views.
    cam_b = cam_b.copy()
    cam_s = cam_s.copy()
    cam_b[0:3] += 0.5 * (CW - WL) * cam_b[12] * cam_b[6:9]
    cam_s[0:3] -= 0.5 * WL * cam_s[12] * cam_s[6:9]
    left = np.zeros((r1 - r0, CW, 3), np.float32)
    right = np.zeros((r1 - r0, CW, 3), np.float32)
    cl = np.zeros((r1 - r0, CW), np.float32)
    cr = np.zeros((r1 - r0, CW), np.float32)
    render_rows(occ_b, mat_b, alb, cam_b, prm, CW, CH, spp, seed, r0, r1, left, cl)
    render_rows(occ_s, mat_s, alb, cam_s, prm, CW, CH, spp, seed + 7, r0, r1, right, cr)
    # Multiply ground-shadow factors without changing the shading of either solid.
    bg = prm[20:23].astype(np.float32)
    out = bg * (left / bg) ** (1.0 - cr[..., None]) * (right / bg) ** (1.0 - cl[..., None])
    assert float((cl * cr).max()) == 0.0, "the two views overlap"
    info = dict(canvas_px=[CW, CH], left_px=[WL, CH], right_px=[WR, CH],
                left_world_units_per_px=float(cam_b[12]), right_world_units_per_px=float(cam_s[12]),
                magnification_right_over_left=float(cam_b[12] / cam_s[12]))
    return out, info


def cmd_hero(args):
    """Save one horizontal strip of the poster to figures/tmp for later assembly."""
    t = time.time()
    rows, info = compose(DESIGN, 1.0, 0, args.spp or DESIGN["spp_hero"], 1, args.parts, args.part)
    TMP.mkdir(exist_ok=True)
    np.save(TMP / f"hero_part_{args.part}.npy", rows.astype(np.float16))
    (TMP / "hero_layout.json").write_text(json.dumps(info))
    print(f"hero part {args.part}/{args.parts}: {rows.shape} in {time.time() - t:.1f}s")


def cmd_assemble(args):
    from PIL import Image
    parts = [np.load(TMP / f"hero_part_{k}.npy").astype(np.float32) for k in range(args.parts)]
    img = lin_to_srgb8(np.concatenate(parts, axis=0))
    RENDERED.mkdir(exist_ok=True)
    Image.fromarray(img).save(RENDERED / "wall.png", optimize=True)
    solid, meta = load_mask()
    layout = json.loads((TMP / "hero_layout.json").read_text())
    info = dict(meta=meta, design={k: v for k, v in DESIGN.items() if k != "light"},
                light=lighting_summary(), layout=layout, size=list(img.shape[:2][::-1]))
    (TMP / "hero_info.json").write_text(json.dumps(info, indent=1))
    for k in range(args.parts):
        (TMP / f"hero_part_{k}.npy").unlink()
    print("wall.png", img.shape)


def cmd_frames(args):
    from PIL import Image
    fdir = TMP / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    for f in range(args.start, min(args.stop, DESIGN["frames"])):
        rows, _ = compose(DESIGN, DESIGN["video_scale"], f, args.spp or DESIGN["spp_video"], 1000 + f)
        Image.fromarray(lin_to_srgb8(rows)).save(fdir / f"f_{f:04d}.png", compress_level=1)
    print(f"frames {args.start}..{args.stop - 1}: {time.time() - t:.1f}s")


def cmd_encode(args):
    import subprocess
    fdir = TMP / "frames"
    n = len(list(fdir.glob("f_*.png")))
    assert n == DESIGN["frames"], f"{n} of {DESIGN['frames']} frames present"
    out = RENDERED / "wall.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(DESIGN["fps"]),
           "-i", str(fdir / "f_%04d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1", "-tune", "film",
           "-movflags", "+faststart", "-an", str(out)]
    subprocess.run(cmd, check=True)
    print("wall.mp4", out.stat().st_size, "bytes")
    if args.clean:
        for p in fdir.glob("f_*.png"):
            p.unlink()
        fdir.rmdir()


def lighting_summary():
    prm, alb = lighting(DESIGN["look"], **DESIGN["light"])
    return dict(key_direction=prm[0:3].round(4).tolist(), key_irradiance=prm[3:6].round(4).tolist(),
                key_softness=float(prm[6]), sky_zenith=prm[7:10].round(4).tolist(),
                sky_horizon=prm[10:13].round(4).tolist(), albedo_srgb=LOOKS[DESIGN["look"]]["albedo"])


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("hero", "assemble", "frames", "encode"):
        ap = argparse.ArgumentParser(description="Create the wall poster, orbit frames or encoded video.")
        ap.add_argument("cmd", help="Operation: hero, assemble, frames or encode.")
        ap.add_argument("--parts", type=int, default=1, help="Number of horizontal poster strips.")
        ap.add_argument("--part", type=int, default=0, help="Poster strip index, starting at zero.")
        ap.add_argument("--spp", type=int, default=0,
                        help="Samples per pixel; zero uses the poster or video setting in DESIGN.")
        ap.add_argument("--start", type=int, default=0, help="Starting orbit frame index, included.")
        ap.add_argument("--stop", type=int, default=0, help="Ending orbit frame index, excluded; use 144 for the full orbit.")
        ap.add_argument("--crf", type=int, default=18, help="H.264 quality setting; lower values make larger files.")
        ap.add_argument("--clean", action="store_true", help="Remove temporary PNG frames after successful encoding.")
        a = ap.parse_args()
        return dict(hero=cmd_hero, assemble=cmd_assemble, frames=cmd_frames, encode=cmd_encode)[a.cmd](a)
    if len(sys.argv) > 1 and sys.argv[1] == "sheet":
        ap = argparse.ArgumentParser(description="Compare the full section and two crops in one image.")
        ap.add_argument("cmd", help="Use sheet to draw the three views.")
        ap.add_argument("--W", type=int, default=600, help="Width of each view in pixels.")
        ap.add_argument("--H", type=int, default=560, help="Height of each view in pixels.")
        ap.add_argument("--spp", type=int, default=16, help="Samples per pixel.")
        ap.add_argument("--out", default=str(TMP / "sheet.png"), help="Destination PNG path.")
        return sheet(ap.parse_args())
    ap = argparse.ArgumentParser(description="Draw a single wall view; see the scripts README for poster and orbit commands.")
    ap.add_argument("--look", default="mono", help="Surface palette: mono, tint, tint2, tint3 or blue.")
    ap.add_argument("--W", type=int, default=800, help="Image width in pixels.")
    ap.add_argument("--H", type=int, default=600, help="Image height in pixels.")
    ap.add_argument("--spp", type=int, default=16, help="Samples per pixel.")
    ap.add_argument("--az", type=float, default=32.0, help="Camera azimuth in degrees from -Y towards +X.")
    ap.add_argument("--el", type=float, default=24.0, help="Camera elevation in degrees above horizontal.")
    ap.add_argument("--out", default=str(TMP / "preview.png"), help="Destination PNG path.")
    a = ap.parse_args()
    from PIL import Image
    solid, meta = load_mask()
    occ, mat = build_grid(solid, meta, 24, notch=(64, 128, 12, 64, 128))
    prm, alb = lighting(a.look)
    cam = camera([64, 12, 64], a.az, a.el, 1.0)
    x0, x1, y0, y1 = projected_bounds(occ.shape, cam)
    scale = max((x1 - x0) / (a.W * 0.86), (y1 - y0) / (a.H * 0.86))
    cam[12] = scale
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    cam[0:3] = cam[0:3] + cx * cam[6:9] + cy * cam[9:12]
    t = time.time()
    img = render_view(occ, mat, alb, prm, cam, a.W, a.H, a.spp)
    print(f"render {a.W}x{a.H} spp={a.spp}: {time.time() - t:.2f}s")
    Path(a.out).parent.mkdir(exist_ok=True)
    Image.fromarray(lin_to_srgb8(img)).save(a.out)


if __name__ == "__main__":
    main()
