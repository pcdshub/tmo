import os, time, socket
from os import path
from typing import Optional, Dict, Any
from pcdsdevices.pim import PPM
from tmo.cam_to_file import h5_img_collect
from tmo.db import camviewer
import pcdsdevices.pim
from time import sleep, strftime
from os import path, mkdir, listdir
import matplotlib.pyplot as plt
import logging
import cv2 as cv
import numpy as np
import h5py

logger = logging.getLogger(__name__)
IMG_PATH = '/cds/home/opr/tmoopr/fiducial_detection'

def _load_best_frame(h5_path: str):
    def _find_dataset(h5: h5py.File):
        # Try common paths first
        for key in ("entry/data/data", "entry_1/data/data", "data"):
            if key in h5:
                return h5[key]
        # Fallback: first 2D/3D integer/unsigned dataset found during DFS
        def _dfs(group):
            for k, v in group.items():
                if isinstance(v, h5py.Dataset):
                    if v.dtype.kind in "ui" and v.ndim in (2, 3):
                        return v
                elif isinstance(v, h5py.Group):
                    hit = _dfs(v)
                    if hit is not None:
                        return hit
            return None
        ds = _dfs(h5)
        if ds is None:
            raise KeyError("No usable 2D/3D integer dataset found in HDF5.")
        return ds

    with h5py.File(h5_path, "r") as f:
        ds = _find_dataset(f)

        if ds.ndim == 2:
            frame = ds[...]
            try:
                logger.info("[_load_best_frame] file=%s shape=%dx%d dtype=%s",
                            h5_path, frame.shape[0], frame.shape[1], frame.dtype)
            except Exception:
                pass
            return frame

        # 3D: choose best frame by robust contrast (p99 - p1)
        best_idx, best_spread = 0, -1.0
        for i in range(ds.shape[0]):
            fr = ds[i, ...]
            # Use percentiles for robustness to outliers
            p1 = float(np.percentile(fr, 1))
            p99 = float(np.percentile(fr, 99))
            spread = p99 - p1
            if spread > best_spread:
                best_spread = spread
                best_idx = i

        frame = ds[best_idx, ...]
        try:
            logger.info("[_load_best_frame] best_frame=%d spread=%.1f", best_idx, best_spread)
            logger.info("[_load_best_frame] file=%s shape=%dx%d dtype=%s",
                        h5_path, frame.shape[0], frame.shape[1], frame.dtype)
        except Exception:
            pass
        return frame

def _match_points(new_pts: np.ndarray, old_pts: np.ndarray, max_px: float) -> np.ndarray:
    """
    Mutual-nearest matching with a distance cap.
    Returns an array of pairs [[i_new, i_old], ...].
    """
    if new_pts.size == 0 or old_pts.size == 0:
        return np.empty((0, 2), dtype=int)

    # squared distance matrix, shape (N_new, N_old)
    d2 = ((new_pts[:, None, :] - old_pts[None, :, :]) ** 2).sum(axis=2)
    n2o = d2.argmin(axis=1)           # for each new, index of nearest old
    o2n = d2.argmin(axis=0)           # for each old, index of nearest new

    pairs = []
    max_d2 = max_px * max_px
    for i_new, j_old in enumerate(n2o):
        if o2n[j_old] == i_new and d2[i_new, j_old] <= max_d2:
            pairs.append((i_new, j_old))
    return np.asarray(pairs, dtype=int)

def _estimate_ecc(old_file: str, new_file: str):
    """
    Global registration fallback using ECC (rotation+translation).
    Returns (angle_deg, tx_px, ty_px, corr) or raises on fatal errors.
    """
    old16 = _load_best_frame(old_file)
    new16 = _load_best_frame(new_file)
    old8, _, _ = _prep_uint8(old16)
    new8, _, _ = _prep_uint8(new16)

    # ECC expects float32, [0..1]
    old32 = old8.astype(np.float32) / 255.0
    new32 = new8.astype(np.float32) / 255.0

    # Try Euclidean (rotation + translation)
    warp_mode = cv.MOTION_EUCLIDEAN
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        corr, warp = cv.findTransformECC(old32, new32, warp, warp_mode, criteria)
        a, b, tx = warp[0]
        c, d, ty = warp[1]
        angle_deg = float(np.degrees(np.arctan2(c, a)))
        return angle_deg, float(tx), float(ty), float(corr)
    except cv.error:
        # Fallback: translation-only via phase correlation
        (dx, dy), resp = cv.phaseCorrelate(old32, new32)
        return 0.0, float(dx), float(dy), float(resp)


def _latest_two_h5(image_dir: str):
    """Return newest two .h5 files, supporting either flat files or per-run subfolders."""
    # 1) If there are ≥2 subfolders, treat each subfolder as a "run"
    subdirs = [d for d in os.listdir(image_dir) if path.isdir(path.join(image_dir, d))]
    if len(subdirs) >= 2:
        subdirs.sort(reverse=True)  # newest first by name
        def newest_h5_in(run_dir):
            fs = [f for f in os.listdir(run_dir) if f.endswith(".h5")]
            if not fs:
                raise RuntimeError(f"No .h5 files in {run_dir}")
            fs.sort()
            return path.join(run_dir, fs[-1])
        newest = newest_h5_in(path.join(image_dir, subdirs[0]))
        prev   = newest_h5_in(path.join(image_dir, subdirs[1]))
        return newest, prev

    # 2) Otherwise, assume a flat directory of .h5 files
    files = [f for f in os.listdir(image_dir) if f.endswith(".h5")]
    if len(files) < 2:
        raise RuntimeError(f"Need ≥2 .h5 files under {image_dir}, found {len(files)}.")
    files.sort()  # your YYYY-MM-DD-HH:MM:SS_###.h5 sorts chronologically
    return path.join(image_dir, files[-1]), path.join(image_dir, files[-2])


def collect_fiducial_image(camera):

    # Look up the camviewer entry for this camera (for PV checks like array rate).
    camviewer_camera = camviewer[camera.name]

    # Quick connectivity test: if acquire_time can't connect, abort early.
    try:
       if not camviewer_camera.cam.acquire_time.connected:
           logger.error('Cannot set camera acquire time!')
           return
    except Exception as e:
       # If camviewer access itself raised, log and exit.
       logger.error(e)
       return

    # Select device-specific illumination/exposure, and the correct target.
    # - PPM uses LED 0–100% and the 'YAG1' target name.
    # - XPIM/IM2K0 use LED power (on/off) and the 'RETICLE' target name.
    target = ''
    if isinstance(camera, PPM):
        camera.led.put(50.0)                 # moderate illumination
        camera.detector.exposure.put(0.0004) # short exposure (s)
        target = 'YAG1'
    elif isinstance(camera, (XPIM, IM2K0)):
        camera.led.power.put(1)              # turn LED on
        camera.detector.exposure.put(0.001)  # short exposure (s)
        target = 'RETICLE'
    else:
        logger.error('Unsupported imager!')
        return

    # Begin acquisition (start streaming).
    camera.detector.acquire.put(1)

    # Ensure the array_rate PV is connected; if not, clean up and exit.
    try:
        if not camviewer_camera.cam.array_rate.connected:
            logger.error('Cannot get camera array rate!')
            # Turn off illumination before returning (device-type specific).
            if isinstance(camera, PPM):
                camera.led.put(0)
            else:
                camera.led.power.put(0)
            camera.detector.acquire.put(0)
            return
    except Exception as e:
        # If reading array_rate raised (e.g., PV missing), clean up and exit.
        logger.error(e)
        if isinstance(camera, PPM):
            camera.led.put(0)
        else:
            camera.led.power.put(0)
        camera.detector.acquire.put(0)
        return

    # Wait until we see at least ~0.5 Hz streaming (max ~10 s, checked every 0.1 s).
    counter = 0
    while camviewer_camera.cam.array_rate.get() < 0.5:
        if counter > 100:
            logger.error('Camera failed to reach 0.5 Hz within 10 seconds')
            # If FPS never comes up, turn off illumination + stop acquire and exit.
            if isinstance(camera, PPM):
                camera.led.put(0)
            else:
                camera.led.power.put(0)
            camera.detector.acquire.put(0)
            return
        sleep(0.1)
        counter += 1

    # Move to the desired target if not already there. Use a blocking move with timeout.
    if camera.target() == target:
        logger.info('Target is already at {}'.format(target))
    else:
        try:
            camera.target.mv(target, wait=True, timeout=20.0)
        except:
            # If the move times out/fails, clean up and exit.
            logger.error('Failed to move to {} within 20 seconds'.format(target))
            if isinstance(camera, PPM):
                camera.led.put(0)
            else:
                camera.led.power.put(0)
            camera.detector.acquire.put(0)
            return

    # Give the system a brief moment to stabilize before capture.
    logger.info('Collecting image')
    sleep(2)

    # Build output path and filename, then collect 2 images to HDF5.
    save_path = path.join(IMG_PATH, camera.name) + '/'
    image_name = strftime('%Y-%m-%d-%H:%M:%S')
    h5_img_collect(camera.name, images=2, path=save_path, file_name=image_name)

    # Post-collection cleanup:
    # - Move the target OUT (note: here a direct call is used, not .mv()).
    # - Turn off illumination.
    # - Stop acquisition.
    camera.target('OUT')
    if isinstance(camera, PPM):
        camera.led.put(0)
    else:
        camera.led.power.put(0)
    camera.detector.acquire.put(0)


def _iter_frames(image_path):
    """Yield (idx, frame16) for each frame in the H5, or (0, 2D) if single frame."""
    with h5py.File(image_path, 'r') as f:
        arr = np.asarray(f['entry/data/data'])
    if arr.ndim == 2:
        yield 0, arr
    else:
        for i in range(arr.shape[0]):
            yield i, arr[i]


def _prep_uint8(img16):
    # Robust per-frame percentile stretch to 8-bit
    p1, p99 = np.percentile(img16, [1, 99])
    if p99 <= p1:
        p1, p99 = float(img16.min()), float(img16.max())
    img_clip = np.clip(img16.astype(np.float32), p1, p99)
    return ((img_clip - p1) * (255.0 / max(p99 - p1, 1.0))).astype(np.uint8), p1, p99


def _detect_centers_from_mask(mask, H, W, minA, maxA, debug_prefix=""):
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    centers, kept = [], 0
    # Sort by area (desc) to log the largest candidates first
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    for c in cnts:
        area = cv.contourArea(c)
        if area < minA or area > maxA:
            logger.debug("%s reject: area=%.0f (bounds %.0f..%.0f)", debug_prefix, area, minA, maxA)
            continue
        (cx, cy), (w, h), _ = cv.minAreaRect(c)
        if w == 0 or h == 0:
            logger.debug("%s reject: degenerate rect", debug_prefix)
            continue
        ar = max(w, h) / max(1e-6, min(w, h))
        if ar > 1.6:  # relax vs 1.35
            logger.debug("%s reject: aspect=%.2f", debug_prefix, ar)
            continue
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.06 * peri, True)  # slightly looser epsilon
        if len(approx) < 4:
            logger.debug("%s reject: sides=%d", debug_prefix, len(approx))
            continue
        centers.append([cx, cy])
        kept += 1
    return np.asarray(centers, dtype=np.float32).reshape(-1, 2), len(cnts)


def analyze_image(image_path, show_image=True):
    """Return (N,2) float32 points of fiducial centers; tries multiple frames/thresholds."""
    best_pts = np.empty((0, 2), np.float32)
    best_info = None
    best_vis = None
    H = W = None

    for idx, frame16 in _iter_frames(image_path):
        H, W = frame16.shape[:2]
        A = float(H * W)
        img8, p1, p99 = _prep_uint8(frame16)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq = clahe.apply(img8)
        blur = cv.GaussianBlur(eq, (5,5), 0)

        k = max(3, int(round(min(H, W) * 0.01)));  k += (1 - k % 2)
        k_close = max(5, int(round(min(H, W) * 0.03)));  k_close += (1 - k_close % 2)

        # Area gates (relaxed a bit)
        minA = 0.0001 * A   # 0.01%
        maxA = 0.10   * A   # 10%

        # Try a few threshold recipes × both polarities
        recipes = [
            ("GAUSS,C=+2", cv.ADAPTIVE_THRESH_GAUSSIAN_C, 61, +2),
            ("MEAN ,C=+2", cv.ADAPTIVE_THRESH_MEAN_C,     61, +2),
            ("OTSU      ", None,                           0,  0),  # global Otsu
        ]
        for name, method, block, C in recipes:
            for pol in (cv.THRESH_BINARY, cv.THRESH_BINARY_INV):
                if method is None:
                    # Otsu global threshold
                    _, t = cv.threshold(blur, 0, 255, pol | cv.THRESH_OTSU)
                else:
                    t = cv.adaptiveThreshold(blur, 255, method, pol, block, C)
                opened = cv.morphologyEx(t, cv.MORPH_OPEN,
                                         cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k)))
                closed = cv.morphologyEx(opened, cv.MORPH_CLOSE,
                                         cv.getStructuringElement(cv.MORPH_RECT, (k_close, k_close)))

                pts, total_cnts = _detect_centers_from_mask(
                    closed, H, W, minA, maxA, debug_prefix=f"[{path.basename(image_path)} f{idx} {name} {'BIN' if pol==cv.THRESH_BINARY else 'BIN_INV'}]"
                )

                logger.info("[analyze_image] file=%s f=%d %s %s: contours=%d kept=%d (A=%d minA=%.0f maxA=%.0f p1=%.1f p99=%.1f)",
                            image_path, idx, name,
                            "BIN" if pol==cv.THRESH_BINARY else "BIN_INV",
                            total_cnts, len(pts), int(A), minA, maxA, p1, p99)

                # Keep the configuration with the most points
                if len(pts) > len(best_pts):
                    best_pts = pts
                    if show_image:
                        vis = cv.cvtColor(eq, cv.COLOR_GRAY2BGR)
                        for (cx, cy) in pts:
                            cv.circle(vis, (int(round(cx)), int(round(cy))), 6, (0,0,255), -1)
                        best_vis = (vis, closed)
                    best_info = (idx, name, "BIN" if pol==cv.THRESH_BINARY else "BIN_INV")

    if best_info is None:
        logger.info("[analyze_image] file=%s: no candidates found", image_path)
        return np.empty((0, 2), np.float32)

    idx, name, pol = best_info
    logger.info("[analyze_image] BEST for %s: frame=%d, recipe=%s, polarity=%s, count=%d",
                image_path, idx, name, pol, len(best_pts))

    if show_image and best_vis is not None:
        vis, mask = best_vis
        f, ax = plt.subplots(1, 2, figsize=(10,4))
        ax[0].imshow(vis);  ax[0].set_title(f"Detected fiducials: {len(best_pts)}")
        ax[1].imshow(mask, cmap='gray'); ax[1].set_title("Chosen mask")
        plt.tight_layout()

    return best_pts.astype(np.float32)



def analyze_image_orb(image_path, show_image=False):
    f = h5py.File(image_path)
    #image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    image = cv.convertScaleAbs(f['entry/data/data'][0], alpha=1/16)
    # eq: equalize image histogram
    eq = cv.equalizeHist(image)
    orb = cv.ORB_create()
    kp, des = orb.detectAndCompute(eq, None)
    kp_filtered = []
    des_filtered = []
    for index, keypoint in enumerate(kp):
        if keypoint.response >= 0.0005:
            kp_filtered.append(keypoint)
            des_filtered.append(des[index])
    if show_image:
        annotated_image = cv.drawKeypoints(eq, kp_filtered, None, (0,0,255), 0)
        plt.imshow(annotated_image)
    return kp_filtered, np.array(des_filtered), eq

def check_fiducial_change_orb(camera_name, show_image=True):
    image_dir = path.join(IMG_PATH, camera_name)
    images = sorted(listdir(path.join(image_dir)), reverse=True)
    if len(images) < 2:
        logger.error('Less than 2 images found in {}'.format(path.join(IMG_PATH, camera_name)))
        return
    kp1, des1, img1 = analyze_image_orb(path.join(image_dir, images[0]))
    kp2, des2, img2 = analyze_image_orb(path.join(image_dir, images[1]))

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key = lambda x:x.distance, reverse=True)
    diffs = []
    good_matches = []
    kp1_arr = cv.KeyPoint_convert(kp1)
    kp2_arr = cv.KeyPoint_convert(kp2)
    for match in matches:
        pt1 = kp1_arr[match.queryIdx]
        pt2 = kp2_arr[match.trainIdx]
        if match.distance <= 35:
            good_matches.append(match)
            diffs.append(pt1 - pt2)
    diffs_norm = np.linalg.norm(diffs, axis=1)
    if show_image:
        f, axs = plt.subplots(2,1)
        img3 = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        axs[0].imshow(img3)
        axs[1].hist(diffs_norm, bins=[0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000])
        axs[1].set_xscale('log')
    return np.sort(diffs_norm)

def microns_per_pixel(hull):
    sides = hull - np.roll(hull, 1, axis=0)
    average_length = np.average(np.linalg.norm(sides, axis=1))
    return 8500 / average_length

def check_fiducial_change(camera_name):
    image_dir = path.join(IMG_PATH, camera_name)
    new_file, old_file = _latest_two_h5(image_dir)
    logger.info(f"Comparing new={new_file}  to  old={old_file}")

    new_pts = analyze_image(new_file, show_image=False).astype(np.float32)
    old_pts = analyze_image(old_file, show_image=False).astype(np.float32)

    # Estimate mpp if we can (from NEW points); if not, we'll report px
    mpp = None
    if new_pts.shape[0] >= 3:
        try:
            new_hull_for_scale = np.squeeze(cv.convexHull(new_pts))
            mpp = microns_per_pixel(new_hull_for_scale)
        except Exception:
            mpp = None

    # Matching
    if new_pts.size and old_pts.size:
        span_new = float(np.ptp(new_pts, axis=0).max())
        span_old = float(np.ptp(old_pts, axis=0).max())
        min_span = max(1.0, min(span_new, span_old))
    else:
        min_span = 2048.0
    max_px = max(30.0, 0.03 * min_span)

    pairs = _match_points(new_pts, old_pts, max_px=max_px)
    logger.info("Matched %d fiducials (new=%d, old=%d, max_px=%.1f)",
                len(pairs), len(new_pts), len(old_pts), max_px)

    # === Case A: full model when we have >=3 matched pairs ===
    if len(pairs) >= 3:
        new_m = new_pts[pairs[:, 0]]
        old_m = old_pts[pairs[:, 1]]

        # (re)compute mpp from matched NEW hull (more robust than all-new_pts)
        try:
            mpp = microns_per_pixel(np.squeeze(cv.convexHull(new_m.astype(np.float32))))
        except Exception:
            pass

        # Pairwise diffs (px -> µm if we can)
        diffs_px = new_m - old_m
        diffs_um = diffs_px * (mpp if mpp is not None else 1.0)
        norm_average = float(np.linalg.norm(diffs_um.mean(axis=0)))
        average_norm = float(np.mean(np.linalg.norm(diffs_um, axis=1)))

        logger.info('Calculated microns per pixel: %s', f"{mpp:.6f}" if mpp is not None else "N/A (px units)")
        logger.info('Average movement vector magnitude: %.2f %s',
                    norm_average, "µm" if mpp is not None else "px")
        logger.info('Per-fiducial average movement: %.2f %s',
                    average_norm, "µm" if mpp is not None else "px")
        return norm_average, average_norm, diffs_um

    # === Case B: 1–2 matched pairs → translation-only from those pairs (fast) ===
    if 1 <= len(pairs) <= 2:
        new_m = new_pts[pairs[:, 0]]
        old_m = old_pts[pairs[:, 1]]
        trans_px = (new_m - old_m).mean(axis=0)     # shape (2,)
        diffs_px = (new_m - old_m)                  # per-pair vectors

        if mpp is not None:
            diffs_um = diffs_px * mpp
            shift_mag = float(np.linalg.norm(trans_px * mpp))
            per_avg = float(np.mean(np.linalg.norm(diffs_um, axis=1)))
            logger.warning("Only %d matched fiducials; using translation-only estimate.", len(pairs))
            logger.info("Translation-only: shift ≈ (%.2f, %.2f) px  (≈ (%.2f, %.2f) µm)",
                        trans_px[0], trans_px[1], trans_px[0]*mpp, trans_px[1]*mpp)
            return shift_mag, per_avg, diffs_um
        else:
            # No scale → report in pixels
            diffs_um = diffs_px  # (units: px)
            shift_mag = float(np.linalg.norm(trans_px))
            per_avg = float(np.mean(np.linalg.norm(diffs_px, axis=1)))
            logger.warning("Only %d matches and no mpp; reporting in pixels.", len(pairs))
            return shift_mag, per_avg, diffs_um

    # === Case C: 0 matches → global image registration fallback (ECC / phase) ===
    logger.warning("No matched fiducials; falling back to global registration.")
    angle_deg, tx_px, ty_px, corr = _estimate_ecc(old_file, new_file)

    if mpp is not None:
        dx_um, dy_um = tx_px * mpp, ty_px * mpp
        mag_um = float(np.hypot(dx_um, dy_um))
        diffs_um = np.array([[dx_um, dy_um]], dtype=np.float32)
        logger.info("ECC fallback: rotation=%.3f deg, translation=(%.2f, %.2f) px "
                    "(≈ (%.2f, %.2f) µm), corr=%.4f",
                    angle_deg, tx_px, ty_px, dx_um, dy_um, corr)
        return mag_um, mag_um, diffs_um
    else:
        mag_px = float(np.hypot(tx_px, ty_px))
        diffs_px = np.array([[tx_px, ty_px]], dtype=np.float32)
        logger.info("ECC fallback (no mpp): rotation=%.3f deg, translation=(%.2f, %.2f) px, corr=%.4f",
                    angle_deg, tx_px, ty_px, corr)
        return mag_px, mag_px, diffs_px




def debug_images_dir(camera_name):
    image_dir = path.join(IMG_PATH, str(camera_name))
    logger.info("[debug] host=%s", socket.gethostname())
    logger.info("[debug] cwd=%s", os.getcwd())
    logger.info("[debug] IMG_PATH=%r", IMG_PATH)
    logger.info("[debug] camera_name=%r", camera_name)
    logger.info("[debug] resolved image_dir=%r", image_dir)
    logger.info("[debug] exists=%s isdir=%s", path.exists(image_dir), path.isdir(image_dir))
    if not path.isdir(image_dir):
        return

    with os.scandir(image_dir) as it:
        entries = list(it)

    files   = [e.name for e in entries if e.is_file()]
    subdirs = [e.name for e in entries if e.is_dir()]
    logger.info("[debug] total entries=%d  files=%d  subdirs=%d",
                len(entries), len(files), len(subdirs))
    if subdirs:
        logger.info("[debug] first 10 subdirs: %s", subdirs[:10])

    # Case-insensitive .h5 filter
    h5s = [f for f in files if f.lower().endswith(".h5")]
    logger.info("[debug] .h5 files count=%d", len(h5s))
    if h5s:
        h5s_lex = sorted(h5s)
        logger.info("[debug] last 5 by lexicographic: %s", h5s_lex[-5:])

        # Sort by mtime as well (helps if timestamps in names are inconsistent)
        h5s_by_mtime = sorted(h5s, key=lambda fn: path.getmtime(path.join(image_dir, fn)))
        def fmt(fn):
            p = path.join(image_dir, fn)
            return f"{fn}  size={path.getsize(p)}B  mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path.getmtime(p)))}"
        logger.info("[debug] last 5 by mtime: %s",
                    [fmt(fn) for fn in h5s_by_mtime[-5:]])

# --- Decision thresholds (tune to your actual desired tolerances!!) ---
DEFAULT_THRESHOLDS = dict(
    min_pairs=6,            # need at least this many matched fiducials
    inlier_frac_fail=0.60,  # RANSAC inlier fraction below this => FAIL
    rot_deg_warn=0.05,      # rotation > 0.05° => WARN
    rot_deg_fail=0.15,      # rotation > 0.15° => FAIL
    shift_um_warn=15.0,     # mean translation magnitude > 15 µm => WARN
    shift_um_fail=40.0,     # > 40 µm => FAIL
    scale_pct_warn=0.20,    # |scale-1|*100 > 0.20% => WARN
    scale_pct_fail=0.50,    # > 0.50% => FAIL
    rms_um_warn=8.0,        # RMS residual (inliers) > 8 µm => WARN
    rms_um_fail=20.0,       # > 20 µm => FAIL
)

def assess_alignment(camera_name: str, thresholds: Optional[Dict[str, float]] = None):
    """
    Modes:
      - 'full' (>=3 matches): RANSAC similarity fit (rot, scale, shift, RMS).
      - 'translation-only' (1–2 matches): average translation from pairs.
      - 'ecc-fallback' (0 matches): ECC (or phase correlation) global registration.
    """
    th = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        th.update(thresholds)

    def bump(status: str, level: str) -> str:
        order = {"OK": 0, "WARN": 1, "FAIL": 2}
        return max(status, level, key=lambda s: order[s])

    image_dir = path.join(IMG_PATH, camera_name)
    new_file, old_file = _latest_two_h5(image_dir)
    logger.info("Assessing %s: new=%s  old=%s", camera_name, new_file, old_file)

    new_pts = analyze_image(new_file, show_image=False).astype(np.float32)
    old_pts = analyze_image(old_file, show_image=False).astype(np.float32)

    # Try to get mpp from the NEW geometry; may be None if we don't have >=3 points
    mpp = None
    if new_pts.shape[0] >= 3:
        try:
            new_hull_all = np.squeeze(cv.convexHull(new_pts))
            mpp = microns_per_pixel(new_hull_all)
        except Exception:
            mpp = None

    # Match points (span-relative cap)
    if new_pts.size and old_pts.size:
        span_new = float(np.ptp(new_pts, axis=0).max())
        span_old = float(np.ptp(old_pts, axis=0).max())
        min_span = max(1.0, min(span_new, span_old))
    else:
        min_span = 2048.0
    max_px = max(30.0, 0.03 * min_span)

    pairs = _match_points(new_pts, old_pts, max_px=max_px)
    logger.info("Matched %d fiducials (new=%d, old=%d, max_px=%.1f)",
                len(pairs), len(new_pts), len(old_pts), max_px)

    # ---------------- Case A: full model (>=3 matches) ----------------
    if len(pairs) >= 3:
        mode = "full"
        new_m = new_pts[pairs[:, 0]]
        old_m = old_pts[pairs[:, 1]]

        # mpp from matched NEW hull (more stable)
        try:
            mpp = microns_per_pixel(np.squeeze(cv.convexHull(new_m.astype(np.float32))))
        except Exception:
            pass

        # RANSAC similarity (rotation + translation + uniform scale)
        M, inliers = cv.estimateAffinePartial2D(
            old_m, new_m, method=cv.RANSAC,
            ransacReprojThreshold=max(3.0, 0.2 * max_px),
            maxIters=2000, confidence=0.99, refineIters=50
        )
        if M is None or inliers is None:
            # degrade gracefully to translation-only from the pairs
            logger.warning("Similarity fit failed; degrading to translation-only.")
            mode = "translation-only"
            trans_px = (new_m - old_m).mean(axis=0)
            if mpp is not None:
                mean_shift_um = float(np.linalg.norm(trans_px * mpp))
                diffs_um = (new_m - old_m) * mpp
                per_avg = float(np.mean(np.linalg.norm(diffs_um, axis=1)))
            else:
                mean_shift_um = None
                diffs_um = new_m - old_m  # px
                per_avg = float(np.mean(np.linalg.norm(diffs_um, axis=1)))
            status = "OK"
            reasons = ["translation-only (fit failed)"]
            # Only evaluate shift thresholds if we have µm
            if mean_shift_um is not None:
                if mean_shift_um > th["shift_um_fail"]:
                    status = bump(status, "FAIL"); reasons.append(f"mean shift {mean_shift_um:.1f}µm > {th['shift_um_fail']}µm")
                elif mean_shift_um > th["shift_um_warn"]:
                    status = bump(status, "WARN"); reasons.append(f"mean shift {mean_shift_um:.1f}µm > {th['shift_um_warn']}µm")
            logger.info("Alignment status: %s | %s", status, "; ".join(reasons))
            return status, dict(
                mode=mode, matches=int(len(pairs)), inlier_frac=None,
                mean_shift_um=mean_shift_um, rot_deg=None, scale_pct=None,
                rms_um=None, mpp=(None if mpp is None else float(mpp)),
                reasons=reasons,
            )

        inliers = inliers.ravel().astype(bool)
        inf = inliers.sum() / len(inliers)

        R = M[:2, :2]; t = M[:2, 2]
        angle_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        sx = np.linalg.norm(R[:, 0]); sy = np.linalg.norm(R[:, 1])
        scale = float((sx + sy) / 2.0); scale_pct = (scale - 1.0) * 100.0

        pred = (old_m @ R.T) + t
        resid_px = np.linalg.norm(pred - new_m, axis=1)
        if mpp is not None:
            resid_um = resid_px * mpp
            rms_um = float(np.sqrt(np.mean(resid_um[inliers] ** 2)))
        else:
            rms_um = None

        mean_vec_px = (new_m - old_m).mean(axis=0)
        mean_shift_um = float(np.linalg.norm(mean_vec_px) * mpp) if mpp is not None else None

        logger.info("Rigid fit: rotation=%.3f deg, scale=%.3f%%, translation=(%.2f, %.2f) px "
                    "(≈ (%.2f, %.2f) µm), inlier_frac=%.2f, RMS(inliers)=%s",
                    angle_deg, scale_pct, t[0], t[1],
                    (t[0]*mpp if mpp is not None else float('nan')),
                    (t[1]*mpp if mpp is not None else float('nan')),
                    inf, (f"{rms_um:.2f} µm" if rms_um is not None else "N/A"))

        # ---- Decision ----
        status = "OK"; reasons = []
        if inf < th["inlier_frac_fail"]:
            status = bump(status, "FAIL"); reasons.append(f"inlier fraction {inf:.2f} < {th['inlier_frac_fail']:.2f}")

        # Only score metrics we actually have
        if abs(angle_deg) > th["rot_deg_fail"]:
            status = bump(status, "FAIL"); reasons.append(f"rotation {abs(angle_deg):.3g}deg > {th['rot_deg_fail']}deg")
        elif abs(angle_deg) > th["rot_deg_warn"]:
            status = bump(status, "WARN"); reasons.append(f"rotation {abs(angle_deg):.3g}deg > {th['rot_deg_warn']}deg")

        if abs(scale_pct) > th["scale_pct_fail"]:
            status = bump(status, "FAIL"); reasons.append(f"scale change {abs(scale_pct):.3g}% > {th['scale_pct_fail']}%")
        elif abs(scale_pct) > th["scale_pct_warn"]:
            status = bump(status, "WARN"); reasons.append(f"scale change {abs(scale_pct):.3g}% > {th['scale_pct_warn']}%")

        if mean_shift_um is not None:
            if mean_shift_um > th["shift_um_fail"]:
                status = bump(status, "FAIL"); reasons.append(f"mean shift {mean_shift_um:.1f}µm > {th['shift_um_fail']}µm")
            elif mean_shift_um > th["shift_um_warn"]:
                status = bump(status, "WARN"); reasons.append(f"mean shift {mean_shift_um:.1f}µm > {th['shift_um_warn']}µm")

        if rms_um is not None:
            if rms_um > th["rms_um_fail"]:
                status = bump(status, "FAIL"); reasons.append(f"RMS residual {rms_um:.1f}µm > {th['rms_um_fail']}µm")
            elif rms_um > th["rms_um_warn"]:
                status = bump(status, "WARN"); reasons.append(f"RMS residual {rms_um:.1f}µm > {th['rms_um_warn']}µm")

        if not reasons:
            reasons = ["within limits"]

        logger.info("Alignment status: %s | %s", status, "; ".join(reasons))
        return status, dict(
            mode=mode, matches=int(len(pairs)), inlier_frac=float(inf),
            mean_shift_um=mean_shift_um, rot_deg=float(angle_deg),
            scale_pct=float(scale_pct), rms_um=rms_um,
            mpp=(None if mpp is None else float(mpp)),
            reasons=reasons,
        )

    # ---------------- Case B: translation-only (1–2 matches) ----------------
    if 1 <= len(pairs) <= 2:
        mode = "translation-only"
        new_m = new_pts[pairs[:, 0]]
        old_m = old_pts[pairs[:, 1]]
        trans_px = (new_m - old_m).mean(axis=0)
        diffs_px = (new_m - old_m)

        if mpp is not None:
            mean_shift_um = float(np.linalg.norm(trans_px * mpp))
            per_avg_um = float(np.mean(np.linalg.norm(diffs_px * mpp, axis=1)))
        else:
            mean_shift_um = None
            per_avg_um = None

        status = "OK"; reasons = [f"{mode}: limited matches ({len(pairs)})"]
        if mean_shift_um is not None:
            if mean_shift_um > th["shift_um_fail"]:
                status = bump(status, "FAIL"); reasons.append(f"mean shift {mean_shift_um:.1f}µm > {th['shift_um_fail']}µm")
            elif mean_shift_um > th["shift_um_warn"]:
                status = bump(status, "WARN"); reasons.append(f"mean shift {mean_shift_um:.1f}µm > {th['shift_um_warn']}µm")

        logger.info("Alignment status: %s | %s", status, "; ".join(reasons))
        return status, dict(
            mode=mode, matches=int(len(pairs)), inlier_frac=None,
            mean_shift_um=mean_shift_um, rot_deg=None, scale_pct=None,
            rms_um=None, mpp=(None if mpp is None else float(mpp)),
            reasons=reasons,
        )

    # ---------------- Case C: ECC fallback (0 matches) ----------------
    mode = "ecc-fallback"
    logger.warning("No matched fiducials; using ECC/phase correlation fallback.")
    angle_deg, tx_px, ty_px, corr = _estimate_ecc(old_file, new_file)

    if mpp is not None:
        dx_um, dy_um = tx_px * mpp, ty_px * mpp
        mean_shift_um = float(np.hypot(dx_um, dy_um))
    else:
        mean_shift_um = None

    status = "OK"; reasons = [f"{mode}: corr={corr:.3f}"]
    # Evaluate whatever we have
    if abs(angle_deg) > th["rot_deg_fail"]:
        status = bump(status, "FAIL"); reasons.append(f"rotation {abs(angle_deg):.3g}deg > {th['rot_deg_fail']}deg")
    elif abs(angle_deg) > th["rot_deg_warn"]:
        status = bump(status, "WARN"); reasons.append(f"rotation {abs(angle_deg):.3g}deg > {th['rot_deg_warn']}deg")

    if mean_shift_um is not None:
        if mean_shift_um > th["shift_um_fail"]:
            status = bump(status, "FAIL"); reasons.append(f"mean shift {mean_shift_um:.1f}µm > {th['shift_um_fail']}µm")
        elif mean_shift_um > th["shift_um_warn"]:
            status = bump(status, "WARN"); reasons.append(f"mean shift {mean_shift_um:.1f}µm > {th['shift_um_warn']}µm")

    logger.info("Alignment status: %s | %s", status, "; ".join(reasons))
    return status, dict(
        mode=mode, matches=0, inlier_frac=None,
        mean_shift_um=mean_shift_um, rot_deg=float(angle_deg), scale_pct=None,
        rms_um=None, mpp=(None if mpp is None else float(mpp)),
        reasons=reasons,
    )
