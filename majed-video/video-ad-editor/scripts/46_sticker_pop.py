#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
46_sticker_pop.py — ملصق 3D يطلع فجأة بدوران سريع (نسخة كراج موسى محسن TRD)

الأوامر:
  sticker <logo.png|jpg> <out.png> [--size 900] [--base "#0E0C0C"] [--border 0.018] [--gap 0.022]
        يقص الشعار من خلفيته (شفافة، أو لون ثابت مثل الأسود) ويسويه ملصق مقصوص:
        حافة بيضا بلمعة + قاعدة غامقة ورا الشعار — نفس ستايل ملصق الكراج.

  pop <video> <out.mp4> --sticker s.png --at 3.2 [--at 7.9 ...] [--pos 540,560] [--size 520]
        [--from x,y] [--in 0.30] [--hold 1.0] [--out 0.20] [--spin 540] [--no-sfx] [--overlay o.mov]
        الملصق يطلع بلحظة الكلمة: يكبر من صفر بارتداد وهو يلف بسرعة (غباش حركة حقيقي)،
        يوقف شوي بلمعة تعبره، ويطلع بسرعة وهو يلف. المدة كلها ~1.5 ث.
        - --at بالثواني (كرّره لأكثر من مرة). --pos = مكانه (وسط الملصق) · --size = عرضه بالبكسل.
        - --from x,y = يطلع من نقطة (مثلاً شعار السيارة الحقيقي بالفريم) ويطير لمكانه وهو يلف.
        - الصوت: سحبة مع الدخول + «بوب» لما يستقر + سحبة خفيفة مع الطلعة (بنك الأصوات CC0)،
          فوق صوت الفيديو (أخف لو فيه كلام). --no-sfx بلا أصوات.
        - --overlay o.mov = نفس الحركة لحالها بخلفية شفافة (ProRes 4444) لبرامج المونتاج.

⛔ المنطقة الآمنة: لا تحطه فوق y 150 ولا تحت 1500، ولا على وجه المتكلم ولا على الكابشن —
   بريل الكلام مكانه الطبيعي فوق (y 300-560) مثل أي رسم بـR_DOWN.
"""
import argparse, json, math, os, subprocess, sys

try:
    import cv2
    import numpy as np
except ImportError:
    print("⚠️ ناقص opencv — شغّل: bash scripts/00_setup.sh --install", file=sys.stderr)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 48000


# ─────────────────────────── الملصق ───────────────────────────

def hex2bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def cutout(img):
    """قص الشعار: لو فيه شفافية ناخذها، وإلا لون الحواف = الخلفية ونمسح المتصل فيها بس (الفراغات الداخلية تبقى)."""
    if img.ndim == 3 and img.shape[2] == 4 and img[..., 3].min() < 250:
        return img
    bgr = img[..., :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    border = np.concatenate([bgr[0], bgr[-1], bgr[:, 0], bgr[:, -1]]).astype(np.float32)
    bgc = np.median(border, 0)
    d = np.linalg.norm(bgr.astype(np.float32) - bgc, axis=2)
    near = (d < 30).astype(np.uint8)
    n, lab = cv2.connectedComponents(near, connectivity=4)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(edge))
    alpha = np.where(bg, 0.0, 255.0).astype(np.float32)
    band = cv2.dilate(bg.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & \
        cv2.dilate((~bg).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    alpha[band] = np.clip((d[band] - 8) / 40, 0, 1) * 255
    out = np.dstack([bgr, alpha.astype(np.uint8)])
    ys, xs = np.where(out[..., 3] > 8)
    if len(xs) == 0:
        sys.exit("⚠️ ما لقيت شعار بالصورة — الخلفية لازم لون ثابت أو شفافة")
    m = 6
    return out[max(0, ys.min() - m):ys.max() + m, max(0, xs.min() - m):xs.max() + m]


def make_sticker(rgba, size=900, base=(14, 12, 12), border=0.018, gap=0.022):
    s = size / max(rgba.shape[:2])
    L = cv2.resize(rgba, (max(1, int(rgba.shape[1] * s)), max(1, int(rgba.shape[0] * s))), interpolation=cv2.INTER_AREA)
    h, w = L.shape[:2]
    P = int(size * 0.09)
    H, W = h + 2 * P, w + 2 * P
    a = np.zeros((H, W), np.uint8)
    a[P:P + h, P:P + w] = L[..., 3]

    def dil(m, r):
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    rg, rb = max(3, int(size * gap)), max(3, int(size * border))
    base_m = dil((a > 40).astype(np.uint8) * 255, rg)
    k = max(5, int(size * 0.06)) | 1
    base_m = cv2.morphologyEx(base_m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    sm = max(1.5, size * 0.006)                    # حافة القص ناعمة (مو مسننة)
    base_m = ((cv2.GaussianBlur(base_m, (0, 0), sm) > 127) * 255).astype(np.uint8)
    ring = dil(base_m, rb)
    ring = ((cv2.GaussianBlur(ring, (0, 0), sm) > 127) * 255).astype(np.uint8)
    ring = cv2.GaussianBlur(ring, (0, 0), 1.1)
    base_m = cv2.GaussianBlur(base_m, (0, 0), 0.9)
    out = np.zeros((H, W, 4), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    grad = 0.86 + 0.14 * (1 - (yy / H * 0.6 + xx / W * 0.4))          # حافة بيضا بإضاءة من فوق-يسار
    out[..., :3] = np.dstack([grad * 246] * 3)
    out[..., 3] = ring
    bm = base_m[..., None] / 255.0
    out[..., :3] = out[..., :3] * (1 - bm) + np.array(base, np.float32) * bm
    la = np.zeros((H, W), np.float32)
    la[P:P + h, P:P + w] = L[..., 3] / 255.0
    lc = np.zeros((H, W, 3), np.float32)
    lc[P:P + h, P:P + w] = L[..., :3]
    out[..., :3] = out[..., :3] * (1 - la[..., None]) + lc * la[..., None]
    sheen = np.clip(1 - np.abs((xx * 0.55 + yy) - W * 0.33) / (W * 0.06), 0, 1) * 0.16 * (ring / 255.0)
    out[..., :3] = out[..., :3] * (1 - sheen[..., None]) + 255 * sheen[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def cmd_sticker(a):
    img = cv2.imread(a.logo, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"⚠️ ما قدرت أقرا {a.logo}")
    st = make_sticker(cutout(img), a.size, hex2bgr(a.base), a.border, a.gap)
    cv2.imwrite(a.out, st)
    print(f"✅ {a.out} ({st.shape[1]}x{st.shape[0]}) — ملصق مقصوص بحافة بيضا")


# ─────────────────────────── الحركة ───────────────────────────

def ease_out_cubic(k):
    return 1 - (1 - k) ** 3


def ease_out_back(k, c=1.9):
    return 1 + (c + 1) * (k - 1) ** 3 + c * (k - 1) ** 2


def ease_in_back(k, c=1.7):
    return (c + 1) * k ** 3 - c * k ** 2


def pose(tl, a, to, frm):
    """(مقياس، زاوية، مكان، لمعة 0-1) بلحظة tl من بداية الطلعة — أو None برّا المدة."""
    tin, hold, tout = a.tin, a.hold, a.tout
    if tl < 0 or tl > tin + hold + tout:
        return None
    glint = -1.0
    if tl < tin:
        k = tl / tin
        sc = max(0.0, ease_out_back(k))
        ang = -a.spin * (1 - ease_out_cubic(k))
        e = ease_out_cubic(min(1.0, k * 1.15))
        pos = frm + (to - frm) * e
    elif tl < tin + hold:
        h = tl - tin
        sc = 1.0 + 0.012 * math.sin(h * 7)
        ang = 6 * math.sin(h * 10) * math.exp(-h * 3.5)                  # يرتجّ شوي ويستقر
        pos = to + np.array([0.0, -5 * math.sin(h * 3.4)])
        g0 = 0.06
        if g0 <= h <= g0 + 0.38:
            glint = (h - g0) / 0.38
    else:
        k = (tl - tin - hold) / tout
        sc = max(0.0, 1 - ease_in_back(k))
        ang = 240 * k * k
        pos = to + np.array([0.0, -6.0])
    return sc, ang, pos, glint


def with_glint(st, g):
    if g < 0:
        return st
    H, W = st.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    c = -0.35 * W + g * 1.7 * W
    band = np.exp(-(((xx * 0.75 + yy * 0.55) - c) / (W * 0.07)) ** 2) * 0.55
    out = st.astype(np.float32)
    al = out[..., 3:4] / 255
    out[..., :3] = out[..., :3] + (255 - out[..., :3]) * band[..., None] * al
    return np.clip(out, 0, 255).astype(np.uint8)


def render_sticker(frame_shape, st, sc, ang, pos, size):
    """يرسم الملصق (premultiplied) بمنطقة محدودة — يرجّع (x0,y0, rgb_premul, alpha) أو None."""
    Hf, Wf = frame_shape[:2]
    h, w = st.shape[:2]
    s = sc * size / w
    if s <= 0.004:
        return None
    R = 0.5 * math.hypot(w, h) * s + 4
    x0, y0 = int(pos[0] - R), int(pos[1] - R)
    x1, y1 = int(pos[0] + R) + 1, int(pos[1] + R) + 1
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(Wf, x1), min(Hf, y1)
    if cx1 - cx0 < 2 or cy1 - cy0 < 2:
        return None
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
    M[0, 2] += pos[0] - w / 2 - cx0
    M[1, 2] += pos[1] - h / 2 - cy0
    pm = st.astype(np.float32)
    pm[..., :3] *= pm[..., 3:4] / 255
    wp = cv2.warpAffine(pm, M, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return cx0, cy0, wp


def compose_at(frame_shape, t, events, st, a, fps):
    """كل الملصقات بلحظة t (مع غباش حركة بالعيّنات الفرعية) ← لوحة RGBA كاملة premultiplied أو None."""
    Hf, Wf = frame_shape[:2]
    acc = None
    for at, to, frm in events:
        tl = t - at
        p = pose(tl, a, to, frm)
        if p is None:
            continue
        fast = tl < a.tin or tl > a.tin + a.hold
        N = 7 if fast else 1
        layer = np.zeros((Hf, Wf, 4), np.float32)
        for i in range(N):
            dt = ((i / (N - 1)) - 0.5) * (0.5 / fps) if N > 1 else 0.0
            q = pose(tl + dt, a, to, frm)
            if q is None:
                continue
            sc, ang, pos, g = q
            r = render_sticker(frame_shape, with_glint(st, g), sc, ang, pos, a.size)
            if r is None:
                continue
            x0, y0, wp = r
            layer[y0:y0 + wp.shape[0], x0:x0 + wp.shape[1]] += wp / N
        acc = layer if acc is None else acc + layer * (1 - acc[..., 3:4] / 255)
    return acc


def shadow_of(layer, size):
    a = layer[..., 3] / 255
    if a.max() <= 0:
        return None
    sig = max(2.0, size * 0.035)
    sh = cv2.GaussianBlur(a, (0, 0), sig)
    M = np.float32([[1, 0, size * 0.03], [0, 1, size * 0.06]])
    sh = cv2.warpAffine(sh, M, (a.shape[1], a.shape[0]))
    return np.clip(sh * 0.45, 0, 1)


# ─────────────────────────── الصوت ───────────────────────────

def load_bank():
    try:
        z = np.load(os.path.join(HERE, "..", "sfx", "bank.npz"))
        ix = json.loads(bytes(z["_index"]).decode("utf-8"))
        return {k: [z[x["n"]].astype(np.float32) / 32767.0 for x in v] for k, v in ix.items()}
    except Exception as e:  # noqa
        print("⚠️ بنك الأصوات ما انقرى —", e)
        return {}


def sfx_track(events, a, dur, bank):
    y = np.zeros(int((dur + 1) * SR), np.float32)
    rng = np.random.RandomState(7)

    def put(kind, t, peak):
        v = bank.get(kind)
        if not v:
            return
        x = v[rng.randint(len(v))]
        x = x / (np.abs(x).max() + 1e-9) * peak
        i = int(max(0, t) * SR)
        j = min(len(y), i + len(x))
        y[i:j] += x[:j - i]
    for at, _, _ in events:
        put("swish", at - 0.03, 0.50)
        put("pop", at + a.tin - 0.04, 0.55)
        put("swish", at + a.tin + a.hold - 0.02, 0.28)
    return y[:int(dur * SR)]


# ─────────────────────────── التشغيل ───────────────────────────

def probe(path):
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=width,height,r_frame_rate:format=duration", "-of", "json", path]))
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    has_audio = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                     "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip())
    return int(s["width"]), int(s["height"]), float(num) / float(den or 1), float(j["format"]["duration"]), has_audio


def cmd_pop(a):
    W, H, fps, dur, has_audio = probe(a.video)
    st = cv2.imread(a.sticker, cv2.IMREAD_UNCHANGED)
    if st is None or st.ndim != 3 or st.shape[2] != 4:
        sys.exit("⚠️ --sticker لازم PNG بشفافية — سوّه أول: 46_sticker_pop.py sticker <شعار> <ملصق.png>")
    to = np.array([float(v) for v in a.pos.split(",")], np.float32)
    frm = np.array([float(v) for v in a.frm.split(",")], np.float32) if a.frm else to.copy()
    events = [(float(t), to, frm) for t in a.at]
    total = a.tin + a.hold + a.tout
    print(f"⏳ {len(events)} طلعة × {total:.2f} ث …")
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.video, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE)
    tmp_v = a.out + ".v.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp_v], stdin=subprocess.PIPE)
    ov = None
    if a.overlay:
        ov = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}",
                               "-r", f"{fps:.6f}", "-i", "-", "-c:v", "prores_ks", "-profile:v", "4444",
                               "-pix_fmt", "yuva444p10le", a.overlay], stdin=subprocess.PIPE)
    size, n = W * H * 3, 0
    while True:
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32)
        t = n / fps
        layer = compose_at(f.shape, t, events, st, a, fps) if any(0 <= t - e[0] <= total + 0.05 for e in events) else None
        if layer is not None:
            sh = shadow_of(layer, a.size)
            if sh is not None:
                f *= (1 - sh[..., None])
            f = f * (1 - layer[..., 3:4] / 255) + layer[..., :3]
        enc.stdin.write(np.clip(f, 0, 255).astype(np.uint8).tobytes())
        if ov:
            o = np.zeros((H, W, 4), np.float32)
            if layer is not None:
                la = layer[..., 3] / 255
                sh = shadow_of(layer, a.size)
                A = la + (sh if sh is not None else 0) * (1 - la)      # الظل أسود: يزيد الشفافية بس
                o[..., :3] = np.where(A[..., None] > 1e-3, layer[..., :3] / np.maximum(A[..., None], 1e-3), 0)
                o[..., 3] = A * 255
            ov.stdin.write(np.clip(o, 0, 255).astype(np.uint8).tobytes())
        n += 1
    for p in [enc] + ([ov] if ov else []):
        p.stdin.close()
        p.wait()
    dec.wait()
    # ---- الصوت ----
    if a.no_sfx:
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", tmp_v] + (["-i", a.video, "-map", "0:v", "-map", "1:a", "-c:a", "copy"] if has_audio else []) + \
              ["-c:v", "copy", "-movflags", "+faststart", a.out]
        subprocess.run(cmd, check=True)
    else:
        bank = load_bank()
        fx = sfx_track(events, a, n / fps, bank)
        peak = 0.25 if has_audio else 0.40          # تحت الكلام لو فيه صوت
        fx = fx / (np.abs(fx).max() + 1e-9) * peak
        wav = a.out + ".fx.wav"
        import wave
        with wave.open(wav, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
            wf.writeframes((np.clip(fx, -1, 1) * 32767).astype(np.int16).tobytes())
        if has_audio:
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", tmp_v, "-i", a.video, "-i", wav, "-filter_complex",
                   "[1:a]aresample=48000[a0];[2:a]aresample=48000,pan=stereo|c0=c0|c1=c0[a1];[a0][a1]amix=inputs=2:normalize=0[a]",
                   "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", a.out]
        else:
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", tmp_v, "-i", wav, "-map", "0:v", "-map", "1:a",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-shortest", "-movflags", "+faststart", a.out]
        subprocess.run(cmd, check=True)
        os.remove(wav)
    os.remove(tmp_v)
    print(f"✅ {a.out} — {n} فريم · الملصق طلع {len(events)} مرة ({', '.join(f'{e[0]:g}ث' for e in events)})"
          + (f" · شفاف: {a.overlay}" if a.overlay else ""))


def main():
    p = argparse.ArgumentParser(description="ملصق 3D يطلع فجأة بدوران سريع")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sticker"); s.add_argument("logo"); s.add_argument("out")
    s.add_argument("--size", type=int, default=900); s.add_argument("--base", default="#0E0C0C")
    s.add_argument("--border", type=float, default=0.018); s.add_argument("--gap", type=float, default=0.022)
    q = sp.add_parser("pop"); q.add_argument("video"); q.add_argument("out")
    q.add_argument("--sticker", required=True); q.add_argument("--at", action="append", required=True)
    q.add_argument("--pos", default="540,430"); q.add_argument("--size", type=float, default=520)
    q.add_argument("--from", dest="frm"); q.add_argument("--in", dest="tin", type=float, default=0.30)
    q.add_argument("--hold", type=float, default=1.0); q.add_argument("--out", dest="tout", type=float, default=0.20)
    q.add_argument("--spin", type=float, default=540); q.add_argument("--no-sfx", action="store_true")
    q.add_argument("--overlay")
    a = p.parse_args()
    {"sticker": cmd_sticker, "pop": cmd_pop}[a.cmd](a)


if __name__ == "__main__":
    main()
