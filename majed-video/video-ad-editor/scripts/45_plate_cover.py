#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
45_plate_cover.py — تغطية لوحات السيارات (نسخة كراج موسى محسن TRD)

قاعدة الكراج: رقم لوحة سيارة الزبون ما يبين أبداً — يتغطّى بتصميم لوحة الكراج.

الأوامر:
  grid  <video> <sec> [out.jpg] [--frame N] [--crop x,y,w,h --step 10 --zoom 4]
        فريم بوقته (أو برقمه --frame) عليه شبكة بأرقام — تقرا منه مكان اللوحة بإحداثيات الفيديو.
        --crop = تكبير منطقة اللوحة بشبكة كل 10px: تقرا منه زوايا اللوحة الأربع بدقة بكسل.

  track <video> <out.mp4> --quad "t,x1,y1,x2,y2,x3,y3,x4,y4" [--quad ...] [--auto-cuts]
        ⭐ الطريقة الأدق (تتبّع مستوي/Planar): الزوايا الأربع للوحة (فوق-يسار، فوق-يمين، تحت-يمين، تحت-يسار)
        بفريم مرجعي وحد بكل لقطة — والسكربت يتبع سطح اللوحة نفسه فريم فريم (نقاط على اللوحة + ECC)،
        فالغطاء يلصق على اللوحة بمقاسها وزاويتها ومنظورها ويمشي مع الكاميرا بلا انزلاق.
        - t بالثواني، أو "f123" = رقم الفريم.
        - أكثر من --quad باللقطة الوحدة = مراسي إضافية (اللقطات الطويلة أو اللي فيها حركة قوية).
        - التصميم ينبني بمقاس اللوحة نفسها: أسود + إطار ذهبي + الشعار بالنص (--logo)،
          واللوحات العريضة (4:1 وأكثر) تاخذ تصميم الكراج الكامل (--full). أو صورتك بـ--cover.
        - رقم عاشر بالـ--quad = نسبة اللوحة الحقيقية (عرض÷ارتفاع) لما الزاوية الجانبية تغشّ الشكل.
        - غباش الفوكس تلقائي (يقارن بنفس اللوحة لو واضحة بلقطة ثانية، وإلا حدّة الحواف) + غباش حركة
          الكاميرا (--shutter). --soft <px> يثبّته يدوي.
        - --debug <dbg.mp4> = نفس الفيديو بخط حول اللوحة المتتبَّعة (بلا غطاء) — للفحص.

  cover <video> <out.mp4> --key "t,x,y,w,h" [--key ...] [--cover plate.png] [--track] [--pad 0.12]
        يغطي اللوحة بكل فريم:
        - مفتاح واحد  = مربع ثابت (لقطة ثابتة).
        - عدة مفاتيح = المربع ينتقل بين المفاتيح بنعومة (خطي).
        - --track     = يتبع اللوحة بين المفاتيح (OpenCV MIL)، ويرجع لمكان المفتاح عند كل مفتاح.
        - مفتاح بـ6 أرقام "t,x,y,w,h,زاوية" = لوحة مايلة: x,y,w,h للوحة قبل الميلان حول مركزها.
        - --auto-cuts / --cut = فيديو فيه أكثر من لقطة: المربع ما ينزلق من لقطة للثانية،
                        ولقطة بلا مفتاح ما تتغطى (يعني ما فيها لوحة).
        - --cover     = صورة لوحة الكراج (PNG). بدونها: تغبيش قوي + إطار، ما يُقرا منه شي.
                        كرّرها (العريض والقصير) ويختار الأقرب لنسبة كل لوحة.
                        تحافظ على نسبة التصميم وتكبر حول مركز اللوحة حتى تغطيها كاملة (--stretch = مطّ بدل كذا).
        - --from/--to = يغطي بس بين هالثانيتين (الباقي يمشي كما هو).
        الصوت ينسخ كما هو.

  check <out.mp4> [--every 1.0]
        ورقة فريمات من الناتج (كل ثانية) — افتحها وتأكد بعينك إن ولا لوحة مقروءة.

الإحداثيات بالبكسل على مقاس الفيديو الأصلي (بعد الدوران)، والمربع (x,y) = الزاوية فوق-يسار.
"""
import argparse, json, math, os, subprocess, sys

try:
    import cv2
    import numpy as np
except ImportError:
    print("⚠️ ناقص opencv — شغّل: bash scripts/00_setup.sh --install", file=sys.stderr)
    sys.exit(1)


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:stream_side_data=rotation:format=duration",
        "-of", "json", path])
    j = json.loads(out)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den or 1)
    w, h = int(s["width"]), int(s["height"])
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    has_audio = bool(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip())
    return w, h, fps, float(j["format"].get("duration", 0)), has_audio


def read_frame(path, sec):
    w, h, *_ = probe(path)
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-ss", str(sec), "-i", path, "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()


def read_frame_n(path, n):
    w, h, *_ = probe(path)
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", path, "-vf", f"select=eq(n\\,{int(n)})", "-vsync", "0",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()


def cmd_grid(a):
    try:
        a.sec = float(a.sec)
    except ValueError:  # مع --frame: الثاني اسم الملف
        a.out, a.sec = a.out or a.sec, 0.0
    if a.frame is not None:
        img, tag = read_frame_n(a.video, a.frame), f"f{a.frame}"
    else:
        img, tag = read_frame(a.video, a.sec), f"{a.sec:g}"
    h, w = img.shape[:2]
    if a.crop:
        x, y, cw, ch = [int(v) for v in a.crop.split(",")]
        x, y = max(0, x), max(0, y); cw, ch = min(cw, w - x), min(ch, h - y)
        z, st = a.zoom, a.step
        big = cv2.resize(img[y:y + ch, x:x + cw], (cw * z, ch * z), interpolation=cv2.INTER_CUBIC)
        for gx in range((x // st + 1) * st, x + cw, st):
            X, major = (gx - x) * z, gx % (st * 5) == 0
            cv2.line(big, (X, 0), (X, ch * z), (0, 255, 255) if major else (0, 150, 255), 1)
            if major:
                cv2.putText(big, str(gx), (X + 3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for gy in range((y // st + 1) * st, y + ch, st):
            Y, major = (gy - y) * z, gy % (st * 5) == 0
            cv2.line(big, (0, Y), (cw * z, Y), (0, 255, 255) if major else (0, 150, 255), 1)
            if major:
                cv2.putText(big, str(gy), (3, Y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        out = a.out or os.path.splitext(a.video)[0] + f"_zoom_{tag}.jpg"
        cv2.imwrite(out, big, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"✅ {out} — خطوط كل {st}px (الأرقام كل {st * 5}). اقرا الزوايا الأربع واكتبها: --quad \"{tag},x1,y1,x2,y2,x3,y3,x4,y4\"")
        return
    for x in range(0, w, 100):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255) if x % 500 == 0 else (0, 180, 255), 1)
        cv2.putText(img, str(x), (x + 3, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for y in range(0, h, 100):
        cv2.line(img, (0, y), (w, y), (0, 255, 255) if y % 500 == 0 else (0, 180, 255), 1)
        cv2.putText(img, str(y), (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    out = a.out or os.path.splitext(a.video)[0] + f"_grid_{tag}.jpg"
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"✅ {out}  ({w}x{h}) — اقرا مكان اللوحة واكتبه: --key \"{tag},x,y,w,h\" — وللتتبّع المستوي كبّر بـ--crop")


def parse_keys(keys):
    ks = []
    for k in keys:
        p = [float(v) for v in k.split(",")]
        if len(p) == 5:
            p.append(0.0)
        if len(p) != 6:
            sys.exit(f"⚠️ المفتاح لازم t,x,y,w,h أو t,x,y,w,h,زاوية — وصلني: {k}")
        ks.append(p)
    return sorted(ks)


def interp(ks, t):
    if t <= ks[0][0]:
        return ks[0][1:]
    if t >= ks[-1][0]:
        return ks[-1][1:]
    for a, b in zip(ks, ks[1:]):
        if a[0] <= t <= b[0]:
            u = 0 if b[0] == a[0] else (t - a[0]) / (b[0] - a[0])
            u = u * u * (3 - 2 * u)  # نعومة
            return [a[i] + (b[i] - a[i]) * u for i in range(1, 6)]


def detect_cuts(path, thr=25.0):
    """قطعات اللقطات (ثواني) — فرق الفريمات المتتالية بصورة صغيرة رمادية."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    prev, i, cuts = None, 0, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (108, 192)).astype(np.float32)
        if prev is not None and np.abs(g - prev).mean() > thr:
            cuts.append(i / fps)
        prev, i = g, i + 1
    return cuts


def segment_of(t, cuts):
    return sum(1 for c in cuts if t >= c - 1e-6)


def load_cover(path):
    if not path:
        return None
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        sys.exit(f"⚠️ ما قدرت أقرا صورة اللوحة: {path}")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    if im.shape[2] == 3:
        im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
    return im


def pick_cover(covers, w, h):
    """أقرب تصميم لنسبة اللوحة (اللوحات الخلفية القصيرة تاخذ التصميم القصير)."""
    if not covers:
        return None
    r = w / max(h, 1)
    return min(covers, key=lambda c: abs(math.log((c.shape[1] / c.shape[0]) / r)))


def paint(frame, box, covers, pad, stretch=False, cover=None):
    H, W = frame.shape[:2]
    x, y, w, h, ang = box
    if cover is None:
        cover = pick_cover(covers, w, h)
    if abs(ang) > 0.5:
        return paint_rotated(frame, (x, y, w, h, ang), cover, pad, stretch)
    if cover is not None and not stretch:
        # نحافظ على نسبة التصميم: نكبّر الضلع الناقص (حول نفس المركز) حتى يغطي اللوحة كاملة
        ar = cover.shape[1] / cover.shape[0]
        cx, cy = x + w / 2, y + h / 2
        if w / h < ar:
            w = h * ar
        else:
            h = w / ar
        x, y = cx - w / 2, cy - h / 2
    px, py = w * pad, h * pad
    x0, y0 = int(max(0, math.floor(x - px))), int(max(0, math.floor(y - py)))
    x1, y1 = int(min(W, math.ceil(x + w + px))), int(min(H, math.ceil(y + h + py)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    roi = frame[y0:y1, x0:x1]
    rw, rh = x1 - x0, y1 - y0
    if cover is not None:
        c = cv2.resize(cover, (rw, rh), interpolation=cv2.INTER_AREA)
        al = c[:, :, 3:4].astype(np.float32) / 255
        # تحت الصورة دايماً تغبيش — لو الصورة فيها شفافية ما يبين الرقم من وراها
        base = cv2.GaussianBlur(roi, (0, 0), max(rw, rh) / 6 + 1)
        roi[:] = (c[:, :, :3] * al + base * (1 - al)).astype(np.uint8)
    else:
        small = cv2.resize(roi, (max(1, rw // 12), max(1, rh // 12)), interpolation=cv2.INTER_AREA)
        roi[:] = cv2.GaussianBlur(cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST),
                                  (0, 0), max(rw, rh) / 10 + 1)
        cv2.rectangle(roi, (0, 0), (rw - 1, rh - 1), (30, 30, 30), max(2, rh // 18))


def paint_rotated(frame, box, cover, pad, stretch):
    """لوحة مايلة: x,y,w,h = اللوحة قبل الميلان (حول مركزها)، والزاوية بالدرجات (موجب = مع عقارب الساعة)."""
    H, W = frame.shape[:2]
    x, y, w, h, ang = box
    cx, cy = x + w / 2, y + h / 2
    if cover is not None and not stretch:
        ar = cover.shape[1] / cover.shape[0]
        if w / h < ar:
            w = h * ar
        else:
            h = w / ar
    w, h = w * (1 + 2 * pad), h * (1 + 2 * pad)
    th = math.radians(ang)
    co, si = math.cos(th), math.sin(th)
    # tl, tr, br, bl — الصورة محورها y لتحت، فالزاوية الموجبة = مع عقارب الساعة
    pts = np.float32([[cx + dx * co - dy * si, cy + dx * si + dy * co]
                      for dx, dy in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]])
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, pts.astype(np.int32), 255, cv2.LINE_AA)
    bx, by, bw, bh = cv2.boundingRect(pts.astype(np.int32))
    x0, y0, x1, y1 = max(0, bx - 4), max(0, by - 4), min(W, bx + bw + 4), min(H, by + bh + 4)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    roi = frame[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1].astype(np.float32)[..., None] / 255
    base = cv2.GaussianBlur(roi, (0, 0), max(w, h) / 6 + 1)
    if cover is None:
        small = cv2.resize(roi, (max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 12)), interpolation=cv2.INTER_AREA)
        top = cv2.GaussianBlur(cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST), (0, 0), max(w, h) / 10 + 1)
        al = m
    else:
        ch, cw = cover.shape[:2]
        src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
        M = cv2.getPerspectiveTransform(src, pts - np.float32([x0, y0]))
        warped = cv2.warpPerspective(cover, M, (x1 - x0, y1 - y0), flags=cv2.INTER_AREA,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        top = warped[:, :, :3].astype(np.float32)
        ca = warped[:, :, 3:4].astype(np.float32) / 255
        # تحت الصورة تغبيش داخل حدود اللوحة، والصورة فوقه
        top = top * ca + base * (1 - ca)
        al = m
    roi[:] = (top * al + roi * (1 - al)).astype(np.uint8)


def cmd_cover(a):
    W, H, fps, dur, has_audio = probe(a.video)
    ks = parse_keys(a.key)
    covers = [load_cover(c) for c in (a.cover or [])]
    cuts = sorted(float(c) for c in (a.cut or []))
    if a.auto_cuts:
        cuts = sorted(set(cuts + detect_cuts(a.video)))
        print("✂️ القطعات: " + (", ".join(f"{c:.2f}" for c in cuts) or "ما فيه"))
    t_from = a.t_from if a.t_from is not None else -1
    t_to = a.t_to if a.t_to is not None else 1e9

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.video, "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    enc_cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{W}x{H}", "-r", f"{fps:.6f}", "-i", "-"]
    if has_audio:
        enc_cmd += ["-i", a.video, "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
    enc_cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-shortest", a.out]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)

    tracker, seg, n, lost = None, -1, 0, 0
    size = W * H * 3
    while True:
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        t = n / fps
        sg = segment_of(t, cuts)
        sk = [k for k in ks if segment_of(k[0], cuts) == sg]   # المفاتيح ما تعبر القطع
        if t_from <= t <= t_to and sk:
            box = interp(sk, t)
            if a.track:
                # أعد البذر عند بداية كل مقطع بين مفتاحين (ومع كل قطعة)
                s = (sg, sum(1 for k in sk if k[0] <= t))
                if s != seg:
                    seg = s
                    tracker = cv2.TrackerMIL_create()
                    tracker.init(f, tuple(int(v) for v in box[:4]))
                else:
                    ok, tb = tracker.update(f)
                    # لو التتبع ضاع أو ابتعد وايد عن المسار، نرجع للمسار المحسوب
                    if ok and abs(tb[0] - box[0]) < box[2] * 1.5 and abs(tb[1] - box[1]) < box[3] * 2:
                        box = [tb[0], tb[1], box[2], box[3], box[4]]
                    else:
                        lost += 1
            # تصميم واحد لكل لقطة (من أول مفتاح فيها) — ما يتبدّل بنص اللقطة
            paint(f, box, covers, a.pad, a.stretch, pick_cover(covers, sk[0][3], sk[0][4]))
        enc.stdin.write(f.tobytes())
        n += 1
    enc.stdin.close()
    enc.wait()
    dec.wait()
    if enc.returncode != 0:
        sys.exit("⚠️ التجميع فشل")
    msg = f"✅ {a.out} — {n} فريم"
    if a.track and lost:
        msg += f" · التتبع رجع للمسار المحسوب {lost} مرة (عادي — بس افحص الورقة)"
    print(msg)
    print(f"   الحين افحص: python3 {sys.argv[0]} check {a.out}")


# ======================= التتبّع المستوي (planar) =======================

GOLD_BGR, BLACK_BGR = (49, 195, 242), (12, 11, 11)


def plate_design(aspect, logo_path=None, full_path=None, H=360):
    """تصميم بمقاس اللوحة بالضبط: أسود + إطار ذهبي + الشعار بالنص. العريضة (4:1+) تاخذ تصميم الكراج الكامل."""
    W = max(int(H * 0.8), int(round(H * aspect)))
    if full_path and aspect >= 4.0 and os.path.exists(full_path):
        im = cv2.imread(full_path)
        s = min(W / im.shape[1], H / im.shape[0])
        r = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        out = np.full((H, W, 3), BLACK_BGR, np.uint8)
        y, x = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
        out[y:y + r.shape[0], x:x + r.shape[1]] = r
        return out
    out = np.full((H, W, 3), BLACK_BGR, np.float32)
    t = max(6, int(H * 0.055))
    if logo_path and os.path.exists(logo_path):
        L = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if L.ndim == 2:
            L = cv2.cvtColor(L, cv2.COLOR_GRAY2BGR)
        if L.shape[2] == 3:
            L = np.dstack([L, np.full(L.shape[:2], 255, np.uint8)])
        s = min((W - 5 * t) / L.shape[1], (H - 5 * t) / L.shape[0])
        L = cv2.resize(L, (max(1, int(L.shape[1] * s)), max(1, int(L.shape[0] * s))),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        x, y = (W - L.shape[1]) // 2, (H - L.shape[0]) // 2
        al = L[..., 3:4] / 255
        out[y:y + L.shape[0], x:x + L.shape[1]] = L[..., :3] * al + out[y:y + L.shape[0], x:x + L.shape[1]] * (1 - al)
    out = out.astype(np.uint8)
    cv2.rectangle(out, (t // 2, t // 2), (W - 1 - t // 2, H - 1 - t // 2), GOLD_BGR, t, cv2.LINE_AA)
    return out


def quad_size(q):
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return w, h


def read_all_gray(path, W, H):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         stdout=subprocess.PIPE)
    fr, size = [], W * H
    while True:
        b = p.stdout.read(size)
        if len(b) < size:
            break
        fr.append(np.frombuffer(b, np.uint8).reshape(H, W))
    p.wait()
    return fr


def cuts_from_gray(fr, thr=25.0):
    cuts, prev = [], None
    for i, g in enumerate(fr):
        sm = cv2.resize(g, (108, 192)).astype(np.float32)
        if prev is not None and np.abs(sm - prev).mean() > thr:
            cuts.append(i)
        prev = sm
    return cuts


def _region(quad, shape, grow=1.2):
    c = quad.mean(0)
    q = c + (quad - c) * grow
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, np.round(q).astype(np.int32), 255)
    return m


def track_from(fr, ref, quad, lo, hi):
    """يتبع سطح اللوحة من فريم المرجع لقدّام ولورا داخل [lo, hi). يرجّع {فريم: الزوايا}."""
    out = {ref: quad.copy()}
    g0 = fr[ref]
    pw, ph = quad_size(quad)
    # قالب ECC: اللوحة نفسها + هامش صغير (سطح مستوي)
    tm = _region(quad, g0.shape, 1.15)
    x, y, w, h = cv2.boundingRect(tm)
    tmpl = g0[y:y + h, x:x + w].copy()  # (inputMask بـOpenCV = قناع الفريم الكامل، مو القالب — القالب مقصوص أصلاً)
    T, Ti = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64), np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]], np.float64)
    p0 = cv2.goodFeaturesToTrack(g0, 500, 0.005, 4, mask=_region(quad, g0.shape, 1.3), blockSize=5)
    p0 = np.zeros((0, 2), np.float32) if p0 is None else p0.reshape(-1, 2)
    lk = dict(winSize=(25, 25), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    stats = {"ecc": 0, "klt": 0}
    for step in (1, -1):
        pr, pc, Hc, prev = p0.copy(), p0.copy(), np.eye(3), g0
        i = ref + step
        while lo <= i < hi:
            g = fr[i]
            if len(pc) >= 4:
                nx, st1, _ = cv2.calcOpticalFlowPyrLK(prev, g, pc.reshape(-1, 1, 2).astype(np.float32), None, **lk)
                bk, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, nx, None, **lk)
                fb = np.linalg.norm(bk.reshape(-1, 2) - pc, axis=1)
                ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
                pr, pc = pr[ok], nx.reshape(-1, 2)[ok]
            Hk = None
            if len(pr) >= 8:
                Hk, inl = cv2.findHomography(pr, pc, cv2.RANSAC, 2.0)
                if Hk is not None:
                    inl = inl.ravel().astype(bool)
                    pr, pc = pr[inl], pc[inl]
            if Hk is None:
                Hk = Hc  # ما فيه نقاط كفاية — نبقى على آخر وضع
            # تدقيق ECC على سطح اللوحة (يمنع الانزلاق التراكمي)
            Hn = Hk
            try:
                W0 = (Hk @ T).astype(np.float32)
                W0 /= W0[2, 2]
                cc, W1 = cv2.findTransformECC(tmpl, g, W0, cv2.MOTION_HOMOGRAPHY,
                                              (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6), None, 5)
                He = W1.astype(np.float64) @ Ti
                qa = cv2.perspectiveTransform(quad.reshape(-1, 1, 2).astype(np.float64), Hk).reshape(-1, 2)
                qb = cv2.perspectiveTransform(quad.reshape(-1, 1, 2).astype(np.float64), He).reshape(-1, 2)
                if cc > 0.55 and np.abs(qa - qb).max() < max(6.0, 0.12 * pw):
                    Hn = He
                    stats["ecc"] += 1
                    if len(pr):
                        pc = cv2.perspectiveTransform(pr.reshape(-1, 1, 2).astype(np.float64), He).reshape(-1, 2).astype(np.float32)
                else:
                    stats["klt"] += 1
            except cv2.error:
                stats["klt"] += 1
            Hc = Hn
            q = cv2.perspectiveTransform(quad.reshape(-1, 1, 2).astype(np.float64), Hc).reshape(-1, 2)
            out[i] = q.astype(np.float32)
            if len(pr) < 80:  # نقاط جديدة على اللوحة، ونرجّعها لإحداثيات المرجع
                npnt = cv2.goodFeaturesToTrack(g, 400, 0.005, 4, mask=_region(q, g.shape, 1.3), blockSize=5)
                if npnt is not None:
                    npnt = npnt.reshape(-1, 2)
                    back = cv2.perspectiveTransform(npnt.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(Hc)).reshape(-1, 2)
                    pr = np.vstack([pr, back.astype(np.float32)])
                    pc = np.vstack([pc, npnt.astype(np.float32)])
            prev, i = g, i + step
    return out, stats


def smooth_quads(qs, idx):
    """تنعيم خفيف بالوقت (بلا تأخير) داخل اللقطة."""
    if len(idx) < 5:
        return qs
    A = np.stack([qs[i].reshape(-1) for i in idx])
    k = np.array([1, 4, 6, 4, 1], np.float64) / 16
    P = np.pad(A, ((2, 2), (0, 0)), mode="edge")
    S = np.stack([np.convolve(P[:, j], k, "valid") for j in range(A.shape[1])], 1)
    return {i: S[n].reshape(4, 2).astype(np.float32) for n, i in enumerate(idx)}


def paint_quad(frame, quad, design, pad=0.03, vel=(0.0, 0.0), shutter=0.5, soft=0.5):
    H, W = frame.shape[:2]
    c = quad.mean(0)
    q = (c + (quad - c) * (1 + 2 * pad)).astype(np.float32)
    pw, ph = quad_size(q)
    if pw < 3 or ph < 3:
        return
    # نصغّر التصميم لقريب مقاسه بالفيديو أول (بلا تشرشر)
    tw, th = max(8, int(pw * 1.5)), max(8, int(ph * 1.5))
    d = cv2.resize(design, (tw, th), interpolation=cv2.INTER_AREA)
    rgba = np.dstack([d, np.full((th, tw), 255, np.uint8)])
    x0, y0 = [int(v) - 3 for v in np.floor(q.min(0))]
    x1, y1 = [int(v) + 3 for v in np.ceil(q.max(0))]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]), q - np.float32([x0, y0]))
    wp = cv2.warpPerspective(rgba, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)).astype(np.float32)
    # غباش حركة بنفس سرعة الكاميرا — حتى يبين ملصوق مو طايف
    vx, vy = vel
    L = math.hypot(vx, vy) * shutter
    if L > 1.5:
        n = int(min(31, L)) | 1
        ker = np.zeros((n, n), np.float32)
        cv2.line(ker, (n // 2 - int(round(vx / math.hypot(vx, vy) * n / 2)), n // 2 - int(round(vy / math.hypot(vx, vy) * n / 2))),
                 (n // 2 + int(round(vx / math.hypot(vx, vy) * n / 2)), n // 2 + int(round(vy / math.hypot(vx, vy) * n / 2))), 1.0, 1)
        ker /= max(ker.sum(), 1e-6)
        pm = wp[..., :3] * (wp[..., 3:4] / 255)
        pm = cv2.filter2D(pm, -1, ker)
        al = cv2.filter2D(wp[..., 3], -1, ker)
        wp = np.dstack([pm / np.maximum(al[..., None] / 255, 1e-3), al])
    # غباش الفوكس: نفس غباش اللوحة الأصلية (الحافة الخارجية تنعم معاه بس الغطاء يبقى أكبر من اللوحة بالهامش)
    wp = cv2.GaussianBlur(wp, (0, 0), max(0.3, soft))
    a = np.clip(wp[..., 3:4] / 255 * (1.0 + 0.6 * min(soft, 4) / 4), 0, 1)
    roi = frame[y0:y1, x0:x1]
    roi[:] = np.clip(wp[..., :3] * a + roi * (1 - a), 0, 255).astype(np.uint8)


def parse_quads(qs, fps):
    """t,x1,y1,x2,y2,x3,y3,x4,y4[,نسبة] — النسبة (عرض÷ارتفاع اللوحة الحقيقي) لما الزاوية تغشّ الشكل."""
    out = []
    for q in qs:
        tok = q.split(",")
        if len(tok) not in (9, 10):
            sys.exit(f"⚠️ --quad لازم 9 قيم (أو 10 مع النسبة): t,x1,y1,x2,y2,x3,y3,x4,y4[,نسبة] — وصلني: {q}")
        f = int(tok[0][1:]) if tok[0].startswith("f") else int(round(float(tok[0]) * fps))
        ar = float(tok[9]) if len(tok) == 10 else None
        out.append((f, np.float32([float(v) for v in tok[1:9]]).reshape(4, 2), ar))
    return sorted(out, key=lambda x: x[0])


def _lap_energy(img, mask):
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    m = mask > 0
    return float(lap[m].var() / max(float(img[m].var()), 1e-3))


def band_ratio(img, m):
    """حدّة الحواف بغض النظر عن المحتوى: طاقة التفاصيل الدقيقة ÷ المتوسطة."""
    g1, g2, g3 = (cv2.GaussianBlur(img, (0, 0), s) for s in (0.8, 1.6, 3.2))
    return float((g1 - g2)[m].var() / max(float((g2 - g3)[m].var()), 1e-6))


def rel_blur(g_ref, q_ref, g_t, q_t):
    """نفس اللوحة بلقطتين: كم نغبّش الواضحة (بمقاس الثانية) لين تطابق الثانية؟ = غباش الفوكس بالبكسل.
    نعدّل بين الصورتين بـECC قبل المقارنة — فرق بكسل أو بكسلين بالزوايا كان يكبّر الغباش للضعف."""
    pw, ph = quad_size(q_t)
    Wc, Hc = max(24, int(round(pw))), max(10, int(round(ph)))
    mx, my = int(Wc * .08), int(Hc * .15)
    dst = np.float32([[mx, my], [Wc + mx, my], [Wc + mx, Hc + my], [mx, Hc + my]])
    S = (Wc + 2 * mx, Hc + 2 * my)
    A = cv2.warpPerspective(g_ref, cv2.getPerspectiveTransform(q_ref.astype(np.float32), dst), S, flags=cv2.INTER_AREA)
    D = cv2.warpPerspective(g_t, cv2.getPerspectiveTransform(q_t.astype(np.float32), dst), S, flags=cv2.INTER_AREA)
    sl = (slice(my + int(Hc * .1), my + int(Hc * .9)), slice(mx + int(Wc * .04), mx + int(Wc * .96)))
    nz = lambda x: (x[sl] - x[sl].mean()) / max(float(x[sl].std()), 1e-3)
    errs = []
    for sg in np.arange(0.0, 6.01, 0.25):
        Ab = cv2.GaussianBlur(A, (0, 0), sg) if sg > 0 else A.copy()
        Dw = D
        try:
            _, Wm = cv2.findTransformECC(Ab, D, np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6), None, 3)
            Dw = cv2.warpAffine(D, Wm, S, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except cv2.error:
            pass
        errs.append((float(sg), float(((nz(Ab) - nz(Dw)) ** 2).mean())))
    best = min(errs, key=lambda e: e[1])
    return best[0], best[1] / max(errs[0][1], 1e-6)   # (سيقما، كم تحسّن التطابق — أقل = أوثق)


def estimate_defocus(gray, quad, design):
    """كم غباش فوكس على اللوحة نفسها؟ نقيس حدّتها ونغبّش التصميم لين يوصل نفس الحدّة (سيقما بالبكسل)."""
    c = quad.mean(0)
    qi = (c + (quad - c) * 0.8).astype(np.float32)
    x0, y0 = [max(0, int(v) - 4) for v in np.floor(quad.min(0))]
    x1, y1 = [int(v) + 4 for v in np.ceil(quad.max(0))]
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    mask = np.zeros(roi.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(qi - [x0, y0]).astype(np.int32), 255)
    if mask.sum() < 255 * 200:
        return 0.5
    e_plate = _lap_energy(roi, mask)
    pw, ph = quad_size(quad)
    dg = cv2.cvtColor(cv2.resize(design, (max(8, int(pw * 1.5)), max(8, int(ph * 1.5))), interpolation=cv2.INTER_AREA),
                      cv2.COLOR_BGR2GRAY)
    th, tw = dg.shape
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]), (quad - [x0, y0]).astype(np.float32))
    wd = cv2.warpPerspective(dg, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR).astype(np.float32)
    m = mask > 0
    rp = band_ratio(roi, m)
    best, err = 0.5, 1e9
    for sg in np.arange(0.3, 8.01, 0.25):
        r = band_ratio(cv2.GaussianBlur(wd, (0, 0), sg), m)
        d = abs(math.log(max(r, 1e-6)) - math.log(max(rp, 1e-6)))
        if d < err:
            best, err = float(sg), d
    return best


def cmd_track(a):
    W, H, fps, dur, has_audio = probe(a.video)
    print("⏳ أقرا الفريمات…")
    fr = read_all_gray(a.video, W, H)
    N = len(fr)
    cuts = sorted(set(int(round(float(c) * fps)) for c in (a.cut or [])))
    if a.auto_cuts:
        cuts = sorted(set(cuts + cuts_from_gray(fr)))
    bounds = [0] + cuts + [N]
    seg_of = lambda f: max(k for k in range(len(bounds) - 1) if bounds[k] <= f)
    anchors = parse_quads(a.quad, fps)
    final, designs, report = {}, {}, []
    full = a.full if a.full and os.path.exists(a.full) else None
    custom = load_cover(a.cover)[:, :, :3] if a.cover else None
    softs, seg_info = {}, {}
    for sg in sorted(set(seg_of(f) for f, _, _ in anchors)):
        lo, hi = bounds[sg], bounds[sg + 1]
        an = [(f, q) for f, q, _ in anchors if seg_of(f) == sg]
        ars = [r for f, _, r in anchors if seg_of(f) == sg and r]
        tracks = []
        for f, q in an:
            tr, st = track_from(fr, f, q, lo, hi)
            tracks.append((f, tr))
            report.append(f"   لقطة {lo / fps:.2f}–{hi / fps:.2f}ث · مرسى f{f} · ECC {st['ecc']} / KLT {st['klt']}")
        seg = {}
        for i in range(lo, hi):
            left = [t for t in tracks if t[0] <= i]
            right = [t for t in tracks if t[0] >= i]
            if left and right and left[-1][0] != right[0][0]:
                (fa, ta), (fb, tb) = left[-1], right[0]
                w = (i - fa) / (fb - fa)
                seg[i] = (1 - w) * ta[i] + w * tb[i]
            else:
                seg[i] = (left[-1] if left else right[0])[1][i]
        idx = list(range(lo, hi))
        seg = smooth_quads(seg, idx)
        final.update(seg)
        pw, ph = quad_size(an[0][1])
        designs[sg] = custom if custom is not None else plate_design(ars[0] if ars else pw / ph, a.logo, full)
        seg_info[sg] = (an[0][0], an[0][1], ars[0] if ars else pw / ph)
    # غباش الفوكس لكل لقطة: نفس اللوحة (نفس النسبة) واضحة بلقطة ثانية؟ نقارن فيها. وإلا نقيس حدّة الحواف.
    sharp = {}
    for sg, (f, q, ar) in seg_info.items():
        c = q.mean(0)
        mk = np.zeros(fr[f].shape, np.uint8)
        cv2.fillConvexPoly(mk, np.round(c + (q - c) * 0.8).astype(np.int32), 255)
        sharp[sg] = band_ratio(fr[f].astype(np.float32), mk > 0)
    for sg, (f, q, ar) in seg_info.items():
        if a.soft is not None:
            softs[sg], how = a.soft, "يدوي"
        else:
            same = [o for o, (_, _, ar2) in seg_info.items() if o != sg and abs(math.log(ar2 / ar)) < 0.15]
            ref = max(same, key=lambda o: sharp[o]) if same else None
            done = False
            if ref is not None and sharp[ref] > sharp[sg] * 1.5:   # فرق حدّة واضح بس (الزاوية وحدها تغشّ المقارنة)
                fr_, q_, _ = seg_info[ref]
                sfit, gain = rel_blur(fr[fr_].astype(np.float32), q_, fr[f].astype(np.float32), q)
                if gain < 0.85:
                    softs[sg], how, done = math.hypot(sfit, 0.5), f"مقارنة بنفس اللوحة f{fr_}", True
            if not done:
                softs[sg], how = estimate_defocus(fr[f].astype(np.float32), q, designs[sg]), "حدّة الحواف"
        report.append(f"   غباش فوكس لقطة {bounds[sg] / fps:.2f}ث = {softs[sg]:.2f}px ({how})")
    print("\n".join(report))
    if a.save:
        json.dump({str(k): v.tolist() for k, v in final.items()}, open(a.save, "w"))
    # ---- الرسم ----
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.video, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE)
    def enc_for(path):
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
               "-r", f"{fps:.6f}", "-i", "-"]
        if has_audio:
            cmd += ["-i", a.video, "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
        cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-shortest", path]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)
    enc = enc_for(a.out)
    dbg = enc_for(a.debug) if a.debug else None
    size, n = W * H * 3, 0
    anchor_f = {f for f, _, _ in anchors}
    while True:
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        if dbg:
            g = f.copy()
            if n in final:
                cv2.polylines(g, [np.round(final[n]).astype(np.int32)], True, (0, 0, 255) if n in anchor_f else (0, 255, 0), 2, cv2.LINE_AA)
            dbg.stdin.write(g.tobytes())
        if n in final:
            sg = seg_of(n)
            vel = (0.0, 0.0)
            if n - 1 in final and seg_of(n - 1) == sg:
                vel = tuple((final[n].mean(0) - final[n - 1].mean(0)).tolist())
            paint_quad(f, final[n], designs[sg], a.pad, vel, a.shutter, softs[sg])
        enc.stdin.write(f.tobytes())
        n += 1
    for e in [enc] + ([dbg] if dbg else []):
        e.stdin.close(); e.wait()
    dec.wait()
    print(f"✅ {a.out} — {n} فريم · لوحات متتبَّعة بـ{len(final)} فريم" + (f" · فحص: {a.debug}" if dbg else ""))
    print(f"   افحص مقرّب: python3 {sys.argv[0]} grid {a.out} --frame <N> --crop x,y,w,h --step 10")


def cmd_check(a):
    W, H, fps, dur, _ = probe(a.video)
    times = np.arange(0, max(dur, 0.01), a.every)[:60]
    tiles = []
    tw = 360
    th = int(H * tw / W)
    for t in times:
        f = cv2.resize(read_frame(a.video, float(t)), (tw, th))
        cv2.putText(f, f"{t:.1f}s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(f)
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    out = a.out or os.path.splitext(a.video)[0] + "_platecheck.jpg"
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"✅ {out} — افتحها بعينك: ولا رقم لوحة مقروء بأي فريم؟")


def main():
    p = argparse.ArgumentParser(description="تغطية لوحات السيارات")
    sp = p.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("grid"); g.add_argument("video"); g.add_argument("sec", nargs="?", default="0"); g.add_argument("out", nargs="?")
    g.add_argument("--frame", type=int); g.add_argument("--crop"); g.add_argument("--step", type=int, default=10); g.add_argument("--zoom", type=int, default=4)
    tk = sp.add_parser("track"); tk.add_argument("video"); tk.add_argument("out")
    tk.add_argument("--quad", action="append", required=True)
    tk.add_argument("--cut", action="append"); tk.add_argument("--auto-cuts", action="store_true")
    here = os.path.dirname(os.path.abspath(__file__))
    tk.add_argument("--logo", default=os.path.join(here, "..", "brand", "logo-wide.png"))
    tk.add_argument("--full", default=os.path.join(here, "..", "brand", "plate-cover.png"))
    tk.add_argument("--cover", help="صورة تصميم جاهزة بدل التصميم اللي ينبني بمقاس اللوحة")
    tk.add_argument("--pad", type=float, default=0.03); tk.add_argument("--shutter", type=float, default=0.5)
    tk.add_argument("--soft", type=float, help="غباش ثابت للغطاء (px) بدل القياس التلقائي من اللوحة")
    tk.add_argument("--debug"); tk.add_argument("--save")
    c = sp.add_parser("cover"); c.add_argument("video"); c.add_argument("out")
    c.add_argument("--key", action="append", required=True)
    c.add_argument("--cover", action="append", help="تصميم اللوحة — كرّره (عريض وقصير) ويختار الأقرب لنسبة كل لوحة")
    c.add_argument("--track", action="store_true")
    c.add_argument("--cut", action="append", help="وقت قطع بين لقطتين (ثواني) — المفاتيح ما تنتقل عبره")
    c.add_argument("--auto-cuts", action="store_true", help="يكشف القطعات بنفسه")
    c.add_argument("--pad", type=float, default=0.12)
    c.add_argument("--stretch", action="store_true", help="مطّ التصميم على مقاس المربع بدل ما يحافظ على نسبته")
    c.add_argument("--from", dest="t_from", type=float); c.add_argument("--to", dest="t_to", type=float)
    k = sp.add_parser("check"); k.add_argument("video"); k.add_argument("--every", type=float, default=1.0); k.add_argument("out", nargs="?")
    a = p.parse_args()
    {"grid": cmd_grid, "cover": cmd_cover, "check": cmd_check, "track": cmd_track}[a.cmd](a)


if __name__ == "__main__":
    main()
