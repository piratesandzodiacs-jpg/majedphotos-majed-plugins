#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
45_plate_cover.py — تغطية لوحات السيارات (نسخة كراج موسى محسن TRD)

قاعدة الكراج: رقم لوحة سيارة الزبون ما يبين أبداً — يتغطّى بتصميم لوحة الكراج.

الأوامر:
  grid  <video> <sec> [out.jpg]
        فريم بوقته عليه شبكة كل 100px بأرقام — تقرا منه مكان اللوحة (x,y,w,h) بإحداثيات الفيديو.

  cover <video> <out.mp4> --key "t,x,y,w,h" [--key ...] [--cover plate.png] [--track] [--pad 0.12]
        يغطي اللوحة بكل فريم:
        - مفتاح واحد  = مربع ثابت (لقطة ثابتة).
        - عدة مفاتيح = المربع ينتقل بين المفاتيح بنعومة (خطي).
        - --track     = يتبع اللوحة بين المفاتيح (OpenCV MIL)، ويرجع لمكان المفتاح عند كل مفتاح.
        - --cover     = صورة لوحة الكراج (PNG). بدونها: تغبيش قوي + إطار، ما يُقرا منه شي.
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


def cmd_grid(a):
    img = read_frame(a.video, a.sec)
    h, w = img.shape[:2]
    for x in range(0, w, 100):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255) if x % 500 == 0 else (0, 180, 255), 1)
        cv2.putText(img, str(x), (x + 3, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for y in range(0, h, 100):
        cv2.line(img, (0, y), (w, y), (0, 255, 255) if y % 500 == 0 else (0, 180, 255), 1)
        cv2.putText(img, str(y), (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    out = a.out or os.path.splitext(a.video)[0] + f"_grid_{a.sec:g}.jpg"
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"✅ {out}  ({w}x{h}) — اقرا مكان اللوحة واكتبه: --key \"{a.sec:g},x,y,w,h\"")


def parse_keys(keys):
    ks = []
    for k in keys:
        p = [float(v) for v in k.split(",")]
        if len(p) != 5:
            sys.exit(f"⚠️ المفتاح لازم 5 أرقام t,x,y,w,h — وصلني: {k}")
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
            return [a[i] + (b[i] - a[i]) * u for i in range(1, 5)]


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


def paint(frame, box, cover, pad, stretch=False):
    H, W = frame.shape[:2]
    x, y, w, h = box
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


def cmd_cover(a):
    W, H, fps, dur, has_audio = probe(a.video)
    ks = parse_keys(a.key)
    cover = load_cover(a.cover)
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
        if t_from <= t <= t_to:
            box = interp(ks, t)
            if a.track:
                # أعد البذر عند بداية كل مقطع بين مفتاحين
                s = sum(1 for k in ks if k[0] <= t)
                if s != seg:
                    seg = s
                    tracker = cv2.TrackerMIL_create()
                    tracker.init(f, tuple(int(v) for v in box))
                else:
                    ok, tb = tracker.update(f)
                    # لو التتبع ضاع أو ابتعد وايد عن المسار، نرجع للمسار المحسوب
                    if ok and abs(tb[0] - box[0]) < box[2] * 1.5 and abs(tb[1] - box[1]) < box[3] * 2:
                        box = [tb[0], tb[1], box[2], box[3]]
                    else:
                        lost += 1
            paint(f, box, cover, a.pad, a.stretch)
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
    g = sp.add_parser("grid"); g.add_argument("video"); g.add_argument("sec", type=float); g.add_argument("out", nargs="?")
    c = sp.add_parser("cover"); c.add_argument("video"); c.add_argument("out")
    c.add_argument("--key", action="append", required=True)
    c.add_argument("--cover"); c.add_argument("--track", action="store_true")
    c.add_argument("--pad", type=float, default=0.12)
    c.add_argument("--stretch", action="store_true", help="مطّ التصميم على مقاس المربع بدل ما يحافظ على نسبته")
    c.add_argument("--from", dest="t_from", type=float); c.add_argument("--to", dest="t_to", type=float)
    k = sp.add_parser("check"); k.add_argument("video"); k.add_argument("--every", type=float, default=1.0); k.add_argument("out", nargs="?")
    a = p.parse_args()
    {"grid": cmd_grid, "cover": cmd_cover, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
