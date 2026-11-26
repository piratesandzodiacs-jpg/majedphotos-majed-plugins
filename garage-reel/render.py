"""Garage reel renderer: 1080x1920 @30fps, cut on the beats of bg-audio.mp3.

python3 render.py preview 1.0 4.2 ...   -> out/sheet.jpg (contact sheet of those times)
python3 render.py video                 -> out/reel-silent.mp4
"""
import json, math, os, subprocess, sys
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 30
DUR = 18.94
RED = (215, 25, 45); BLACK = (17, 17, 17); CREAM = (241, 238, 230); WHITE = (255, 255, 255)
ANTON = os.path.join(HERE, 'fonts/Anton-Regular.ttf')
MONO = '/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Medium.ttf'


def eo(x):  # ease-out cubic
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def eio(x):
    x = min(max(x, 0.0), 1.0)
    return 3 * x * x - 2 * x * x * x


def lerp(a, b, x):
    return a + (b - a) * x


@lru_cache(maxsize=64)
def font(path, size):
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=400)
def frame(clip, idx):
    d = os.path.join(HERE, 'frames', str(clip))
    n = len(os.listdir(d))
    idx = max(1, min(n, idx))
    return Image.open(os.path.join(d, '%04d.jpg' % idx)).convert('RGB')


def clip_img(clip, t_local, speed=1.0, zoom=1.0, dx=0.0):
    """Full-screen frame of a clip (1188x2112 source), zoom >= 1 crops in."""
    im = frame(clip, int(t_local * speed * FPS) + 1)
    sw, sh = im.size
    cw, ch = sw / zoom, sh / zoom
    cx = sw / 2 + dx * (sw - cw) / 2
    box = (cx - cw / 2, (sh - ch) / 2, cx + cw / 2, (sh + ch) / 2)
    return im.resize((W, H), Image.BILINEAR, box=box)


def crop_clip(clip, t_local, box_frac, size, speed=1.0):
    im = frame(clip, int(t_local * speed * FPS) + 1)
    sw, sh = im.size
    x0, y0, x1, y1 = box_frac
    return im.resize(size, Image.BILINEAR, box=(x0 * sw, y0 * sh, x1 * sw, y1 * sh))


# ---------------- text ----------------
@lru_cache(maxsize=200)
def word_img(text, size, color, track=0):
    f = font(ANTON, size)
    if track == 0:
        bb = f.getbbox(text)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        im = Image.new('RGBA', (w + 4, h + 4), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((2 - bb[0], 2 - bb[1]), text, font=f, fill=color)
        return im
    ws = [f.getbbox(c) for c in text]
    tot = sum(b[2] - b[0] if c != ' ' else size // 4 for c, b in zip(text, ws)) + track * (len(text) - 1)
    top = min(b[1] for b in ws); bot = max(b[3] for b in ws)
    im = Image.new('RGBA', (tot + 4, bot - top + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(im); x = 2
    for c, b in zip(text, ws):
        if c == ' ':
            x += size // 4 + track; continue
        d.text((x - b[0], 2 - top), c, font=f, fill=color)
        x += b[2] - b[0] + track
    return im


def boxed_word(canvas, text, x, y, size, t, t0, fg=BLACK, bg=WHITE, pad=(22, 14), dur=0.16, out=None):
    """Rectangle wipes in from the left, then the word slides up inside it."""
    if t < t0 or (out is not None and t >= out):
        return 0
    wi = word_img(text, size, fg)
    bw, bh = wi.width + pad[0] * 2, wi.height + pad[1] * 2
    p = eo((t - t0) / dur)
    if out is not None:  # wipe out to the right
        q = eo((t - (out - 0.12)) / 0.12) if t > out - 0.12 else 0
    else:
        q = 0
    x0 = x + int(bw * q); x1 = x + int(bw * p)
    if x1 > x0:
        ImageDraw.Draw(canvas).rectangle((x0, y, x1, y + bh), fill=bg)
        tp = eo((t - t0 - dur * 0.4) / dur)
        if tp > 0:
            layer = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
            layer.alpha_composite(wi, (pad[0], pad[1] + int((1 - tp) * bh * 0.6)))
            m = Image.new('L', (bw, bh), 0)
            ImageDraw.Draw(m).rectangle((x0 - x, 0, x1 - x, bh), fill=255)
            layer.putalpha(ImageChops.multiply(layer.getchannel('A'), m))
            canvas.alpha_composite(layer, (x, y))
    return bw


def pop_word(canvas, text, cx, cy, size, t, t0, color=WHITE, track=0, out=None, shadow=False):
    """Big word punches in (scale 1.18 -> 1) on the beat, centred at (cx, cy)."""
    if t < t0 or (out is not None and t >= out):
        return
    wi = word_img(text, size, color, track)
    p = eo((t - t0) / 0.18)
    s = lerp(1.18, 1.0, p)
    im = wi.resize((max(1, int(wi.width * s)), max(1, int(wi.height * s))), Image.BICUBIC)
    a = min(1.0, (t - t0) / 0.06)
    if out is not None and t > out - 0.1:
        a *= max(0.0, (out - t) / 0.1)
    if a < 1:
        im.putalpha(im.getchannel('A').point(lambda v: int(v * a)))
    if shadow:
        sh = Image.new('RGBA', im.size, (0, 0, 0, 0)); sh.putalpha(im.getchannel('A').point(lambda v: int(v * 0.55)))
        sh = sh.filter(ImageFilter.GaussianBlur(10))
        canvas.alpha_composite(sh, (int(cx - im.width / 2), int(cy - im.height / 2) + 6))
    canvas.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))


def mono(canvas, text, x, y, size=24, color=BLACK, t=None, t0=0.0):
    if t is not None and t < t0:
        return
    ImageDraw.Draw(canvas).text((x, y), text, font=font(MONO, size), fill=color)


def handles(canvas, box, color=WHITE, s=12, line=2, alpha=255):
    d = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    d.rectangle(box, outline=color + (alpha,), width=line)
    for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        d.rectangle((cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2), fill=color + (alpha,))


def grid_lines(canvas, color=(0, 0, 0, 28)):
    d = ImageDraw.Draw(canvas)
    for x in (90, 540, 990):
        d.line((x, 0, x, H), fill=color, width=2)
    for y in (300, 760, 1300):
        d.line((0, y, W, y), fill=color, width=2)


def flash(canvas, t, t0, dur=0.2, color=WHITE):
    if t0 <= t < t0 + dur:
        a = int(230 * (1 - (t - t0) / dur))
        canvas.alpha_composite(Image.new('RGBA', (W, H), color + (a,)))


# ---------------- scenes ----------------
def s_hook(t):
    z = lerp(1.14, 1.04, eo(t / 0.45)) if t < 0.45 else lerp(1.04, 1.09, (t - 0.45) / 2.9)
    c = clip_img(6459, t, speed=0.56, zoom=z).convert('RGBA')
    x = 90
    w1 = boxed_word(c, 'HOW', x, 330, 120, t, 0.10)
    boxed_word(c, 'IT', x + w1 + 14, 330, 120, t, 0.55)
    w3 = boxed_word(c, 'CAME', x, 520, 120, t, 1.25)
    boxed_word(c, 'IN..', x + w3 + 14, 520, 120, t, 1.96)
    boxed_word(c, 'HOW IT LEAVES', x, 750, 120, t, 2.66, fg=WHITE, bg=RED)
    mono(c, 'MOOSA MOHSIN TRD // BODY & PAINT', x, 262, 24, WHITE, t, 0.1)
    return c


CARDS = [  # clip, crop (fractions of the frame), label, beat
    (6458, (0.05, 0.42, 0.95, 0.80), '01  LIGHTS', 3.34),
    (6455, (0.00, 0.40, 0.80, 0.82), '02  BUMPER', 4.06),
    (6457, (0.15, 0.30, 0.95, 0.72), '03  HANDLES', 4.75),
]


def s_strip(t):
    c = Image.new('RGBA', (W, H), CREAM + (255,))
    grid_lines(c)
    lt = t - 3.34
    pop_word(c, 'STRIP', 540, 520, 330, t, 3.34, BLACK, out=4.06)
    pop_word(c, 'DOWN', 540, 520, 330, t, 4.06, BLACK, out=4.75)
    pop_word(c, 'PIECE BY', 540, 450, 210, t, 4.75, BLACK)
    pop_word(c, 'PIECE', 540, 660, 210, t, 4.85, RED)
    pos = [(60, 860), (385, 990), (640, 800)]
    cw, ch = 290, 390
    # connecting curve, drawn progressively
    d = ImageDraw.Draw(c)
    prog = min(1.0, lt / 1.9)
    pts = [(lerp(260, 800, i / 60), 1430 + 40 * math.sin(i / 60 * math.pi * 3)) for i in range(int(60 * prog) + 1)]
    if len(pts) > 1:
        d.line(pts, fill=(0, 0, 0, 120), width=2)
    for i, (clip, fr, label, b) in enumerate(CARDS):
        if t < b:
            continue
        p = eo((t - b) / 0.22)
        x, y = pos[i]
        y += int((1 - p) * 160)
        img = crop_clip(clip, t - b, fr, (cw, ch)).convert('RGBA')
        a = int(255 * min(1, (t - b) / 0.08))
        img.putalpha(a)
        c.alpha_composite(Image.new('RGBA', (cw + 20, ch + 20), BLACK + (a,)), (x - 10, y - 10))
        c.alpha_composite(img, (x, y))
        handles(c, (x - 10, y - 10, x + cw + 10, y + ch + 10), BLACK, alpha=a)
        mono(c, label, x, y - 46, 26, BLACK)
    return c


def s_prep(t):
    lt = t - 5.43
    c = clip_img(6456, lt, speed=0.62, zoom=lerp(1.02, 1.12, lt / 3.56)).convert('RGBA')
    handles(c, (120, 980, 940, 1460), WHITE, alpha=200)
    mono(c, 'SURFACE PREP // STAGE 02', 120, 930, 26, WHITE)
    for word, b, nxt, fg, bg in (('SAND', 5.43, 6.15, BLACK, WHITE), ('FILL', 6.15, 6.87, BLACK, WHITE),
                                 ('PREP', 6.87, 7.59, BLACK, WHITE), ('MASKED.', 7.59, 9.0, WHITE, RED)):
        boxed_word(c, word, 90, 380, 220, t, b, fg=fg, bg=bg, pad=(30, 18), out=nxt)
    # air-blow: halftone dots sweep the screen 8.29 -> 8.99
    if t >= 8.29:
        p = (t - 8.29) / 0.70
        d = ImageDraw.Draw(c); sp = 46
        for gy in range(0, H + sp, sp):
            for gx in range(0, W + sp, sp):
                k = (gx / W * 0.4 + (H - gy) / H * 0.6)
                r = max(0.0, min(1.0, (p * 1.6 - k * 0.6))) * sp * 0.78
                if r > 0.8:
                    d.ellipse((gx - r, gy - r, gx + r, gy + r), fill=RED)
    flash(c, t, 5.43)
    return c


def magnifier(c, t, t0, src_box, dst_xy, size=330):
    if t < t0:
        return
    p = eo((t - t0) / 0.2)
    s = int(size * lerp(0.6, 1.0, p))
    region = c.crop(src_box).resize((s, s), Image.BICUBIC)
    x, y = dst_xy[0] + (size - s) // 2, dst_xy[1] + (size - s) // 2
    ImageDraw.Draw(c).line((src_box[2], src_box[1], x, y + s), fill=(255, 255, 255, 160), width=2)
    handles(c, src_box, WHITE, s=8, alpha=200)
    c.alpha_composite(region.convert('RGBA'), (x, y))
    handles(c, (x, y, x + s, y + s), WHITE)
    mono(c, 'TEXTURE x2.0', x, y + s + 12, 22, WHITE)


def s_paint(t):
    if t < 10.41:
        lt = t - 8.99
        c = clip_img(6455, lt, zoom=lerp(1.04, 1.12, lt / 1.42)).convert('RGBA')
        tr = int(lerp(0, 28, lt / 1.42))
        pop_word(c, 'FRESH', 540, 420, 230, t, 8.99, WHITE, track=tr, shadow=True)
        pop_word(c, 'PAINT', 540, 660, 230, t, 9.69, RED, track=tr, shadow=True)
    elif t < 11.83:
        lt = t - 10.41
        c = clip_img(6457, lt, zoom=lerp(1.10, 1.03, lt / 1.42)).convert('RGBA')
        boxed_word(c, 'MATTE', 90, 360, 230, t, 10.41, pad=(30, 18))
        boxed_word(c, 'BLACK', 90, 650, 230, t, 11.10, fg=WHITE, bg=BLACK, pad=(30, 18))
    else:
        lt = t - 11.83
        c = clip_img(6454, lt, speed=0.85, zoom=lerp(1.0, 1.08, lt / 2.55)).convert('RGBA')
        magnifier(c, t, 12.51, (560, 1230, 700, 1370), (110, 860))
        magnifier(c, t, 13.19, (300, 1290, 420, 1410), (640, 740), size=300)
        boxed_word(c, 'FACTORY', 90, 300, 170, t, 11.83, pad=(26, 14))
        boxed_word(c, 'FINISH', 90, 520, 170, t, 12.51, fg=WHITE, bg=RED, pad=(26, 14))
    return c


def s_result(t):
    lt = t - 14.38
    c = clip_img(6458, lt, speed=0.7, zoom=lerp(1.12, 1.0, eo(lt / 2.7))).convert('RGBA')
    boxed_word(c, 'THE', 90, 340, 200, t, 15.07, pad=(28, 16))
    boxed_word(c, 'RESULT', 90, 600, 200, t, 15.68, fg=WHITE, bg=RED, pad=(28, 16))
    mono(c, 'MATTE BLACK // FULL BODY', 90, 290, 24, WHITE, t, 15.07)
    return c


@lru_cache(maxsize=1)
def sticker():
    im = Image.open(os.path.join(HERE, 'brand/garage-sticker.png')).convert('RGBA')
    w = 960
    return im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)


def s_end(t):
    lt = t - 17.08
    c = Image.new('RGBA', (W, H), BLACK + (255,))
    grid_lines(c, (255, 255, 255, 22))
    st = sticker()
    p = eo(lt / 0.28)
    s = lerp(1.35, 1.0, p)
    im = st.resize((int(st.width * s), int(st.height * s)), Image.BICUBIC)
    # moving shine across the sticker
    sp = (lt - 0.25) / 0.7
    if 0 < sp < 1:
        yy, xx = np.mgrid[0:im.height, 0:im.width].astype(np.float32)
        pos = (xx + yy * 0.5) / (im.width + im.height * 0.5)
        g = np.exp(-((pos - lerp(-0.1, 1.1, sp)) / 0.05) ** 2) * 0.75
        a = np.asarray(im.getchannel('A')).astype(np.float32) / 255
        sh = Image.fromarray(np.dstack([np.full_like(g, 255)] * 3 + [g * a * 255]).astype(np.uint8), 'RGBA')
        im = Image.alpha_composite(im, sh)
    a = min(1.0, lt / 0.08)
    if a < 1:
        im.putalpha(im.getchannel('A').point(lambda v: int(v * a)))
    c.alpha_composite(im, (540 - im.width // 2, 640 - im.height // 2))
    pop_word(c, 'BACK TO', 540, 1050, 170, t, 17.77, WHITE)
    pop_word(c, 'FACTORY NEW', 540, 1250, 170, t, 17.95, RED)
    flash(c, t, 17.08)
    return c


def hand_wipe(c, t, t_mid=3.34, color=BLACK):
    """Panel sweeps right->left, fully covering at t_mid (stand-in for the hand wipe)."""
    p = (t - (t_mid - 0.22)) / 0.44
    if not 0 <= p <= 1:
        return c
    x = int(lerp(W, -W, eio(p)))
    panel = Image.new('RGBA', (W + 160, H), color + (255,))
    grad = np.tile(np.linspace(0, 255, 160, dtype=np.float32), (H, 1))
    a = np.asarray(panel.getchannel('A')).copy(); a[:, :160] = grad.astype(np.uint8)
    a2 = a.copy(); a2[:, -160:] = grad[:, ::-1].astype(np.uint8)
    panel.putalpha(Image.fromarray(a2))
    c.alpha_composite(panel, (x - 80, 0))
    return c


def whip(t, t_mid=14.38, span=0.18):
    """Whip pan: the camera swings right; outgoing and incoming shots slide left side by side, motion-blurred."""
    p = eio((t - (t_mid - span)) / (2 * span))
    shift = int(p * W)
    out_im = s_paint(t_mid - span).convert('RGB')
    in_im = s_result(t_mid + span).convert('RGB')
    rgb = Image.new('RGB', (W, H))
    rgb.paste(out_im, (-shift, 0)); rgb.paste(in_im, (W - shift, 0))
    speed = math.sin(math.pi * min(max((t - (t_mid - span)) / (2 * span), 0), 1))
    k = int(140 * speed)
    if k > 2:
        arr = np.asarray(rgb).astype(np.float32); acc = np.zeros_like(arr); n = 11
        for i in range(n):
            acc += np.roll(arr, int((i - n // 2) * k / n), axis=1)
        rgb = Image.fromarray((acc / n).astype(np.uint8))
    return rgb.convert('RGBA')


def render(t):
    if t < 3.34:
        c = s_hook(t)
        c = hand_wipe(c, t)
    elif t < 5.43:
        c = s_strip(t)
        c = hand_wipe(c, t)
    elif t < 8.99:
        c = s_prep(t)
    elif t < 14.38 - 0.18:
        c = s_paint(t)
        if t < 9.3:  # dots shrink away, revealing the paint booth
            p = (t - 8.99) / 0.31
            d = ImageDraw.Draw(c); sp = 46
            for gy in range(0, H + sp, sp):
                for gx in range(0, W + sp, sp):
                    r = (1 - eo(p)) * sp * 0.78
                    if r > 0.8:
                        d.ellipse((gx - r, gy - r, gx + r, gy + r), fill=RED)
    elif t < 14.38 + 0.18:
        c = whip(t)
    elif t < 17.08:
        c = s_result(t)
    else:
        c = s_end(t)
    return c.convert('RGB')


def main():
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    mode = sys.argv[1]
    if mode == 'preview':
        ts = [float(x) for x in sys.argv[2:]]
        th = [render(t).resize((360, 640)) for t in ts]
        cols = 4; rows = math.ceil(len(th) / cols)
        sheet = Image.new('RGB', (cols * 360, rows * 640), 'gray')
        dr = ImageDraw.Draw(sheet)
        for i, (im, t) in enumerate(zip(th, ts)):
            sheet.paste(im, ((i % cols) * 360, (i // cols) * 640))
            dr.text(((i % cols) * 360 + 6, (i // cols) * 640 + 6), '%.2f' % t, fill='yellow', font=font(MONO, 22))
        sheet.save(os.path.join(HERE, 'out/sheet.jpg'), quality=85)
        return
    n = int(round(DUR * FPS))
    out = os.path.join(HERE, 'out/reel-silent.mp4')
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H),
                           '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
                           '-pix_fmt', 'yuv420p', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                           '-colorspace', 'bt709', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    for i in range(n):
        ff.stdin.write(render(i / FPS).tobytes())
        if i % 60 == 0:
            print('frame', i, '/', n, flush=True)
    ff.stdin.close(); ff.wait()
    print('done', out)


if __name__ == '__main__':
    main()
