"""Clean beat-cut reel, no on-screen text (car-showroom style).

Fast cuts in the opening, then one cut per beat; slow push-in on every shot;
ends on a black card with the logo and the account handle.

python3 clean.py --clips DIR --audio song.mp3 --logo logo.png --handle @name [--phone "+968 ..."]
                 [--length 13] [--end 4] [--out out/reel-clean.mp4]
"""
import argparse, glob, json, os, shutil, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS, SR = 1080, 1920, 30, 22050
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def duration(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)['format']['duration'])


# ---------------- beats ----------------
def beats(audio, start=0.0, length=30.0):
    """Beat times (s, relative to start): spectral-flux onsets + autocorrelation tempo + best-phase grid."""
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(start), '-t', str(length), '-i', audio,
                          '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    y = np.frombuffer(raw, np.float32)
    hop, n = 512, 2048
    frames = np.lib.stride_tricks.sliding_window_view(y, n)[::hop] * np.hanning(n)
    mag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(np.diff(mag, axis=0), 0).sum(1)
    flux = (flux - flux.mean()) / (flux.std() + 1e-9)
    fr = SR / hop
    ac = np.correlate(flux, flux, 'full')[len(flux) - 1:]
    lo, hi = int(fr * 60 / 180), int(fr * 60 / 70)  # 70..180 BPM
    period = lo + int(np.argmax(ac[lo:hi]))
    phase = int(np.argmax([flux[p::period].sum() for p in range(period)]))
    bt = np.arange(phase, len(flux), period) / fr
    return bt, 60 * fr / period


# ---------------- shot picking ----------------
def clip_scores(path, step=0.25):
    """Sharpness per sampled time (higher = crisper, less motion blur)."""
    d = duration(path)
    tmp = tempfile.mkdtemp()
    run(['ffmpeg', '-v', 'error', '-i', path, '-vf', 'fps=%g,scale=180:-2' % (1 / step), os.path.join(tmp, '%05d.png')])
    sc = []
    for f in sorted(glob.glob(os.path.join(tmp, '*.png'))):
        g = np.asarray(Image.open(f).convert('L'), np.float32)
        lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        bright = g.mean()
        sc.append(lap.var() * (0.3 if bright < 25 else 1.0))  # dark frames lose
    shutil.rmtree(tmp)
    return d, np.array(sc), step


def plan(clips, cut_lengths):
    """Assign each cut a (clip, in-point): round-robin over clips, sharpest unused window each time."""
    info = {c: clip_scores(c) for c in clips}
    used = {c: [] for c in clips}
    shots = []
    for i, L in enumerate(cut_lengths):
        c = clips[i % len(clips)]
        d, sc, step = info[c]
        best, bt = -1, 0.0
        for k in range(len(sc)):
            t = k * step
            if t + L > d - 0.05:
                break
            gap = min((abs(t - u) for u in used[c]), default=99.0)
            # sharpness, but a window close to one already used is penalised (fully blocked when overlapping)
            win = sc[k:k + max(1, int(L / step))].mean() * min(1.0, gap / max(L, 0.8)) ** 2
            if win > best:
                best, bt = win, t
        used[c].append(bt)
        shots.append((c, bt, L))
    return shots


# ---------------- render ----------------
def render_shot(src, t0, L, out, idx):
    n = max(2, int(round(L * FPS)))
    z0, z1 = (1.00, 1.06) if idx % 2 == 0 else (1.07, 1.01)  # alternate push-in / pull-out
    vf = ('scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,'
          "zoompan=z='%f+(%f)*on/%d':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=%dx%d:fps=%d,"
          'eq=contrast=1.06:saturation=1.08,setsar=1') % (W, H, W, H, z0, z1 - z0, n, W, H, FPS)
    run(['ffmpeg', '-v', 'error', '-y', '-ss', '%.3f' % t0, '-i', src, '-frames:v', str(n), '-vf', vf, '-an',
         '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '14', '-pix_fmt', 'yuv420p', '-r', str(FPS), out])


def end_card(logo, handle, phone, secs, out, tmp):
    frames_dir = os.path.join(tmp, 'end'); os.makedirs(frames_dir)
    lg = Image.open(logo).convert('RGBA') if logo else None
    if lg:
        w = 520; lg = lg.resize((w, int(lg.height * w / lg.width)), Image.LANCZOS)
    f1 = ImageFont.truetype(FONT, 46); f2 = ImageFont.truetype(FONT, 34)
    n = int(secs * FPS)
    for i in range(n):
        t = i / FPS
        c = Image.new('RGBA', (W, H), (0, 0, 0, 255))
        p = min(1.0, t / 0.35); e = 1 - (1 - p) ** 3
        if lg:
            s = 1.12 - 0.12 * e
            im = lg.resize((int(lg.width * s), int(lg.height * s)), Image.BICUBIC)
            im.putalpha(im.getchannel('A').point(lambda v: int(v * e)))
            c.alpha_composite(im, ((W - im.width) // 2, 820 - im.height // 2))
        q = max(0.0, min(1.0, (t - 0.35) / 0.3))
        d = ImageDraw.Draw(c)
        for txt, f, y in ((handle, f1, 1130), (phone, f2, 1200)):
            if txt and q > 0:
                bw = d.textlength(txt, font=f)
                d.text(((W - bw) / 2, y + (1 - q) * 20), txt, font=f, fill=(255, 255, 255, int(255 * q)))
        c.convert('RGB').save(os.path.join(frames_dir, '%04d.png' % (i + 1)))
    run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(FPS), '-i', os.path.join(frames_dir, '%04d.png'),
         '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '14', '-pix_fmt', 'yuv420p', out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', required=True); ap.add_argument('--audio', required=True)
    ap.add_argument('--audio-start', type=float, default=0.0)
    ap.add_argument('--logo'); ap.add_argument('--handle', default=''); ap.add_argument('--phone', default='')
    ap.add_argument('--length', type=float, default=13.0, help='seconds of footage before the end card')
    ap.add_argument('--end', type=float, default=4.0)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out/reel-clean.mp4'))
    a = ap.parse_args()

    clips = sorted(p for p in glob.glob(os.path.join(a.clips, '*')) if p.lower().endswith(('.mp4', '.mov', '.m4v')))
    assert clips, 'no clips found'
    bt, bpm = beats(a.audio, a.audio_start, a.length + a.end + 2)
    print('tempo %.1f BPM, %d beats' % (bpm, len(bt)))

    # cut points: opening burst on half/quarter beats, then one per beat, end card lands on a beat
    beat = 60 / bpm
    cuts = [0.0]
    first = bt[bt > 0.05][0] if (bt > 0.05).any() else beat
    burst_end = first + 3 * beat
    t = 0.0
    while t + beat / 4 < first: t += beat / 4; cuts.append(t)
    t = first; cuts.append(t)
    while t + beat / 2 < burst_end - 1e-3: t += beat / 2; cuts.append(round(t, 3))
    for b in bt:
        if b > burst_end - 1e-3 and b < a.length: cuts.append(float(b))
    end_at = float(bt[np.argmin(np.abs(bt - a.length))])
    cuts = sorted(set(round(c, 3) for c in cuts if c < end_at - 0.15)) + [end_at]
    lengths = np.diff(cuts)
    print('%d shots, end card at %.2fs' % (len(lengths), end_at))

    shots = plan(clips, lengths)
    tmp = tempfile.mkdtemp()
    parts = []
    for i, (src, t0, L) in enumerate(shots):
        p = os.path.join(tmp, '%03d.mp4' % i); render_shot(src, t0, L, p, i); parts.append(p)
        print('  %5.2fs  %-28s in %.2f  len %.2f' % (cuts[i], os.path.basename(src), t0, L))
    ec = os.path.join(tmp, 'end.mp4'); end_card(a.logo, a.handle, a.phone, a.end, ec, tmp); parts.append(ec)
    with open(os.path.join(tmp, 'list.txt'), 'w') as f:
        f.writelines("file '%s'\n" % p for p in parts)
    total = end_at + a.end
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', os.path.join(tmp, 'list.txt'),
         '-ss', str(a.audio_start), '-i', a.audio, '-map', '0:v', '-map', '1:a', '-t', '%.3f' % total,
         '-af', 'afade=t=out:st=%.3f:d=1.2' % (total - 1.2), '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
         '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', a.out])
    shutil.rmtree(tmp)
    print('done', a.out, '%.2fs' % total)


if __name__ == '__main__':
    main()
