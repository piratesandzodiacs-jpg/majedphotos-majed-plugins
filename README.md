# Moosa Mohsin TRD Garage — Claude plugins

Claude plugins customized for **Moosa Mohsin TRD Garage** (Al Maabilah Industrial, Muscat, Oman).

## `majed-video/` — Arabic video editing, garage edition

Based on [majed-video](https://github.com/majedphotos/video-ad-editor) v6.8.1 by Majed Alzaabi (MIT). What the garage edition adds:

| | |
|---|---|
| **Garage identity** | `video-ad-editor/references/garage-brand.md`: name, services, contact, Omani tone, OMR pricing rules. Every skill reads it first, so Claude never asks "what are your colors?" and never opens the setup page |
| **End card** | WhatsApp 98880043 · `@repair_painting_cars` · the garage's services as chips. It replaces the plugin's own ad, which otherwise shows at the end of every video |
| **Licence plates** | New `scripts/45_plate_cover.py` (`grid` / `cover` / `check`) covers plates with the garage's plate design, either fixed, keyframed or tracked. No video is delivered while a plate is readable |
| **Garage video types** | Before/after montage, chassis-straightening explainers, work reels with voiceover, and ready-made hook ideas |
| **Updates** | Claude no longer offers to auto-update from upstream, because that would wipe the garage edition. New upstream versions get merged here instead |

### Still needed (ask once, then saved on the device)
- ~~Garage logo~~ ✅ bundled in `video-ad-editor/brand/` (square `logo.png` for the engine, `logo-wide.png` for covers); colors extracted from it
- Plate-cover design (PNG) → `~/.video-ad-editor/plate-cover.png`
- Confirm the colors taken from the logo (black `#0B0B0C`, crimson `#A8261F`, white `#F5F5F5`)

### Install
Zip the `majed-video/` folder and upload it in claude.ai → Settings → Plugins (replacing the current *majed-video* upload). Or add this repo as a marketplace in Claude Code.
