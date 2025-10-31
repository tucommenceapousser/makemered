#!/usr/bin/env python3
"""
genredeyess_ai.py
Ultra-complet: TUI/CLI image variant generator (Pillow + NumPy) + intégration OpenAI (GPT-4 suggestions & DALL·E)
Usage:
  python3 genredeyess_ai.py input.png output.png [--glow 20 --glitch --threshold 60 --font /path/to.ttf]
  python3 genredeyess_ai.py                -> interactive TUI
IA:
  export OPENAI_API_KEY="..."  # required for AI features
  Add --ai to use AI prompt generation or --ai-generate to call DALL·E (optionnel)
"""

import os
import sys
import argparse
from collections import deque
import math
import base64
from io import BytesIO

from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageFont, ImageOps
import numpy as np

# optional niceties
try:
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt
    console = Console()
except Exception:
    console = None

# optional OpenAI (may not be installed / may be older/newer)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
openai_client = None
try:
    # modern official client pattern (if available)
    from openai import OpenAI
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    try:
        # fallback to legacy openai package object (older usage)
        import openai
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            openai_client = openai
    except Exception:
        openai_client = None

# -------------------------
# Small helpers for output
# -------------------------
def info(msg):
    if console:
        console.print(f"[bold cyan]i[/] {msg}")
    else:
        print("[i] " + msg)

def warn(msg):
    if console:
        console.print(f"[bold yellow]![/] {msg}")
    else:
        print("[!] " + msg)

def error(msg):
    if console:
        console.print(f"[bold red]ERROR[/] {msg}")
    else:
        print("ERROR: " + msg)

# -------------------------
# Image utility functions
# -------------------------
def load_image(path):
    img = Image.open(path).convert("RGBA")
    return img

def save_image(img, path):
    img.save(path)
    info(f"Saved: {path}")

def pil_to_gray_np(img):
    return np.array(img.convert("L"))

def resize_for_analysis(np_gray, max_dim=700):
    h,w = np_gray.shape
    scale = 1.0
    if max(h,w) > max_dim:
        scale = max_dim / max(h,w)
        nh = int(h*scale); nw = int(w*scale)
        small = Image.fromarray(np_gray).resize((nw,nh), Image.LANCZOS)
        return np.array(small), scale
    return np_gray, scale

def binary_mask_from_gray(np_gray, threshold=60):
    return (np_gray < threshold).astype(np.uint8)

def connected_components(mask):
    h,w = mask.shape
    labels = np.zeros_like(mask, dtype=np.int32)
    label = 0
    comps = {}
    for y in range(h):
        for x in range(w):
            if mask[y,x] and labels[y,x] == 0:
                label += 1
                q = deque()
                q.append((x,y))
                labels[y,x] = label
                minx,miny,maxx,maxy = x,y,x,y
                area = 0
                while q:
                    cx,cy = q.popleft()
                    area += 1
                    for nx,ny in ((cx-1,cy),(cx+1,cy),(cx,cy-1),(cx,cy+1)):
                        if 0 <= nx < w and 0 <= ny < h:
                            if mask[ny,nx] and labels[ny,nx] == 0:
                                labels[ny,nx] = label
                                q.append((nx,ny))
                                if nx < minx: minx = nx
                                if ny < miny: miny = ny
                                if nx > maxx: maxx = nx
                                if ny > maxy: maxy = ny
                comps[label] = {"area": area, "bbox": (minx,miny,maxx,maxy)}
    return comps, labels

def pick_eye_regions(comps, scale, orig_size, prefer_top_fraction=0.6, max_regions=2, min_area_ratio=0.0005):
    # orig_size passed as (H_orig, W_orig)
    H_orig, W_orig = orig_size
    items = sorted(comps.items(), key=lambda kv: kv[1]["area"], reverse=True)
    selected = []
    for lid, info in items:
        minx,miny,maxx,maxy = info["bbox"]
        # map back to original coords
        if scale != 1.0 and scale != 0:
            rx1 = int(minx / scale); ry1 = int(miny / scale)
            rx2 = int(maxx / scale); ry2 = int(maxy / scale)
        else:
            rx1,ry1,rx2,ry2 = minx,miny,maxx,maxy
        area = info["area"]
        # filter tiny components (relative to image)
        if area < (W_orig * H_orig * min_area_ratio):
            continue
        # prefer top area
        if ry1 < H_orig * prefer_top_fraction:
            selected.append((rx1,ry1,rx2,ry2, area))
        else:
            # keep as backup
            selected.append((rx1,ry1,rx2,ry2, area))
        if len(selected) >= max_regions:
            break
    return selected

# -------------------------
# Effects
# -------------------------
def make_elliptical_mask(size, bbox, margin=0.12):
    W,H = size
    x1,y1,x2,y2 = bbox
    w = max(2, x2 - x1); h = max(2, y2 - y1)
    padw = int(w * margin); padh = int(h * margin)
    xs = max(0, x1 - padw); ys = max(0, y1 - padh)
    xe = min(W, x2 + padw); ye = min(H, y2 + padh)
    mask = Image.new("L", (W,H), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([xs,ys,xe,ye], fill=255)
    return mask, (xs,ys,xe,ye)

def add_red_glow(base_img, masks, red=(255,30,30), glow_radius=18, glow_strength=0.85):
    base = base_img.convert("RGBA")
    W,H = base.size
    glow_layer = Image.new("RGBA", (W,H), (0,0,0,0))
    for m in masks:
        red_img = Image.new("RGBA", (W,H), (*red,0))
        red_img.putalpha(m)
        blurred = red_img.filter(ImageFilter.GaussianBlur(radius=glow_radius))
        glow_layer = Image.alpha_composite(glow_layer, blurred)
    # use screen to produce neon effect, then composite base to keep details
    combined = ImageChops.screen(base, glow_layer)
    result = Image.alpha_composite(combined.convert("RGBA"), base)
    if glow_strength > 0.5:
        result = Image.blend(combined.convert("RGBA"), result, 0.6)
    return result

def fill_eye_centers(img, masks, color=(255,30,30)):
    base = img.convert("RGBA")
    W,H = base.size
    overlay = Image.new("RGBA", (W,H), (0,0,0,0))
    for m in masks:
        solid = Image.new("RGBA", (W,H), (*color,255))
        solid.putalpha(m)
        overlay = Image.alpha_composite(overlay, solid)
    out = Image.alpha_composite(base, overlay)
    return out

def add_neon_text(img, text="TRHACKNON", font_path=None, size_ratio=0.12, glow_layers=[(30,20),(18,60),(6,180)], color=(255,180,200)):
    W,H = img.size
    fontsize = max(12, int(W * size_ratio))
    try:
        font = ImageFont.truetype(font_path if font_path else "DejaVuSans-Bold.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()
    txt_layer = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(txt_layer)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (W - text_w)//2
    y = int(H * 0.78) - text_h//2
    for blur,alpha in glow_layers:
        layer = Image.new("RGBA", img.size, (0,0,0,0))
        d = ImageDraw.Draw(layer)
        d.text((x,y), text, font=font, fill=(color[0], color[1], color[2], alpha))
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur))
        txt_layer = Image.alpha_composite(txt_layer, layer)
    d = ImageDraw.Draw(txt_layer)
    d.text((x,y), text, font=font, fill=(255,255,255,255))
    out = Image.alpha_composite(img.convert("RGBA"), txt_layer)
    return out

def add_scanlines(img, spacing=3, opacity=20):
    W,H = img.size
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    for yy in range(0,H,spacing):
        draw.line([(0,yy),(W,yy)], fill=(0,0,0,opacity))
    return Image.alpha_composite(img.convert("RGBA"), overlay)

def add_noise(img, intensity=6):
    np_img = np.array(img.convert("RGB")).astype(np.int16)
    noise = (np.random.randn(*np_img.shape) * intensity).astype(np.int16)
    np_img = np_img + noise
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)

def chromatic_aberration(img, shift=2):
    np_img = np.array(img.convert("RGB"))
    b = np_img[:,:,0]
    g = np_img[:,:,1]
    r = np_img[:,:,2]
    h,w = b.shape
    def shift_channel(ch, dx):
        imgch = Image.fromarray(ch)
        return np.array(imgch.transform((w,h), Image.AFFINE, (1,0,dx,0,1,0)))
    try:
        b2 = shift_channel(b, shift)
        r2 = shift_channel(r, -shift)
        merged = np.stack([b2,g,r2], axis=2)
        return Image.fromarray(np.clip(merged,0,255).astype(np.uint8))
    except Exception:
        return img

def apply_glitch(img, bands=10, max_shift=40):
    W,H = img.size
    out = img.copy()
    for i in range(bands):
        y = int((H / bands) * i)
        h = int(H / bands)
        shift = int(np.random.randint(-max_shift, max_shift))
        box = (0, y, W, y+h)
        region = out.crop(box)
        out.paste(region, (shift, y))
    out = chromatic_aberration(out, shift=4)
    return out

# -------------------------
# AI helpers (text & DALL·E suggestions)
# -------------------------

from PIL import Image
import os

def ensure_png(input_path):
    """Force le format PNG si l'image n'est pas déjà en vrai PNG."""
    try:
        img = Image.open(input_path)
        # Si ce n'est pas déjà un vrai PNG, on reconvertit
        if img.format != "PNG":
            png_path = os.path.splitext(input_path)[0] + "_true.png"
            img.convert("RGBA").save(png_path, "PNG")
            return png_path
        return input_path
    except Exception as e:
        warn(f"Impossible de convertir en PNG: {e}")
        return input_path
        
def ai_suggest_settings(image_path, candidates):
    """
    Ask GPT-4 (chat) to suggest which candidate bbox looks like eyes, and return manual recommended bbox if provided.
    candidates: list of (x1,y1,x2,y2,area)
    returns: possibly modified list of candidate boxes
    """
    if not openai_client:
        warn("OpenAI client non disponible — impossible d'utiliser ai_suggest_settings().")
        return candidates
    try:
        # build a short prompt describing candidates and ask GPT to pick 2 best
        msg = "Tu es un assistant d'analyse d'image. J'ai détecté ces boîtes candidates (x1,y1,x2,y2,area) dans une image. Choisis les 2 boîtes qui ressemblent le plus à des yeux humains/masque et renvoie la liste JSON des boîtes choisies.\nCandidates:\n"
        for c in candidates:
            msg += f"- {c}\n"
        msg += "\nRéponds uniquement par un JSON array de boîtes, exemple: [[x1,y1,x2,y2],[...]]\n"

        # use chat completions style if available
        if hasattr(openai_client, "chat") or hasattr(openai_client, "ChatCompletion"):
            # try common interfaces
            try:
                # modern client
                resp = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role":"user","content":msg}],
                    max_tokens=300,
                    temperature=0.2
                )
                content = resp.choices[0].message["content"]
            except Exception:
                # legacy openai
                resp = openai_client.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role":"user","content":msg}],
                    max_tokens=300,
                    temperature=0.2
                )
                content = resp.choices[0].message["content"]
        else:
            warn("Le client OpenAI installé ne propose pas l'interface attendue. Retour sans IA.")
            return candidates

        # parse JSON from content
        import json
        parsed = None
        try:
            parsed = json.loads(content.strip())
        except Exception:
            # try to extract JSON substring
            import re
            m = re.search(r"(.*)", content, re.S)
            if m:
                parsed = json.loads(m.group(1))
        if parsed and isinstance(parsed, list):
            new = []
            for box in parsed[:2]:
                if isinstance(box, (list,tuple)) and len(box) == 4:
                    new.append((int(box[0]),int(box[1]),int(box[2]),int(box[3]), 1))
            if new:
                return new
    except Exception as e:
        import traceback
        warn(f"Génération IA échouée: {e}")
        traceback.print_exc()
    return candidates

def ai_generate_with_dalle(prompt, output_image_path, size="1024x1024"):
    import openai, base64, traceback

    if not os.environ.get("OPENAI_API_KEY"):
        warn("OPENAI_API_KEY non défini.")
        return False

    info("Lancement génération IA (DALL·E / image create)...")

    try:
        resp = openai.Image.create(
            model="gpt-image-1",
            prompt=prompt,
            size="auto"   # ✅ ici on met "1024x1024" ou "auto"
        )
        img_bytes = base64.b64decode(resp['data'][0]['b64_json'])
        with open(output_image_path, "wb") as out_file:
            out_file.write(img_bytes)
        info(f"Image IA générée -> {output_image_path}")
        return True
    except Exception as e:
        warn(f"Génération IA échouée: {e}")
        traceback.print_exc()
        return False
# -------------------------
# Pipeline principal
# -------------------------
def process_image(input_path, output_path,
                  threshold=60, glow_radius=18, glow_strength=0.85,
                  add_text=True, font_path=None, scanlines=True, scan_spacing=3,
                  noise_level=6, glitch=False, glitch_bands=12, force_eyes=False, ai_suggest=False):
    info(f"Loading image: {input_path}")
    img = load_image(input_path)
    W,H = img.size
    gray = pil_to_gray_np(img)
    small_gray, scale = resize_for_analysis(gray, max_dim=700)
    info(f"Detection scale: {scale:.3f}")

    mask_small = binary_mask_from_gray(small_gray, threshold=threshold)
    comps, labels = connected_components(mask_small)
    info(f"{len(comps)} composants détectés (analyse).")

    selected = []
    if force_eyes:
        # fixed eyes suitable for typical Guy Fawkes mask composition
        cx, cy = W // 2, H // 2 - int(H * 0.09)
        eye_w = max(16, W // 10)
        eye_h = max(8, H // 20)
        selected = [
            (cx - eye_w - int(W*0.03), cy - eye_h, cx - int(W*0.03), cy + eye_h, 1),
            (cx + int(W*0.03), cy - eye_h, cx + eye_w + int(W*0.03), cy + eye_h, 1)
        ]
    else:
        selected = pick_eye_regions(comps, scale, (H,W), prefer_top_fraction=0.7, max_regions=4, min_area_ratio=0.0008)
        if ai_suggest and openai_client and selected:
            info("Envoi des candidats à GPT-4 pour choix...")
            # give GPT up to first 6 candidates (map to original coords)
            chosen = ai_suggest_settings(openai_client,selected[:6])
            if chosen:
                selected = chosen
        # reduce to 2 best (largest area)
        if selected:
            selected = sorted(selected, key=lambda x: x[4], reverse=True)[:2]

    if not selected:
        warn("Aucun composant pertinent trouvé — utilisation d'un masque par défaut centré.")
        cx, cy = W//2, H//2 - H//8
        w_eye = W//8
        h_eye = H//16
        selected = [(cx - w_eye - 10, cy - h_eye, cx - 10, cy + h_eye, 1),
                    (cx + 10, cy - h_eye, cx + w_eye + 10, cy + h_eye, 1)]

    info(f"Selected regions (x1,y1,x2,y2,area): {selected}")

    # create masks
    masks = []
    for (x1,y1,x2,y2,a) in selected:
        # auto margin: smaller for large boxes
        box_area = max(1, (x2-x1)*(y2-y1))
        image_area = max(1, W*H)
        margin = 0.12 if box_area > image_area * 0.02 else 0.22
        m, bbox = make_elliptical_mask((W,H), (x1,y1,x2,y2), margin=margin)
        # soften
        m = m.filter(ImageFilter.GaussianBlur(radius=2))
        masks.append(m)

    # add red glow and fill center
    info("Applying red glow...")
    img_glow = add_red_glow(img, masks, red=(255,30,30), glow_radius=glow_radius, glow_strength=glow_strength)
    img_red = fill_eye_centers(img_glow, masks, color=(255,60,60))

    # subtle gradient tint overlay cyan->purple
    overlay = Image.new("RGBA", (W,H), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    for i in range(H):
        t = i / max(1, H)
        r = int(10*(1-t) + 200*t)
        g = int(160*(1-t) + 30*t)
        b = int(220*(1-t) + 200*t)
        draw.line([(0,i),(W,i)], fill=(r,g,b, int(8*(1-t)+6)))
    img_tinted = Image.alpha_composite(img_red, overlay)

    # darken background lightly
    enhancer = Image.new("RGBA", (W,H), (0,0,0,90))
    img_dark = Image.alpha_composite(img_tinted, enhancer)

    # chromatic aberration, noise, scanlines, glitch
    img_ca = chromatic_aberration(img_dark, shift=2)
    img_noisy = add_noise(img_ca, intensity=noise_level) if noise_level>0 else img_ca
    img_scan = add_scanlines(img_noisy, spacing=scan_spacing, opacity=18) if scanlines else img_noisy

    if glitch:
        info("Applying glitch effect...")
        img_final = apply_glitch(img_scan, bands=glitch_bands, max_shift=max(24, W//40))
    else:
        img_final = img_scan

    if add_text:
        info("Adding neon text TRHACKNON...")
        img_final = add_neon_text(img_final, text="TRHACKNON", font_path=font_path, size_ratio=0.12)

    img_final = img_final.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    save_image(img_final, output_path)
    return output_path

# -------------------------
# CLI & TUI
# -------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="TRHACKNON image variant generator (Pillow + NumPy) + OpenAI")
    p.add_argument("input", nargs="?", help="input image path (if omitted, interactive TUI starts)")
    p.add_argument("output", nargs="?", help="output image path (optional; defaults to input.variant.png)")
    p.add_argument("--glow-radius", type=int, default=18, help="radius for red glow (default: 18)")
    p.add_argument("--glow-strength", type=float, default=0.85, help="glow blend strength 0.0-1.0 (default: 0.85)")
    p.add_argument("--threshold", type=int, default=60, help="grayscale threshold for mask detection (default: 60)")
    p.add_argument("--font", type=str, default=None, help="path to TTF font to use for neon text")
    p.add_argument("--no-text", action="store_true", help="do not add neon text")
    p.add_argument("--no-scanlines", action="store_true", help="do not add scanlines")
    p.add_argument("--scan-spacing", type=int, default=3, help="scanline spacing (default: 3)")
    p.add_argument("--noise", type=int, default=6, help="noise intensity (default: 6; 0 to disable)")
    p.add_argument("--glitch", action="store_true", help="apply glitch effect")
    p.add_argument("--glitch-bands", type=int, default=12, help="glitch bands (default: 12)")
    p.add_argument("--force-eyes", action="store_true", help="force-eye placement (useful for masks/faces)")
    p.add_argument("--ai-suggest", action="store_true", help="ask GPT-4 to pick best candidate eye boxes (requires OPENAI_API_KEY)")
    p.add_argument("--ai-generate", action="store_true", help="call DALL·E / image edit to generate a variant (requires OPENAI_API_KEY)")
    p.add_argument("--ai-prompt", type=str, default=None, help="prompt to use with --ai-generate (optional; a default is generated if omitted)")
    p.add_argument("--quiet", action="store_true", help="suppress informational prints (if rich not available)")
    return p

def interactive_prompt(prompt_text, default=None):
    if console:
        try:
            return Prompt.ask(prompt_text, default=default)  # returns str
        except Exception:
            pass
    # fallback
    if default is not None:
        resp = input(f"{prompt_text} [{default}]: ").strip()
        return resp if resp != "" else default
    return input(f"{prompt_text}: ").strip()

def derive_output_path(input_path, provided_output):
    if provided_output:
        return provided_output
    base, ext = os.path.splitext(input_path)
    return f"{base}.variant{ext or '.png'}"

def safe_process_call(**kwargs):
    try:
        return process_image(**kwargs)
    except Exception as e:
        error(f"Processing failed: {e}")
        import traceback
        if console:
            console.print_exception()
        else:
            traceback.print_exc()
        return None

def run_tui():
    info("Mode interactive (TUI). Répondez aux prompts pour configurer l'effet.")
    while True:
        inp = interactive_prompt("Chemin vers l'image d'entrée (ou 'quit' pour sortir)")
        if not inp:
            continue
        if inp.lower() in ("q","quit","exit"):
            info("Au revoir.")
            return
        if not os.path.isfile(inp):
            warn("Fichier introuvable. Réessayez.")
            continue
        out = interactive_prompt("Chemin de sortie (laisser vide pour auto)", default=derive_output_path(inp, None))
        threshold = int(interactive_prompt("Seuil détection (0-255)", default="60"))
        glow_radius = int(interactive_prompt("Glow radius", default="18"))
        glow_strength = float(interactive_prompt("Glow strength 0.0-1.0", default="0.85"))
        noise = int(interactive_prompt("Noise intensity (0 pour désactiver)", default="6"))
        add_text_resp = interactive_prompt("Ajouter texte neon TRHACKNON ? (y/n)", default="y")
        add_text = add_text_resp.strip().lower().startswith("y")
        scanlines_resp = interactive_prompt("Ajouter scanlines ? (y/n)", default="y")
        scanlines = scanlines_resp.strip().lower().startswith("y")
        glitch_resp = interactive_prompt("Appliquer glitch ? (y/n)", default="n")
        glitch = glitch_resp.strip().lower().startswith("y")
        ai_suggest_resp = "n"
        ai_generate_resp = "n"
        ai_prompt = None
        if OPENAI_API_KEY:
            ai_suggest_resp = interactive_prompt("Utiliser GPT-4 pour suggérer les yeux ? (y/n)", default="n")
            ai_generate_resp = interactive_prompt("Utiliser DALL·E pour variante IA ? (y/n)", default="n")
            if ai_generate_resp.strip().lower().startswith("y"):
                ai_prompt = interactive_prompt("Prompt pour DALL·E (laisser vide pour prompt automatique)", default="")
        # call
        res = safe_process_call(
            input_path=inp,
            output_path=out,
            threshold=threshold,
            glow_radius=glow_radius,
            glow_strength=glow_strength,
            add_text=add_text,
            font_path=None,
            scanlines=scanlines,
            scan_spacing=3,
            noise_level=noise,
            glitch=glitch,
            glitch_bands=12,
            force_eyes=False,
            ai_suggest=(ai_suggest_resp.strip().lower().startswith("y"))
        )
        if res:
            info(f"Image générée: {res}")
            if OPENAI_API_KEY and ai_generate_resp.strip().lower().startswith("y"):
                prompt_to_use = ai_prompt if ai_prompt else f"Create a highly stylized, eerie, devil-eyed variant of the provided image. Emphasize intense red neon glows in the eyes, with a dark, cyberpunk hacker atmosphere. Include subtle symbols of anti-capitalism, digital anarchy, and Anonymous-style masks. Use high-contrast neon colors, glitch effects, scanlines, and a futuristic, underground hacker aesthetic, while keeping the original image composition recognizable."
                success = ai_generate_with_dalle(prompt_to_use, inp, out)
                if success:
                    info("DALL·E a généré une image (sauvegardée).")
                else:
                    warn("DALL·E n'a pas pu générer l'image.")
        else:
            warn("Aucun fichier généré.")
        cont = interactive_prompt("Traiter une autre image ? (y/n)", default="n")
        if not cont.strip().lower().startswith("y"):
            info("Fin de session interactive.")
            break

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    p = build_arg_parser()
    args = p.parse_args(argv)

    # quiet mode handling
    if args.quiet and console:
        # replace console print helpers with no-ops
        global info, warn
        def info(msg): pass
        def warn(msg): pass

    if not args.input:
        # enter interactive mode
        run_tui()
        return 0

    input_path = args.input
    if not os.path.isfile(input_path):
        error(f"Input file not found: {input_path}")
        return 2

    output_path = derive_output_path(input_path, args.output)

    # basic sanity bounds
    threshold = max(0, min(255, args.threshold))
    glow_radius = max(0, args.glow_radius)
    glow_strength = float(max(0.0, min(1.0, args.glow_strength)))
    noise_level = max(0, args.noise)
    scanlines = not args.no_scanlines
    add_text = not args.no_text

    # Process
    result = safe_process_call(
        input_path=input_path,
        output_path=output_path,
        threshold=threshold,
        glow_radius=glow_radius,
        glow_strength=glow_strength,
        add_text=add_text,
        font_path=args.font,
        scanlines=scanlines,
        scan_spacing=args.scan_spacing,
        noise_level=noise_level,
        glitch=args.glitch,
        glitch_bands=args.glitch_bands,
        force_eyes=args.force_eyes,
        ai_suggest=args.ai_suggest
    )

    if not result:
        error("Erreur lors du traitement de l'image.")
        return 3

    # optionally call DALL·E / image edit if requested
    if args.ai_generate:
        if not openai_client:
            warn("OPENAI_API_KEY non configuré ou client OpenAI non disponible — génération IA impossible.")
        else:
            ai_prompt = args.ai_prompt if args.ai_prompt else (
                "Create a highly stylized, dark and eerie devil-eyed variant of the provided image. Emphasize intense red neon glows in the eyes, with a deep cyberpunk hacker atmosphere. Include a hooded figure wearing a Guy Fawkes mask, surrounded by subtle symbols of anti-capitalism, digital anarchy, and Anonymous-style rebellion. Use high-contrast neon colors, glitch effects, scanlines, and a gritty underground hacker aesthetic. Integrate the developer's name subtly as graffiti or code in the environment, without interfering with the main composition. Keep the original image composition recognizable."
            )
            ok = ai_generate_with_dalle(ai_prompt, input_path, output_path)
            if not ok:
                warn("La génération IA a échoué. Le fichier local a été produit par la pipeline offline.")
            else:
                info(f"Image IA sauvegardée -> {output_path}")

    info("Terminé.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
