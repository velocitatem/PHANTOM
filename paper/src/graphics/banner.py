from PIL import Image, ImageDraw, ImageFont

text = open("banner.txt", "r", encoding="utf-8").read()

scale = 4          # 2–6 is typical
pad = 10
font_px = 18

font = ImageFont.truetype("DejaVuSansMono.ttf", font_px * scale)

# Measure at high res
dummy = Image.new("RGB", (1, 1), "white")
d = ImageDraw.Draw(dummy)
bbox = d.multiline_textbbox((0, 0), text, font=font)
w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

# Render at high res
hi = Image.new("RGB", (w + 2*pad*scale, h + 2*pad*scale), "white")
d = ImageDraw.Draw(hi)
d.multiline_text((pad*scale, pad*scale), text, font=font, fill="black")

# Downscale with a good filter
out = hi.resize((hi.width // scale, hi.height // scale), resample=Image.Resampling.LANCZOS)
out.save("banner.png", dpi=(300, 300))
