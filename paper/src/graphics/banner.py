from PIL import Image, ImageDraw, ImageFont

text = open("banner.txt", "r", encoding="utf-8").read()
font = ImageFont.truetype("DejaVuSansMono.ttf", 18)

dummy = Image.new("RGB", (1, 1))
d = ImageDraw.Draw(dummy)
bbox = d.multiline_textbbox((0,0), text, font=font)
w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]

img = Image.new("RGB", (w+20, h+20), "white")
d = ImageDraw.Draw(img)
d.multiline_text((10,10), text, font=font, fill="black")
img.save("banner.png")
