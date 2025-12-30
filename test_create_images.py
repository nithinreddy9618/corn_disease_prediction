from PIL import Image, ImageDraw
import os
os.makedirs('test_images', exist_ok=True)

# Solid green image (uniform healthy color)
img = Image.new('RGB', (224, 224), (34, 139, 34))
img.save('test_images/green.png')

# Simple leaf-like shape drawn on green background
img2 = Image.new('RGB', (224, 224), (200, 255, 200))
d = ImageDraw.Draw(img2)
d.ellipse((20, 40, 204, 184), fill=(34, 139, 34))
d.ellipse((80, 80, 144, 144), fill=(50,205,50))
img2.save('test_images/leaf_like.png')

print('Created test images in test_images/')
