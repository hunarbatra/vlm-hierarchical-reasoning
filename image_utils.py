from PIL import Image

import base64
import io

def image_to_base64(image_path: str) -> str:
    with Image.open(image_path) as image:
        buffered = io.BytesIO()

        # Convert image to RGB if it's in RGBA mode
        if image.mode in ("RGBA", "P"):  # P mode means it has a palette
            # This converts to RGB and removes alpha (transparency)
            image = image.convert("RGB")

        # Save the image as JPEG in the buffer
        image.save(buffered, format="JPEG")

        # Get the base64-encoded string
        img_str = base64.b64encode(buffered.getvalue())
        return img_str.decode('utf-8')
    
def save_images(images: list[str], exp_dir: str, filename: str, original: bool = False) -> None:
    if original:
        if "," in images[0]:
            img = images[0].split(",")[1]
        else: 
            img = images[0]
        imgdata = base64.b64decode(img)
        path = f'experiments{exp_dir}/{filename}'
        with open(path, 'wb') as f:
            f.write(imgdata)
        return
    for i, img in enumerate(images):
        # Ensure there's no data URL scheme in the base64 string
        if "," in img:
            img = img.split(",")[1]
        imgdata = base64.b64decode(img)
        path = f'experiments{exp_dir}/{filename}-{i}.jpg'
        with open(path, 'wb') as f:
            f.write(imgdata)