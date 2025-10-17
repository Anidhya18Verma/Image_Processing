import cv2
import numpy as np
import gradio as gr

# ---------- Image Processing ----------

def pixora_pop(img):
    img_float = img.astype(np.float32)/255.0
    contrast = 1.3
    img_float = np.clip((img_float-0.5)*contrast+0.5,0,1)
    hsv = cv2.cvtColor((img_float*255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = cv2.add(hsv[:,:,1],40)
    img_saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blur = cv2.GaussianBlur(img_saturated,(0,0),sigmaX=5,sigmaY=5)
    pop = cv2.addWeighted(img_saturated,1.3,blur,0.7,0)
    return pop

def apply_filter(img, name):
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if name=="grayscale":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif name=="sepia":
        kernel = np.array([[0.272,0.534,0.131],
                           [0.349,0.686,0.168],
                           [0.393,0.769,0.189]])
        return np.clip(cv2.transform(img,kernel),0,255).astype(np.uint8)
    elif name=="invert":
        return cv2.bitwise_not(img)
    elif name=="vintage":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
    elif name=="sharpen":
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        return cv2.filter2D(img,-1,kernel)
    return img

def adjust_brightness_contrast(img, brightness=0, contrast=1.0):
    img_float = img.astype(np.float32)/255.0
    img_float = np.clip(img_float*contrast + brightness/100.0,0,1)
    return (img_float*255).astype(np.uint8)

def adjust_saturation(img, saturation=1.0):
    if len(img.shape)==2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*saturation,0,255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ---------- Gradio Processing ----------

def process_image(img, filter_name, pop, brightness, contrast, saturation):
    if img is None:
        return None
    original = img.copy()
    img_proc = img.copy()
    if pop:
        img_proc = pixora_pop(img_proc)
    img_proc = apply_filter(img_proc, filter_name)
    img_proc = adjust_brightness_contrast(img_proc, brightness, contrast/100)
    img_proc = adjust_saturation(img_proc, saturation/100)

    # Side-by-side Before/After
    if len(img_proc.shape)==2:
        img_proc = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2BGR)
    if len(original.shape)==2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    combined = np.concatenate((original, img_proc), axis=1)
    return combined

# ---------- Gradio Interface ----------

filters = ["None", "grayscale","sepia","invert","vintage","sharpen"]

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Image"),
        gr.Dropdown(choices=filters, label="Filter"),
        gr.Checkbox(label="PixORA Pop"),
        gr.Slider(-100,100,value=0,label="Brightness"),
        gr.Slider(50,200,value=100,label="Contrast"),
        gr.Slider(0,300,value=100,label="Saturation")
    ],
    outputs=gr.Image(type="numpy", label="Before / After"),
    live=True,
    title="PixORA - Professional Image Editor",
    description="Upload your image and apply filters, PixORA Pop effect, and adjust sliders in real-time. Original (left) vs Processed (right) comparison."
)

iface.launch(share=True)
