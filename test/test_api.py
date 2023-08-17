import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin
import time
import os
from gradio.processing_utils import encode_pil_to_base64
import sys

start_time = time.time()
count = 1

url = "http://127.0.0.1:8080"

payload_file = 'payload.json'
f = open(payload_file)
aigc_params = json.load(f)

payload={}
payload['task'] = 'txt2img'
payload['txt2img_payload'] = {"denoising_strength": 0.75, "prompt": "a flower", "styles": [], "seed": 12345, "subseed": 23456, "subseed_strength": 0.0, "seed_resize_from_h": 0, "seed_resize_from_w": 0, "sampler_name": "Euler a", "batch_size": 1, "n_iter": 1, "steps": 20, "cfg_scale": 7.0, "width": 1024, "height": 1024, "negative_prompt": "", "eta": 1, "s_churn": 0, "s_tmax": 1, "s_tmin": 0, "s_noise": 1, "override_settings": {}, "script_name": "", "script_args": []}
#payload['task'] = 'img2img'
#payload['img2img_payload'] = aigc_params['img2img_payload']

for i in range(count):
    response_diffuer = requests.post(url=f'{url}/invocations', json=payload)

print(f"diffuser average run time is {(time.time()-start_time)/count}")

start_time = time.time()
# for i in range(count):
#     response_local = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload['img2img_payload'])

for i in range(count):
   response_local = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload['txt2img_payload'])


print(f"webui average run time is {(time.time()-start_time)/count}")

#print(f"response is {response.json()}")

r = response_diffuer.json()

id = 0
for i in r['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

    png_payload = {
        "image": "data:image/png;base64," + i
    }
    response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", response2.json().get("info"))
    image.save('output_diffuser_%d.png'%id, pnginfo=pnginfo)
    id += 1

r = response_local.json()

id = 0
for i in r['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

    png_payload = {
        "image": "data:image/png;base64," + i
    }
    response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", response2.json().get("info"))
    image.save('output_local_%d.png'%id, pnginfo=pnginfo)
    id += 1

