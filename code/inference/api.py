import io
import torch
import base64
import os.path
import numpy as np
import torchvision.transforms.v2 as transforms

from PIL import Image
from transformers import ViTMAEForPreTraining
from flask import Flask, jsonify, request, send_file, abort


DATASET_DIRECTORY = "../data/dataset"
MODEL_DIRECTORY = "trained/24-12-30_mae-swissaerial_conditional_as17/checkpoint_939/"

model = ViTMAEForPreTraining.from_pretrained(MODEL_DIRECTORY)
model.vit.embeddings.config.mask_ratio = 0.75


def load_image(name):
    base_img = Image.open(os.path.join(DATASET_DIRECTORY, name)).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.5], [0.5]),
    ])

    return transform(base_img)

def load_base64_image(base64_img):
    image_bytes = base64.b64decode(base64_img)
    image = Image.open(io.BytesIO(image_bytes))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.5], [0.5]),
    ])

    return transform(image)

def load_segmentation(segmentation_json):
    segmentation = torch.IntTensor(segmentation_json)

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST_EXACT)
    ])

    segmentation = segmentation.unsqueeze(0)
    segmentation = transform(segmentation)

    return segmentation

def eval(base_img, segmentation):
    input = torch.cat([base_img, segmentation], dim=0)
    input = input.unsqueeze(0)

    outputs = model(input)

    y = model.unpatchify(outputs.logits)
    y = y.detach().cpu()
    y = extract_rgb(y)
    y = y.squeeze(0)


    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 *4)
    mask = model.unpatchify(mask)
    mask = extract_rgb(mask)
    # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    mask = mask.squeeze(0)

    x = base_img.detach().cpu()
    # x = torch.einsum('chw->hwc', pixel_values)

    print(x.shape)
    print(y.shape)
    print(mask.shape)

    im_paste = x * (1 - mask) + y * mask
    im_paste = im_paste.squeeze(0)


    transform = transforms.Compose([ 
        transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.5, 1/0.5, 1/0.5 ]),
        transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ], std = [ 1., 1., 1. ]),
        transforms.ToPILImage(mode='RGB')
    ])
    y = transform(im_paste)

    img_io = io.BytesIO()
    y.save(img_io, format='PNG')
    img_io.seek(0)

    img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
    return img_base64


def extract_rgb(tensor):
    return tensor[:,:3, :, :]

app = Flask(__name__, static_folder=DATASET_DIRECTORY)

@app.route('/base/<filename>', methods=['GET'])
def get_image(filename):
    try:
        file_path = os.path.join(DATASET_DIRECTORY, 'base', f"{filename}.png")

        if os.path.isfile(file_path):
            try:
                return send_file(file_path, as_attachment=False)
            except Exception as e:
                abort(500, description=f"Error serving file: {str(e)}")
        else:
            abort(404, description="Image not found")
    except FileNotFoundError:
        abort(404, description="Image not found")

@app.route('/seg/<filename>', methods=['GET'])
def get_segmentation(filename):
    try:
        file_path = os.path.join(DATASET_DIRECTORY, 'lulc_segmentation_as17_npy', f"{filename}.npy")

        if os.path.isfile(file_path):
            try:
                segmentation = np.load(file_path)
                segmentation = torch.from_numpy(segmentation)

                transform = transforms.Compose([
                    transforms.Resize((14, 14), interpolation=transforms.InterpolationMode.NEAREST_EXACT)
                ])

                segmentation = segmentation.unsqueeze(0)
                segmentation = transform(segmentation)
                segmentation = segmentation.squeeze(0).tolist()

                return jsonify(segmentation)
            except Exception as e:
                abort(500, description=f"Error serving file: {str(e)}")
        else:
            abort(404, description="File not found")
    except FileNotFoundError:
        abort(404, description="File not found")

@app.route('/reeval', methods=['POST'])
def reeval_endpoint():
    segmentation = load_segmentation(request.json['segmentation'])
    base_img = load_base64_image(request.json['image'])

    img_base64 = eval(base_img, segmentation)
    return jsonify({"image": img_base64})

@app.route('/eval', methods=['POST'])
def eval_endpoint():
    segmentation = load_segmentation(request.json['segmentation'])
    base_img = load_image(f"base/{request.json['base']}.png")

    img_base64 = eval(base_img, segmentation)
    return jsonify({"image": img_base64})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8001)
