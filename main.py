import cv2
from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from albumentations import (Normalize, Compose)
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import base64


app = Flask(__name__)

CKPT_PATH = "./model/model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = smp.Unet("resnet18", encoder_weights=None, classes=4, activation=None)
MODEL.to(device)
MODEL.eval()
state = torch.load(CKPT_PATH, map_location=lambda storage, loc: storage)
MODEL.load_state_dict(state["state_dict"])

def preprocessing(data_path):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transformer = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )
    data = cv2.imread(data_path)
    image = transformer(image=data)['image']
    return data, image.reshape((-1, 3, 256, 1600)).to(device)


def post_process(output, threshold=0.5):
    probabilities = torch.sigmoid(output).detach().cpu().numpy()
    best_mask_num = -1
    for probability in probabilities:
        mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
        if mask.sum() > best_mask_num:
            best_mask = mask
            best_mask_num = mask.sum()
    return mask, best_mask_num


def image_saver(origin, mask, save_path="./output/output.png"):
    z = np.zeros((256, 1600, 4))
    for w in range(len(mask)):
        for c in range(len(mask[w])):
            if mask[w][c] > 0:
                z[w][c] = np.concatenate((((0.7*origin[w][c])+(0.3*np.array([244, 89, 163]))), [150]))
            else:
                z[w][c] = np.concatenate((origin[w][c], [255]))
    z = np.array(z, dtype=np.uint8)
    plt.imsave(save_path, z)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_file = request.files['source']
        img_path = "./input/" + input_file.filename
        input_file.save(img_path)
        image, data = preprocessing(img_path)
        output = MODEL(data)[0]
        mask, best_mask_num = post_process(output)
        image_saver(image, mask)
        with open("./output/output.png", "rb") as img_file:
            my_string = base64.b64encode(img_file.read())
        response = jsonify({"image": my_string.decode()})
        return response, 200

    except Exception as e:
        return jsonify({'message': 'Error! Please upload another file'}), 400


@app.route('/')
def main():
    return render_template('main.html'), 200


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
