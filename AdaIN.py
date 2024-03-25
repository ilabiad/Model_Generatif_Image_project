import torch

from AdaIN_utils import *
from torchvision import transforms
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import os
from PIL import Image

# credit to : https://github.com/ziwei-jiang/AdaIN-Style-Transfer-PyTorch/tree/master

decoder_path = "saved_models/decoder.pth"
encoder_path = "saved_models/encoder_state_dict.pt"


class AdaIN:
    img_size = 512
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    def __init__(self, encoder_sd_path=encoder_path, decoder_sd_path=decoder_path):
        self.model = StyleTransferNet(encoder_sd_path=encoder_sd_path, decoder_sd_path=decoder_sd_path)
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def transfer_style(self, content, style):
        with torch.no_grad():
            self.model.eval()
            content_image = self.transform(content.convert("RGB")).unsqueeze(0).to(self.device)
            style_image = self.transform(style.convert("RGB")).unsqueeze(0).to(self.device)

            output = self.model([content_image, style_image], alpha=1.0)
        return output

    def get_content_from_image(self, image, pil=True):
        eps = 1e-5
        if pil:
            content_image = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        else:
            content_image = image.to(self.device)
        with torch.no_grad():
            self.model.eval()
            encoder_content = self.model.encoder(content_image)
            mean_x = torch.mean(encoder_content, dim=[2, 3])
            mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)
            std_x = torch.std(encoder_content, dim=[2, 3])
            std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps
        return (encoder_content - mean_x) / std_x

    def get_style_from_image(self, image, pil=True):
        if pil:
            style_image = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        else:
            style_image = image.to(self.device)

        with torch.no_grad():
            self.model.eval()
            fm11_style = self.model.encoder[:3](style_image)
            fm21_style = self.model.encoder[3:8](fm11_style)
            fm31_style = self.model.encoder[8:13](fm21_style)
            encode_style = self.model.encoder[13:](fm31_style)

            return torch.concat((
                torch.mean(fm11_style, dim=[2, 3]),
                torch.mean(fm21_style, dim=[2, 3]),
                torch.mean(fm31_style, dim=[2, 3]),
                torch.mean(encode_style, dim=[2, 3]),
                torch.std(fm11_style, dim=[2, 3]),
                torch.std(fm21_style, dim=[2, 3]),
                torch.std(fm31_style, dim=[2, 3]),
                torch.std(encode_style, dim=[2, 3])
            ), dim=1)


    def postprocess(self, image):
        tmp = image[0].detach().cpu().permute(1, 2, 0)
        m = tmp.view(-1, 3).min(axis=0)[0]
        M = tmp.view(-1, 3).max(axis=0)[0]
        out_img = (tmp - m) / (M - m)
        return out_img

    def display_image(self, image, pil=False):
        if not pil:
            out_img = self.postprocess(image)
        else:
            out_img = image
        imshow(out_img)

    def save_output_image(self, image, output_dir="./generated_images", filename="style_transfert.jpg", pil=False):
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        if not pil:
            out_img = self.postprocess(image)
        else:
            out_img = image
        plt.imshow(out_img)
        plt.axis('off')
        plt.savefig(save_path)
        #print(f"Saved output image to: {save_path}")