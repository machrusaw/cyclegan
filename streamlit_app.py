import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import io
import gdown

# Fungsi conv
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=False, instance_norm=False):
    """
    Creates a convolutional layer, with optional batch or instance normalization.
    """
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))

    return nn.Sequential(*layers)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1, instance_norm=True)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1, instance_norm=True)

    def forward(self, x):
        out_1 = F.relu(self.conv1(x))
        out_2 = x + self.conv2(out_1)
        return out_2

# Transpose Convolution Layer
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=False, instance_norm=False, dropout=False, dropout_ratio=0.5):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))

    if dropout:
        layers.append(nn.Dropout2d(dropout_ratio))

    return nn.Sequential(*layers)

# Generator (CycleGAN)
class CycleGenerator(nn.Module):
    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()

        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=4)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=4, instance_norm=True)
        self.conv3 = conv(in_channels=conv_dim*2, out_channels=conv_dim*4, kernel_size=4, instance_norm=True)

        res_layers = [ResidualBlock(conv_dim*4) for _ in range(n_res_blocks)]
        self.res_blocks = nn.Sequential(*res_layers)

        self.deconv4 = deconv(in_channels=conv_dim*4, out_channels=conv_dim*2, kernel_size=4, instance_norm=True)
        self.deconv5 = deconv(in_channels=conv_dim*2, out_channels=conv_dim, kernel_size=4, instance_norm=True)
        self.deconv6 = deconv(in_channels=conv_dim, out_channels=3, kernel_size=4)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2)
        out = self.res_blocks(out)
        out = F.leaky_relu(self.deconv4(out), negative_slope=0.2)
        out = F.leaky_relu(self.deconv5(out), negative_slope=0.2)
        out = torch.tanh(self.deconv6(out))
        return out

# Fungsi denormalisasi
def reverse_normalize(image, mean=0.5, std=0.5):
    image = (image * std + mean) * 255
    return np.clip(image, 0, 255).astype(np.uint8)

# Load model from Google Drive using gdown
@st.cache_resource
def load_model_from_drive(model_url, device):
    # Unduh model dari Google Drive
    output_path = "model.pth"
    gdown.download(model_url, output_path, quiet=False)
    
    # Load model
    model = CycleGenerator()
    model.load_state_dict(torch.load(output_path, map_location=device))
    model.eval()
    return model

# Streamlit app
def main():
    st.title("CycleGAN Image Translation")
    st.write("Upload gambar dan lihat hasil transformasi CycleGAN.")

    # Pilih model
    model_option = st.sidebar.selectbox(
        "Pilih model untuk digunakan",
        ("G_XtoY (Domain X ke Y)", "G_YtoX (Domain Y ke X)")
    )

    # URL model di Google Drive
    model_url_G_XtoY = "https://drive.google.com/uc?id=1_qfDTjxTsV4ezAUjE-1vA5YYq2RyvuVk"
    model_url_G_YtoX = "https://drive.google.com/uc?id=1iLimH67dF8ImZeYU1kxSfzCBdfYjQMql"

    # Pilih perangkat
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model sesuai pilihan
    if model_option == "G_XtoY (Domain X ke Y)":
        model = load_model_from_drive(model_url_G_XtoY, device)
        st.write("Model G_XtoY dipilih.")
    else:
        model = load_model_from_drive(model_url_G_YtoX, device)
        st.write("Model G_YtoX dipilih.")

    # Upload gambar
    uploaded_files = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            input_image = Image.open(uploaded_file).convert("RGB")

            transform = transforms.Compose([ 
                transforms.Resize((512, 512)),  # Resolusi lebih tinggi
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

            input_tensor = transform(input_image).unsqueeze(0).to(device)
            with torch.no_grad():
                output_tensor = model(input_tensor)

            output_image = output_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
            output_image = reverse_normalize(output_image)

            # Tampilkan gambar berdampingan
            col1, col2 = st.columns(2)
            with col1:
                st.image(input_image, caption=f"Gambar Asli - {uploaded_file.name}", use_column_width=True)
            with col2:
                st.image(output_image, caption=f"Gambar Hasil - {uploaded_file.name}", use_column_width=True)

            # Tambahkan tombol unduh hasil
            result_image = Image.fromarray(output_image.astype('uint8'))
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label=f"Unduh Gambar Hasil - {uploaded_file.name}",
                data=byte_im,
                file_name=f"hasil_{uploaded_file.name}",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
