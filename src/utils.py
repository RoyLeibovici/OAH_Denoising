import cv2
import numpy as np
import torch
import torch.nn.functional as F
import csv
import piq
from pathlib import Path
import matplotlib.pyplot as plt

def compute_speckle_contrast(image_path):
    """
    Compute speckle contrast from a grayscale .png image using OpenCV.

    Args:
        image_path (str or Path): Path to the .png grayscale image

    Returns:
        float: Speckle contrast (std / mean)
    """
    # Load grayscale image as float32
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Normalize to [0, 1]
    image /= 255.0

    mean_intensity = np.mean(image)
    std_intensity = np.std(image)

    speckle_contrast = std_intensity / mean_intensity
    return speckle_contrast

def compute_speckle_contrast_tensor(image_tensor: torch.Tensor) -> float:
    """
    Compute speckle contrast from a grayscale PyTorch tensor.

    Args:
        image_tensor (Tensor): A 2D tensor (H, W) or 3D (1, H, W), with values in [0, 1]

    Returns:
        float: Speckle contrast (std / mean)
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.squeeze(0)  # Remove channel dimension if exists

    mean_intensity = torch.mean(image_tensor)
    std_intensity = torch.std(image_tensor)

    speckle_contrast = std_intensity / mean_intensity
    return speckle_contrast.item()

def compute_gradient_energy(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    # Use Sobel operator
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    gradient_energy = np.sum(grad_x**2 + grad_y**2)
    return gradient_energy

def compute_gradient_energy_tensor(image_tensor: torch.Tensor) -> float:
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.squeeze(0)  # (H, W)

    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    device = image_tensor.device  # <<< get device

    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    grad_x = F.conv2d(image_tensor, sobel_kernel_x, padding=1)
    grad_y = F.conv2d(image_tensor, sobel_kernel_y, padding=1)

    energy = (grad_x ** 2 + grad_y ** 2).sum()
    return energy.item()

def compute_laplacian_energy(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(image, cv2.CV_32F)
    laplacian_energy = np.sum(laplacian ** 2)
    return laplacian_energy

def compute_laplacian_energy_tensor(image_tensor: torch.Tensor) -> float:
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.squeeze(0)  # (H, W)

    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    device = image_tensor.device  # <<< get device
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    laplacian = F.conv2d(image_tensor, laplacian_kernel, padding=1)
    energy = (laplacian ** 2).sum()
    return energy.item()

def compute_freq_sharpness(image_path, low_freq_radius=10):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    H, W = magnitude.shape
    center = (H // 2, W // 2)

    Y, X = np.ogrid[:H, :W]
    distance = np.sqrt((X - center[1])**2 + (Y - center[0])**2)

    high_freq_mask = distance > low_freq_radius

    high_energy = np.sum(magnitude[high_freq_mask]**2)
    total_energy = np.sum(magnitude**2)

    sharpness = high_energy / total_energy
    return sharpness

def compute_freq_sharpness_tensor(image_tensor: torch.Tensor, low_freq_radius=10) -> float:
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.squeeze(0)  # (H, W)

    H, W = image_tensor.shape
    fft = torch.fft.fft2(image_tensor)
    fft_shifted = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shifted)

    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_y, center_x = H // 2, W // 2
    distance = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = distance > low_freq_radius

    high_energy = torch.sum(magnitude[mask] ** 2)
    total_energy = torch.sum(magnitude ** 2)

    sharpness = high_energy / total_energy
    return sharpness.item()

def export_metrics_csv_full_eval(csv_path, filenames, metrics_dict):
    headers = [
        "Filename", "SSIM", "PSNR",
        "SC_Input", "SC_Output",
        "Grad_Input", "Grad_Output",
        "Lap_Input", "Lap_Output",
        "Sharp_Input", "Sharp_Output"
    ]

    rows = []
    for i, name in enumerate(filenames):
        row = [
            name,
            f"{metrics_dict['ssim'][i]:.4f}", f"{metrics_dict['psnr'][i]:.2f}",
            f"{metrics_dict['sc_input'][i]:.4f}", f"{metrics_dict['sc_output'][i]:.4f}",
            f"{metrics_dict['grad_input'][i]:.2f}", f"{metrics_dict['grad_output'][i]:.2f}",
            f"{metrics_dict['lap_input'][i]:.2f}", f"{metrics_dict['lap_output'][i]:.2f}",
            f"{metrics_dict['sharp_input'][i]:.4f}", f"{metrics_dict['sharp_output'][i]:.4f}"
        ]
        rows.append(row)

    # Append average row
    row_avg = [
        "Average",
        f"{np.mean(metrics_dict['ssim']):.4f}", f"{np.mean(metrics_dict['psnr']):.2f}",
        f"{np.mean(metrics_dict['sc_input']):.4f}", f"{np.mean(metrics_dict['sc_output']):.4f}",
        f"{np.mean(metrics_dict['grad_input']):.2f}", f"{np.mean(metrics_dict['grad_output']):.2f}",
        f"{np.mean(metrics_dict['lap_input']):.2f}", f"{np.mean(metrics_dict['lap_output']):.2f}",
        f"{np.mean(metrics_dict['sharp_input']):.4f}", f"{np.mean(metrics_dict['sharp_output']):.4f}"
    ]
    rows.append(row_avg)

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def full_eval(model_class, model_weights_path, data_loader, max_visuals=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    ssim_scores = []
    psnr_scores = []
    filenames = [f"image_{i}" for i in range(len(data_loader))]
    metrics = {
        'speckle_contrast_in': [],
        'speckle_contrast_out': [],
        'gradient_energy_in': [],
        'gradient_energy_out': [],
        'laplacian_energy_in': [],
        'laplacian_energy_out': [],
        'freq_sharpness_in': [],
        'freq_sharpness_out': [],
    }

    visd_inputs = []
    visd_outputs = []

    with torch.no_grad():
        for batch, _ in data_loader:
            inputs = batch.to(device)
            outputs, _, _ = model(inputs)
            outputs = outputs.clamp(0, 1)

            for i in range(inputs.size(0)):
                inp = inputs[i].unsqueeze(0)
                out = outputs[i].unsqueeze(0)

                ssim = piq.ssim(inp, out, data_range=1.0).item()
                psnr = piq.psnr(inp, out, data_range=1.0).item()

                ssim_scores.append(ssim)
                psnr_scores.append(psnr)

                img_in = inp.squeeze()
                img_out = out.squeeze()

                metrics['speckle_contrast_in'].append(compute_speckle_contrast_tensor(img_in))
                metrics['speckle_contrast_out'].append(compute_speckle_contrast_tensor(img_out))

                metrics['gradient_energy_in'].append(compute_gradient_energy_tensor(img_in))
                metrics['gradient_energy_out'].append(compute_gradient_energy_tensor(img_out))

                metrics['laplacian_energy_in'].append(compute_laplacian_energy_tensor(img_in))
                metrics['laplacian_energy_out'].append(compute_laplacian_energy_tensor(img_out))

                metrics['freq_sharpness_in'].append(compute_freq_sharpness_tensor(img_in))
                metrics['freq_sharpness_out'].append(compute_freq_sharpness_tensor(img_out))

                if len(visd_inputs) < max_visuals:
                    visd_inputs.append(img_in.cpu().numpy())
                    visd_outputs.append(img_out.cpu().numpy())

    metrics_dict = {
        'ssim': ssim_scores,
        'psnr': psnr_scores,
        'sc_input': metrics['speckle_contrast_in'],
        'sc_output': metrics['speckle_contrast_out'],
        'grad_input': metrics['gradient_energy_in'],
        'grad_output': metrics['gradient_energy_out'],
        'lap_input': metrics['laplacian_energy_in'],
        'lap_output': metrics['laplacian_energy_out'],
        'sharp_input': metrics['freq_sharpness_in'],
        'sharp_output': metrics['freq_sharpness_out']
    }

    result_dir = Path(model_weights_path) / "Full_Eval_Results"
    result_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(model_weights_path) / "Full_Eval_Results" / "metrics_summary.csv"
    export_metrics_csv_full_eval(csv_path, filenames, metrics_dict)
    # Visualization
    fig, axes = plt.subplots(len(visd_inputs), 4, figsize=(12, len(visd_inputs) * 2))
    if len(visd_inputs) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(len(visd_inputs)):
        inp_img = visd_inputs[i]
        out_img = visd_outputs[i]
        diff_img = inp_img - out_img

        axes[i][0].imshow(inp_img, cmap='gray')
        axes[i][0].set_title(f"Input")
        axes[i][0].axis('off')

        axes[i][1].imshow(out_img, cmap='gray')
        axes[i][1].set_title(f"Output\nSSIM: {ssim_scores[i]:.4f}\nPSNR: {psnr_scores[i]:.2f} dB")
        axes[i][1].axis('off')

        axes[i][2].imshow(diff_img, cmap='bwr', vmin=-1, vmax=1)
        axes[i][2].set_title("Difference")
        axes[i][2].axis('off')

        axes[i][3].hist(diff_img.flatten(), bins=511, range=(-1, 1), color='gray')
        axes[i][3].set_title("Diff Histogram")
        axes[i][3].set_xlim(-1, 1)
        axes[i][3].set_yticks([])

    plt.tight_layout()
    plt.savefig(Path(model_weights_path).parent / "Full_Eval_Results" / "summary.png")

    # Average Metrics Summary
    avg_data = [
        ["Metric", "Input", "Output"],
        ["SSIM", "-", f"{np.mean(ssim_scores):.4f}"],
        ["PSNR", "-", f"{np.mean(psnr_scores):.2f} dB"],
        ["Speckle Contrast", f"{np.mean(metrics['speckle_contrast_in']):.4f}", f"{np.mean(metrics['speckle_contrast_out']):.4f}"],
        ["Gradient Energy", f"{np.mean(metrics['gradient_energy_in']):.2f}", f"{np.mean(metrics['gradient_energy_out']):.2f}"],
        ["Laplacian Energy", f"{np.mean(metrics['laplacian_energy_in']):.2f}", f"{np.mean(metrics['laplacian_energy_out']):.2f}"],
        ["Frequency Sharpness", f"{np.mean(metrics['freq_sharpness_in']):.4f}", f"{np.mean(metrics['freq_sharpness_out']):.4f}"],
    ]

    fig, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=avg_data, colLabels=None, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.title("Evaluation Metrics Summary", fontsize=14)
    plt.savefig(Path(model_weights_path).parent / "Full_Eval_Results" / "table_results.png")
