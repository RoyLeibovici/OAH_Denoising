from pathlib import Path
import csv
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import piq
from src.utils import compute_speckle_contrast, compute_gradient_energy, compute_laplacian_energy, compute_freq_sharpness

def save_images(i, inp_img, out_img, dir):
    output_dir = Path(dir) / "Results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to uint8
    inp_uint8 = (inp_img * 255).astype(np.uint8)
    out_uint8 = (out_img * 255).astype(np.uint8)

    # Save input and output as separate files
    cv2.imwrite(str(output_dir / f"image_{i:02d}_input.png"), inp_uint8)
    cv2.imwrite(str(output_dir / f"image_{i:02d}_output.png"), out_uint8)

def export_metrics_csv(csv_path, filenames, metrics_dict):
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

def evaluate_autoencoder(model_class, model_weights_path, input_folder, output_folder,save=True, max_images=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_class().to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    image_paths = sorted(Path(input_folder).glob("*"))[:max_images]
    images_tensor = []

    filenames = []
    ssim_scores = []
    psnr_scores = []
    sc_input_scores = []
    grad_input_scores = []
    lap_input_scores = []
    sharp_input_scores = []
    sc_output_scores = []
    grad_output_scores = []
    lap_output_scores = []
    sharp_output_scores = []

    for path in image_paths:
        filenames.append(path.name)
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        images_tensor.append(img_tensor)

        sc_input_scores.append(compute_speckle_contrast(path))
        grad_input_scores.append(compute_gradient_energy(path))
        lap_input_scores.append(compute_laplacian_energy(path))
        sharp_input_scores.append(compute_freq_sharpness(path))

    images_tensor = torch.stack(images_tensor).to(device)

    with torch.no_grad():
        outputs, _, _ = model(images_tensor)
        outputs = outputs.clamp(0, 1)

    for i in range(len(images_tensor)):
        inp = images_tensor[i].unsqueeze(0)
        out = outputs[i].unsqueeze(0)

        ssim_scores.append(piq.ssim(inp, out, data_range=1.0).item())
        psnr_scores.append(piq.psnr(inp, out, data_range=1.0).item())

        out_np = out.squeeze().cpu().numpy()
        out_path = Path(output_folder) / f"temp_output_{i:03d}.png"
        cv2.imwrite(str(out_path), (out_np * 255).astype(np.uint8))

        sc_output_scores.append(compute_speckle_contrast(out_path))
        grad_output_scores.append(compute_gradient_energy(out_path))
        lap_output_scores.append(compute_laplacian_energy(out_path))
        sharp_output_scores.append(compute_freq_sharpness(out_path))

        if save:
            save_images(i, inp.squeeze().cpu().numpy(), out_np, output_folder)

        out_path.unlink()

    metrics_dict = {
        'ssim': ssim_scores,
        'psnr': psnr_scores,
        'sc_input': sc_input_scores,
        'sc_output': sc_output_scores,
        'grad_input': grad_input_scores,
        'grad_output': grad_output_scores,
        'lap_input': lap_input_scores,
        'lap_output': lap_output_scores,
        'sharp_input': sharp_input_scores,
        'sharp_output': sharp_output_scores
    }

    csv_path = Path(output_folder) / "Results" / "metrics_summary.csv"
    export_metrics_csv(csv_path, filenames, metrics_dict)

    # Plotting
    fig, axes = plt.subplots(len(images_tensor), 4, figsize=(12, len(images_tensor) * 2))
    if len(images_tensor) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(len(images_tensor)):
        inp_img = images_tensor[i].squeeze().cpu().numpy()
        out_img = outputs[i].squeeze().cpu().numpy()
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
    plt.savefig(Path(output_folder) / "Results" / "summary.png")

    # Show average metrics table
    avg_data = [
        ["Metric", "Input", "Output"],
        ["SSIM", "-", f"{np.mean(ssim_scores):.4f}"],
        ["PSNR", "-", f"{np.mean(psnr_scores):.2f} dB"],
        ["Speckle Contrast", f"{np.mean(sc_input_scores):.4f}", f"{np.mean(sc_output_scores):.4f}"],
        ["Gradient Energy", f"{np.mean(grad_input_scores):.2f}", f"{np.mean(grad_output_scores):.2f}"],
        ["Laplacian Energy", f"{np.mean(lap_input_scores):.2f}", f"{np.mean(lap_output_scores):.2f}"],
        ["Freq Sharpness", f"{np.mean(sharp_input_scores):.4f}", f"{np.mean(sharp_output_scores):.4f}"],
    ]

    fig, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=avg_data, colLabels=None, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.6)
    plt.title("Evaluation Metrics Summary", fontsize=14)
    plt.savefig(Path(input_folder) / "Results" / "table_results.png")
