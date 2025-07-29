import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from autoencoder import ConvAutoencoder_gated, DenoisingLoss
from pathlib import Path
import matplotlib.pyplot as plt
import piq
from src.utils import full_eval

def train(
    data_path,
    batch_size=64,
    epochs=6,
    lr=1e-4,
    weight_decay=1e-5,
    recon=1,
    smooth=0,
    sparse=0,
    gates_w=15,
    validation_ratio=0.01,
    test_ratio=0.1,
    gate_freeze_epoch=4
):
    # Transforms
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    # Save path for model weights
    model_save_path = f'src/Denoising/weights/weights_(bs={batch_size},ne={epochs},gfe={gate_freeze_epoch},lr={lr},wd={weight_decay},lw={recon},{smooth},{sparse},{gates_w}).pth'

    # Load dataset
    full_dataset = datasets.ImageFolder(root=Path(data_path), transform=transform)
    train_size = int(len(full_dataset) * (1 - validation_ratio - test_ratio))
    validation_size = int(len(full_dataset) * validation_ratio)
    test_size = len(full_dataset) - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = random_split(full_dataset, [train_size, validation_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    # Model, Loss, Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvAutoencoder_gated().to(device)
    criterion = DenoisingLoss(recon_weight=recon, smooth_weight=smooth, sparsity_weight=sparse, gates_weight=gates_w).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=5, verbose=True)

    val_losses = []
    val_ssim_scores = []
    val_psnr_scores = []

    print("Starting training...")
    for epoch in range(epochs):
        model.train()

        if epoch == gate_freeze_epoch:
            for param in model.gate1.parameters():
                param.requires_grad = False
            for param in model.gate2.parameters():
                param.requires_grad = False
            for param in model.gate3.parameters():
                param.requires_grad = False
            criterion.gates_weight = 0
            criterion.recon_weight = 0.5

        running_loss = 0.0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            outputs, encoded, gates = model(images)
            loss = criterion(outputs, images, encoded, gates)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        ssim_total = 0.0
        psnr_total = 0.0
        num_batches = 0

        with torch.no_grad():
            for val_images, _ in val_loader:
                val_images = val_images.to(device)
                val_outputs, val_encoded, val_gates = model(val_images)
                loss = criterion(val_outputs, val_images, val_encoded, val_gates)
                val_loss += loss.item() * val_images.size(0)

                gates_loss = sum(mask.mean() for mask in val_gates)
                gate_contribution = gates_w * gates_loss
                gate_ratio = (gate_contribution.item() / loss.item()) if epoch <= gate_freeze_epoch else 0

                # print(f"Total Loss: {loss.item():.4f}, " # legacy will be removed
                #       f"Gate Loss: {gate_contribution.item():.4f}, "
                #       f"Gate Open criterion: {criterion.gates_loss / 4:.4f}, "
                #       f"Gate Open gates: {gates_loss / 4:.4f}, "
                #       f"Contribution: {gate_ratio * 100:.2f}%")

                for i in range(val_images.size(0)):
                    inp = val_images[i].unsqueeze(0)
                    out = val_outputs[i].unsqueeze(0)
                    ssim = piq.ssim(inp, out, data_range=1.0).item()
                    psnr = piq.psnr(inp, out, data_range=1.0).item()
                    ssim_total += ssim
                    psnr_total += psnr

                num_batches += val_images.size(0)

                # fig, axes = plt.subplots(2, 2, figsize=(6, 6)) #legacy will be removed
                # for i in range(2):
                #     inp_img = val_images[i].squeeze().cpu().numpy()
                #     out_img = val_outputs[i].squeeze().cpu().numpy()
                #     axes[i][0].imshow(inp_img, cmap='gray')
                #     axes[i][0].set_title(f"Validation Input {i}")
                #     axes[i][0].axis('off')
                #     axes[i][1].imshow(out_img, cmap='gray')
                #     axes[i][1].set_title(f"Reconstruction {i}")
                #     axes[i][1].axis('off')
                # plt.tight_layout()
                # plt.show()

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_ssim = ssim_total / num_batches
        avg_psnr = psnr_total / num_batches

        val_losses.append(avg_val_loss)
        val_ssim_scores.append(avg_ssim)
        val_psnr_scores.append(avg_psnr)

        # print(f"Validation — Loss: {avg_val_loss:.4f}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f} dB") #legacy will be removed

    # Plotting
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, val_losses, marker='o')
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, val_ssim_scores, marker='o', color='green')
    plt.title("Validation SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, val_psnr_scores, marker='o', color='orange')
    plt.title("Validation PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")

    plt.tight_layout()
    plt.savefig('Training Results.png')
    #plt.show()

    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")
    full_eval(model_class=ConvAutoencoder_gated, model_weights_path=model_save_path, data_loader=test_loader)