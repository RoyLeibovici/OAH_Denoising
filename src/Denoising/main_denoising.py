from pathlib import Path
from data_preperation import plot_histogram_of_cell_dimensions, discard_cells, resize_cells

# Configure paths
current_file = Path(__file__).resolve()
project_path = current_file.parents[2]
original_cells = Path(project_path, "Data", "Output_files", "HTB5-170122", "Original Cells")
filtered_cells = Path(project_path, "Data", "Output_files", "HTB5-170122", "filtered_cells")

# Prepare data for autoencoder
discard_cells(original_cells, filtered_cells)
plot_histogram_of_cell_dimensions(filtered_cells)
dataset_path = Path(project_path, "Data", "Output_files", "HTB5-170122", "Dataset")
resize_cells(filtered_cells, dataset_path, target_size=(128, 128))

# Declare autencoder parameters

"""dasdsa
ads
sad
sad"""


# Train autoencoder