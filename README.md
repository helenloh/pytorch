# pytorch
# Dataset : MNIST
https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=mnist

# Using simple pytorch to predict MNIST
    Import the necessary libraries:
        import matplotlib.pyplot as plt for visualization.
        import torchvision and import torchvision.transforms as transforms for working with the MNIST dataset and image transformations.

    Define the transformation for the dataset:
        transform = transforms.Compose([transforms.ToTensor()]) converts images to PyTorch tensors.

    Prepare the test dataset and data loader:
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True) creates the test dataset using MNIST, specifying that it's the test split and applying the defined transformation.
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10, shuffle=False) creates a data loader to iterate through the test dataset in mini-batches of size 10 without shuffling.

    Get a batch of images and labels:
        images, labels = next(iter(test_loader)) retrieves the next batch of 10 images and their corresponding labels for visualization.

    Load the saved model's state dictionary to do predict :
      using torch.load() and the specified file path (PATH). 
      Create an instance of the SimpleNet model (loaded_net) and load the state dictionary into it.
      Set the loaded model to evaluation mode using loaded_net.eval().
    
    Visualize the images using Matplotlib:
        fig, axes = plt.subplots(1, 10, figsize=(20, 2)) creates a figure with 1 row and 10 columns of subplots for image visualization.
        for idx, ax in enumerate(axes): iterates over each subplot.
            image = images[idx].squeeze().numpy() converts the image tensor to a NumPy array and removes the single channel dimension.
            ax.imshow(image, cmap='gray') displays the NumPy array as an image in grayscale on the subplot.
            ax.axis('off') removes axis ticks and labels.
            ax.set_title(f'Label: {labels[idx].item()}') sets the title of the subplot to display the corresponding label of the image.
        plt.show() displays the Matplotlib figure containing the visualized images and labels.

 
