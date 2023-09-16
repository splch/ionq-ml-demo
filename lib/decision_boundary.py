import numpy as np
import torch
import matplotlib.pyplot as plt


def dataloader_to_arrays_3d(dataloader):
    X_list, y_list = zip(*[(X.numpy(), y.numpy()) for X, y in dataloader])
    return np.vstack(X_list), np.concatenate(y_list)


def evaluate_on_mesh_3d(model, xx, yy, zz):
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    grid_tensor = torch.tensor(grid.astype("float32"))
    output = model(grid_tensor)
    _, predicted = torch.max(output.data, 1)
    return predicted.numpy().reshape(xx.shape)


def plot_3d_decision_boundary(model, train_loader, test_loader):
    # Convert DataLoader to numpy arrays
    X_train, y_train = dataloader_to_arrays_3d(train_loader)
    X_test, y_test = dataloader_to_arrays_3d(test_loader)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Create a 3D mesh grid
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    z_min, z_max = X_train[:, 2].min() - 1, X_train[:, 2].max() + 1
    xx, yy, zz = np.meshgrid(
        np.linspace(x_min, x_max, 10),
        np.linspace(y_min, y_max, 10),
        np.linspace(z_min, z_max, 10),
    )

    # Evaluate the model on the mesh grid
    Z = evaluate_on_mesh_3d(model, xx, yy, zz)

    # Plot decision boundary and data
    ax.scatter(
        xx[Z == 1],
        yy[Z == 1],
        zz[Z == 1],
        alpha=0.4,
        c="blue",
        label="Good Wine Boundary",
    )
    ax.scatter(
        xx[Z == 0],
        yy[Z == 0],
        zz[Z == 0],
        alpha=0.4,
        c="red",
        label="Bad Wine Boundary",
    )
    # train_scatter = ax.scatter(
    #     X_train[:, 0],
    #     X_train[:, 1],
    #     X_train[:, 2],
    #     c=y_train,
    #     marker="o",
    #     edgecolors="k",
    #     label="Train Data",
    # )
    test_scatter = ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        X_test[:, 2],
        c=y_test,
    )

    ax.set_xlabel("Alcohol")
    ax.set_ylabel("Sulphates")
    ax.set_zlabel("Volatile Acidity", rotation=90)
    ax.zaxis.set_rotate_label(False)
    ax.legend()
    cbar = plt.colorbar(
        test_scatter, ax=ax, orientation="vertical", label="Wine Quality"
    )

    plt.tight_layout()
    plt.show()
