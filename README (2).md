# **Solving 2D and 3D PDEs using Fourier Neural Operators (FNO)**

This repository contains the final project for the **Scientific Machine Learning** course. It implements a purely data-driven, frequency-domain approach to solve 2D and 3D Partial Differential Equations (PDEs) using **Fourier Neural Operators (FNOs)**.

By learning infinite-dimensional operator mappings instead of grid-dependent functions, this approach achieves **Zero-Shot Super-Resolution** (mesh-independence) and accelerates physical simulations by orders of magnitude compared to traditional numerical solvers like FDM and FEM.

## **✨ Key Features**

* **2D FNO Implementation (fno.py)**:  
  * Takes a 3-channel input ![][image1] and predicts the scalar field ![][image2].  
  * Explicit physical consistency checks (calculates and plots physical Flux vs. Potential).  
  * **Interactive Predictor:** A CLI interface to input arbitrary physical parameter ![][image3] and instantly generate visualizations and CSV data.  
* **3D FNO Implementation (fno3d.py)**:  
  * Scales the architecture to highly complex 9-channel inputs capturing spatial dimensions ![][image4], and Time ![][image5].  
  * Optimized memory management using PyTorch persistent data workers, prefetching, and memory pinning to prevent OOM errors during 3D FFT operations.  
* **Advanced Training Dynamics**:  
  * Implements OneCycleLR and CosineAnnealingLR schedulers to stabilize deep learning in the complex frequency domain.  
* **Comprehensive Evaluation Pipeline**:  
  * Automatically exports predictions to .csv files.  
  * Generates 2D heatmaps, error difference maps, and 3D surface/scatter plots.

## **📁 Repository Structure**

.  
├── fno.py              \# 2D Fourier Neural Operator model, training, and evaluation  
├── fno3d.py            \# 3D Fourier Neural Operator model, optimized dataloader, and 3D plotting  
├── README.md           \# Project documentation  
└── data/               \# (Placeholder) Directory for training and validation datasets

## **⚙️ Requirements**

To run this project, you need the following dependencies installed:

pip install torch torchvision torchaudio  
pip install numpy pandas matplotlib tqdm

*(Note: Code is optimized for CUDA execution. Ensure you have appropriate NVIDIA drivers and PyTorch CUDA builds installed.)*

## **🚀 Usage**

### **1\. Training the 2D FNO Model**

You can train the 2D FNO model by running fno.py. The script accepts several command-line arguments to tune the hyperparameters:

python fno.py \--max\_steps 5000 \--lr 0.001 \--lr\_sch onecycle \--modes1 64 \--modes2 64 \--width 32

**Arguments:**

* \--max\_steps: Total number of training epochs (Default: 5000).  
* \--lr: Initial learning rate (Default: 0.001).  
* \--lr\_sch: Learning rate scheduler, choose between onecycle or cos (Default: onecycle).  
* \--modes1, \--modes2: Number of Fourier modes to keep during frequency domain truncation (Default: 64).  
* \--width: The channel width of the latent space (Default: 32).

### **2\. Interactive 2D Prediction**

Once a model is trained and weights are saved, fno.py automatically enters an interactive mode. You can type an arbitrary physical parameter ![][image3] directly into the terminal, and the script will:

1. Dynamically generate the input grid.  
2. Output a fast inference prediction.  
3. Plot the spatiotemporal heatmap and the **Flux vs. Potential** curve.  
4. Save the results as a CSV file.

### **3\. Training and Evaluating the 3D FNO Model**

The 3D model processes much larger tensors. Run the following command to initiate training and 3D plotting:

python fno3d.py \--max\_steps 300 \--lr 0.001 \--modes1 24 \--modes2 24 \--modes3 24 \--width 32

*Note: Make sure to adjust the num\_workers and train\_dir/val\_dir paths in the if \_\_name\_\_ \== "\_\_main\_\_": block to match your local hardware and dataset paths.*

## **📊 Results & Visualizations**

*(You can drag and drop your generated images here when uploading to GitHub)*

### **2D Prediction & Physics Consistency**

The 2D FNO accurately predicts the PDE solution and strictly adheres to nonlinear conservation laws, as demonstrated by the perfect overlap in the Flux vs. Potential graphs.

\<\!-- Replace the source below with your actual images uploaded to GitHub \--\>

**\[Insert your 2D True vs Predicted Heatmap here\]**

*Left: True Solution | Middle: FNO Prediction | Right: Absolute Error*

**\[Insert your Flux vs Potential plot here\]**

*Flux vs. Potential demonstrating physical consistency.*

### **3D Spatiotemporal Dynamics**

The 3D FNO tracks spatial data across complex boundaries, with discrepancies strictly confined to extreme high-frequency corners.

\<\!-- Replace the source below with your actual images uploaded to GitHub \--\>

**\[Insert your 3D Plot (Surface/Scatter) here\]**

*3D visualization showing True Data, Predicted Solution, and Spatial Error Distribution.*

## **📜 Acknowledgments**

Developed as the final project for **Scientific Machine Learning**. Special thanks to the course staff for the theoretical foundations in operator learning and physics-informed neural networks.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD4AAAAZCAYAAABpaJ3KAAADIUlEQVR4Xu2XvWtUQRTFdzGCoih+rIv79fYLQRAUVhALraJooY2CgnYiNtb6B4h9ilQSCFYWChYSSCGaPoVNJGClEBARsdpG2MRz3t5ZZ8++t++tyYKL+4Mh7517Z+beuTOzL5nMlClTpvwP1Ov1g5VK5ZPqk0a1Wj0TBMFL1ePIwnmLf9XwL1EqlZqI85fFGgsKOIf2SPUBMNAbtIeqjwPM8w1Vea56WtC/xsQxxhW1+dAHC7VX9T7g1KnVannVxwEDQnuq+gjMcAxUdFENPvDZgM8l1XvA4STapurjoFAoHLWg62obBYyxznFU9+GOgM8P1XvA+AptSfVxgHnuJAWcBizcLMcpl8un1OZA4ntsrhm1hcDYZkCqk1artZu7Ae0eWsdWsePe1T8Kq/JHBiEtvhopsDFWVPehD47wadVDaMQKtlTn+YCt7b0/pi8vjMAS4ar6fZKwYLdzvnsgnmWOp7oP4w9iiuqCOR6h8/a97b0vuYXAKl7E840/3skg0EOca7vnm9hOZNxDb3fYv7BgqofEJa7AZzPpJh1GsEPn23bcFj+4LPbP6uNITDypCu6iiDoSaUH/VbSvqo+CSxY7LuA7C2GLGXmBwdZOSnw2Ql+jrVgsHsHted0m6H3Z4X3dc2dQJ/L5/D5f82H/wM63Ve2FZ84imcve+wDoc9jG6B0x9/OIwjzwfR3mf1X1kCDmArBO4a2Iv/N8djZMVMX7E/WvDPnWp50LyGd0v4XnC86GfvdtrrCSiiX9zmLo+6y2eZd9zeHiVz0kiPkdt8QXUPFS0N2mrD6f+cn43fd1R4EtrupB9464W+n+Wiw43fq+tv5v/T7EnWk2Xmpqh75Cm+oupkzc/x/8CGBQqoNdsDX8c43na8M+bXmecrncftUd7M9zqrqDSahG2K/ZbB5Q3cjCfpPb3hfteG742gBMfFhCacE4q6qlpdL9uduxf5SYNNp51fvAtjgHpzXVR6HRaBxD8HOqp8Xmj96WI4JYyhjvp+qRwPEZF0D1tKD/e9XSwrsDW/Os6n8LYulE3QexoGKLcZfTpICkP0x6DmPhN3KD5g+fopujAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAZCAYAAABzVH1EAAAC30lEQVR4Xu1XPWhUQRB+hxFURBE8xPt5e38oioVwkkotRC0ERWKhoL0iVga0tbFLZSCNAbEwFhaHSCRgI1ZCCktBCGglImkERQQTv+/dzN1mfO/dvcudhdwHQ3a/mZ2dnZ2dewmCMcYYY4wkOOd+4k/O8qMG9n1WqVSOWn4ghGH4oVQqNSz/r4DD/K7Varstnwk4xBk4em95BXRfkLHHlh8myuXyEezz1fKZAAcvILcsr4BuHXLf8kPGBPcJNlPadICMb7M8USgU9lKPW6tZ3bDBqkAc1y3fFxDgcThYs7wCuquSqZGDh8Beny3fAYK9I+VxE9KCvMzn8zupY+1jvuzb6y3EyKpvl4Zms7kV9jOQV5BZyHfINOQh5B3exAW7Bvwh7mP5COxGFJ1rUODOy3wl7SGLfeb3IYFf8ubcc8lL0qJvTzC5cpCJDQoEeIMKtNXtymHegsPD3nwV87s69wF+jwSQ6X2gjR7A3ld0zvcnfpqY5jCew98t3RVd0E6rxSfXmJkNpAH1SQdxQ3of8H9a/PTsSHEHidoZZN4nLaD/5GJKR25j2aU9vj4BH4u9EkpoadmDaF1Gb8FHvV4v8zGKzWtuZG2c3IYekuWJ8VPVV6vVk1IqfwE6J2ujJiLjJ6pHqd2Dzb7uijZgs5+2lo8cxPVl8D8gUxzHdS2xmed67S6wu4zxCY7lUAxuXRPiA/wD0X8L2m+CCe2UL+Yr5LsrOnx81+JtUNFoNHZhmpM+/SvwHlqY8DsCboo89NfC9idMp0SRzYMSKOW2v45gxmmPDrUD4zcYL0Deiu45EnLMriEkvo+WV+R4IGx+lmOrDCRj7CxWQXBt0sccazmpUSDYOnxeDCRpLKW4MvfBQ1S8bpcZ2GDJpXxrJYG3yY89yw8I/dYaHPIrnMmJlFvPTtQvWIqQactnBq5+EsE9snwSsGkrqeSyolgslkLv62PTQHCnIOcsP2q4lP+F/hv8AczOzZUGrieiAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAbCAYAAACqenW9AAAA1ElEQVR4XmNgGAU4gLy8/F8FBYVLcnJyYehyWAFIAxD/l5GRkUaXwwBAhSdAimVlZf3Q5TAAUKElEP8E4gfoclgB0M3lINOBTBZ0OQwgKirKA1IM9GwEuhxWAFIMxMvRxbECoFN2QJ2CHxgbG7NCTf4P1KSELo8MmIFurQQqXAbVEI2uAAYYoabdAnGA7PdA/BxdEQiAFQJxMEwAaEM6SAxIc6ArBEXzLGRBIGCBKk6Hi4BMAwkqKSnxIymEyYFs24ouALceGQDdPx0kD2QyosuNbAAAHpg1T72eX9AAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACoAAAAZCAYAAABHLbxYAAABu0lEQVR4Xu2WPUsDQRCGE4ggKohFDBwke3cBQRCbQ0VrKwub/A1rBSt/goWFpBOLtDaChWAbsbYR7WKdyiqY+I7MyuYlF+4DNOA9MOT2nY/Mzt3mUioVFBT8YzzPWzDGjNgajcaJ+PF5yT6uEYfU4FzYLcdxDPvHCMNwWQP79Xp9w/VBe4N26GppQH5HasfV8H1/B/4n1mOxO0LimdV0A30nLDXSoNbusE/A5F+CIDCsx4JCF7ZZWUdRNIfrAS7LFJqWslvXBVob1mJ9Ks7tH2GHm9KkNMtxWUCtZ6mL6YWO1sL62I1LDJJ7tllpnP1ZQb1dbfRO133YPcclBskHsKEUxbLC/jzYAUiDuGPb7E+MPTiway14xDF5kGnqVLPdbsE5OIJ9+IdjQTkx+pyyngZpbOx0Y93V3f88/HnRzfdYTwySP/jgQAu0cNfVGeSt1Wq1RdYZbHhF652yLxFI3DOTT1+i2y8x8oPNOoOYSGP32TeNChJutBGxT9g6+R8d/zus7fi/wRts3sbETRVvpS3EXel3SOzrpFq/gvz5qFarS6zPHCbNH4q/otlsrmKi56zPHJjmA2sFs8oX3AqX0MRmhzgAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAZCAYAAAABmx/yAAAAxElEQVR4XmNgGAU0AnJycoLy8vL/icVwjUBOKxAvRjILJAZTqAkTk5aWFgby/yErOg0ShAtAxMAaQa5BE/+EzLmOJAdyuhJMI7I4CADF9oAZQEXGQGyDJglyOkjjaWRxRUVFcaBYEbIYCgBKPodqjEaXwwugmq5KSUmJoMvhBDD/KSgopKPL4QUw/5FkGwgANT0BaUQXJwig/ruLLo4XAP3FAdW4Bl0OKwAqnAbUtBBIf4Fq/AXkrwTSs4C0Arr6UUArAADkLkOPW29ooQAAAABJRU5ErkJggg==>