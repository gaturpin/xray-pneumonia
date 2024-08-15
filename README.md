## Model Architecture  

### CNN_test:  
Image size: 150x150  
- Conv2D (32 filters, 3x3 kernel, ReLU activation) followed by MaxPooling2D (2x2)  
- Conv2D (64 filters, 3x3 kernel, ReLU activation) followed by MaxPooling2D (2x2)  
- Conv2D (128 filters, 3x3 kernel, ReLU activation) followed by MaxPooling2D (2x2)  
- Conv2D (128 filters, 3x3 kernel, ReLU activation) followed by MaxPooling2D (2x2)  
- Flatten Layer: Converts the 3D feature maps into a 1D feature vector.  

#### Fully Connected (Dense) Layers:  
- Dense (512 units, ReLU activation) followed by Dropout (50%) for regularization.  
- Dense (1 unit, Sigmoid activation) for binary output.  

#### Compilation Details:  
- Optimizer: Adam  
- Loss Function: Binary Crossentropy  
- Metrics: Accuracy  

### CNN1:
- CNN_test with image size: 512x512
- 200 epochs

### CNN2:  
- Equivalent to CNN1 with BatchNormalization after each Conv2D layer  