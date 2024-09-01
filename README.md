# Vehicle Detection and License Plate Recognition

This project implements a system for detecting and tracking vehicles in video streams, and reading their license plates. Vehicles are detected using YOLOv8, while license plates are read using OCR.

![Output](https://github.com/user-attachments/assets/0c70b274-d67d-4781-b9db-6193ec72edb4)

## Features
- Vehicle detection and tracking
- License plate detection and recognition
- Real-time FPS and video duration display
  
## Usage

1. **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

# Notes
- Make sure that the necessary libraries and CUDA integration are present for the models in use to function correctly
  ```python
    import torch
    torch.cuda.is_available()
    ```


- When you test the project with a different video, it is likely that you will get inconsistent results. By identifying models that are not suitable for your own projects and training them with a dataset that fits your purpose, you can achieve more accurate results.

## License

This project is licensed under the [MIT License]

[Uploading LICENSE…]()

