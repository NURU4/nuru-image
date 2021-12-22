# NURU image module with AWS SageMaker 

This is a module which creates a spot-the-difference image for playing NURU.

## How does the image module create an image?

The module consists of the opencv2 library and [LaMa inpainter](https://github.com/saic-mdal/lama). First, the module finds some erasable areas using Active Countour and erases them. Then, the module inpaints the erased areas using the LaMa inpainter. Through this procedure, the module can make new images which look natural but different from the original images.

## How does the image module serve the created images?

The image module serves the created images through AWS SageMaker. For more information, visit [this repository](https://github.com/aws-samples/amazon-sagemaker-custom-container).

## Output examples

<img src="https://user-images.githubusercontent.com/70506921/147038984-755488fd-b12e-4c10-aa85-e2612904bcb5.png" width="400" height="400"/> <img src="https://user-images.githubusercontent.com/70506921/147038183-bd659c97-cfb3-4ffa-83b1-505c815f6cce.png" width="400" height="400"/>

<img src="https://user-images.githubusercontent.com/70506921/147038991-c815c15d-ad5c-4ed6-ad3c-94dda244a98b.png" width="400" height="400" /> <img src="https://user-images.githubusercontent.com/70506921/147038179-12375b8d-b5f7-4856-aa09-ae79100dbff1.png" width="400" height="400"/>

<img src="https://user-images.githubusercontent.com/70506921/147038995-a2e423ef-d06d-4077-90d2-87400a4bf76d.png" width="400" height="400"/> <img src="https://user-images.githubusercontent.com/70506921/147038172-7e81d916-cb66-4a71-8b5e-118876738ec0.png" width="400" height="400"/>

## Citation
```
@article{suvorov2021resolution,
  title={Resolution-robust Large Mask Inpainting with Fourier Convolutions},
  author={Suvorov, Roman and Logacheva, Elizaveta and Mashikhin, Anton and Remizova, Anastasia and Ashukha, Arsenii and Silvestrov, Aleksei and Kong, Naejin and Goka, Harshith and Park, Kiwoong and Lempitsky, Victor},
  journal={arXiv preprint arXiv:2109.07161},
  year={2021}
}
```
