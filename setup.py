from setuptools import setup

setup(
    name="guided-diffusion",
    py_modules=["guided_diffusion"],
    install_requires=["blobfile>=1.0.5", "torchvision==0.13","torch==1.12.0", "tqdm","matplotlib", "timm","composer","zipp","torchmetrics==0.10.0","numpy<1.23.0"],
)
