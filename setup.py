import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="webapp-mnist-classifier",
    version="1.0",
    author="Marcel Hinsche, Leon Klein",
    author_email="leon.klein@fu-berlin.de",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'flask',
        'pillow',
        'tensorflow<2.0'
    ],
)
