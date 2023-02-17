import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name="cellnet",
    version="0.1.0",
    url="https://github.com/theislab/cellnet",
    license='MIT',
    author="Felix Fischer",
    author_email="felix.fischer@helmholtz-muenchen.de",
    description="Scaling single cell models to bigger data sets.",
    long_description=read("README.rst"),
    packages=find_packages(exclude=('tests',)),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
